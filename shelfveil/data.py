import hashlib
import json
import sys
import math
import shutil
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


KEY_COLUMNS = ["city_id", "store_id", "product_id"]
HIERARCHY_COLUMNS = [
    "management_group_id", "first_category_id", "second_category_id",
    "third_category_id",
]
CONTEXT_COLUMNS = [
    "discount", "holiday_flag", "activity_flag", "precpt",
    "avg_temperature", "avg_humidity", "avg_wind_level",
]
REQUIRED_COLUMNS = KEY_COLUMNS + HIERARCHY_COLUMNS + [
    "dt", "sale_amount", "hours_sale", "hours_stock_status", "stock_hour6_22_cnt",
] + CONTEXT_COLUMNS
TRAIN_SHA256 = "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d"
SALES_ATOL = 1e-8
SALES_RTOL = 1e-6
EXPECTED_SERIES = 50000


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def training_path(workspace):
    path = Path(workspace).resolve() / "data" / "raw" / "train.parquet"
    if path.resolve().name != "train.parquet":
        raise ValueError("Training input must resolve to train.parquet")
    return path


def require_columns(table):
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.column_names))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def hourly_values(column):
    lengths = pc.list_value_length(column).to_numpy()
    if column.null_count == 0 and np.all(lengths == 24):
        return pc.list_flatten(column).to_numpy().reshape(-1, 24)
    values = np.full((len(column), 24), np.nan)
    for index, row in enumerate(column.to_pylist()):
        if row is not None and len(row) == 24:
            values[index] = np.asarray(row, dtype=float)
    return values


def validate_batch(table):
    require_columns(table)
    issues = {}
    examples = {}
    invalid_rows = np.zeros(len(table), dtype=bool)

    def record(name, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 2:
            mask = mask.any(axis=1)
        count = int(mask.sum())
        if count:
            issues[name] = count
            indexes = np.flatnonzero(mask)[:3]
            examples[name] = [
                {column: table[column][int(index)].as_py() for column in KEY_COLUMNS + ["dt"]}
                for index in indexes
            ]
            invalid_rows[:] |= mask

    for name in REQUIRED_COLUMNS:
        record(f"null_{name}", pc.is_null(table[name]).to_numpy())
    for name in KEY_COLUMNS + HIERARCHY_COLUMNS:
        values = table[name].to_numpy()
        record(f"invalid_{name}", ~np.isfinite(values) | (values < 0) | (values != np.floor(values)))
    dates = pd.to_datetime(table["dt"].to_numpy(), format="%Y-%m-%d", errors="coerce")
    record("invalid_date", dates.isna())
    for name in ["hours_sale", "hours_stock_status"]:
        record(f"invalid_length_{name}", pc.list_value_length(table[name]).to_numpy() != 24)
    sales = hourly_values(table["hours_sale"])
    flags = hourly_values(table["hours_stock_status"])
    daily = table["sale_amount"].to_numpy()
    record("invalid_hourly_sales", ~np.isfinite(sales) | (sales < 0))
    record("invalid_daily_sales", ~np.isfinite(daily) | (daily < 0))
    record("invalid_stock_flags", ~np.isin(flags, [0, 1]))
    for name in CONTEXT_COLUMNS:
        values = table[name].to_numpy()
        record(f"nonfinite_{name}", ~np.isfinite(values))
        if name in ["holiday_flag", "activity_flag"]:
            record(f"invalid_{name}", ~np.isin(values, [0, 1]))
    for name in ["discount", "precpt", "avg_humidity", "avg_wind_level"]:
        record(f"negative_{name}", table[name].to_numpy() < 0)
    published = table["stock_hour6_22_cnt"].to_numpy()
    record("invalid_published_count", ~np.isfinite(published) | (published < 0) | (published > 16) | (published != np.floor(published)))
    total = sales.sum(axis=1, dtype=np.float64)
    # Allow floating arithmetic noise while preserving the original float64 quantities.
    record("daily_hourly_mismatch", ~np.isclose(daily, total, atol=SALES_ATOL, rtol=SALES_RTOL))
    valid_flags = np.isin(flags, [0, 1]).all(axis=1)
    comparisons = {
        "eligible_rows": int(valid_flags.sum()),
        "agreement_6_21": int(((flags[:, 6:22].sum(axis=1) == published) & valid_flags).sum()),
        "agreement_6_22": int(((flags[:, 6:23].sum(axis=1) == published) & valid_flags).sum()),
    }
    count_examples = {}
    for name, end in [("hours_6_21", 22), ("hours_6_22", 23)]:
        sums = flags[:, 6:end].sum(axis=1)
        indexes = np.flatnonzero(valid_flags & (sums != published))[:3]
        count_examples[name] = [
            {**{column: table[column][int(index)].as_py() for column in KEY_COLUMNS + ["dt"]},
             "published": int(published[index]), "hourly_flag_sum": int(sums[index])}
            for index in indexes if np.isfinite(published[index])
        ]
    finite = np.isfinite(total) & np.isfinite(daily)
    maximum_difference = float(np.abs(daily[finite] - total[finite]).max(initial=0))
    return {
        "issues": issues, "examples": examples, "invalid_rows": invalid_rows,
        "count_audit": comparisons, "maximum_sales_difference": maximum_difference,
        "stock_count_examples": count_examples,
    }


def ordered_training_dates(path, batch_size=16384):
    dates = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["dt"]):
        dates.update(batch.column(0).to_pylist())
    if None in dates:
        raise ValueError("Null training dates require review")
    ordered = sorted(dates)
    parsed = [date.fromisoformat(value) for value in ordered]
    if len(parsed) != 90 or any(right - left != timedelta(days=1) for left, right in zip(parsed, parsed[1:])):
        raise ValueError("Expected 90 consecutive official training dates")
    return ordered


def audit_batches(batches, dates):
    day_lookup = {value: index + 1 for index, value in enumerate(dates)}
    issues = Counter()
    comparisons = Counter()
    examples = {}
    count_examples = {"hours_6_21": [], "hours_6_22": []}
    series = {}
    hierarchy_maps = [dict() for _ in range(5)]
    row_count = 0
    maximum_difference = 0.0
    for batch in batches:
        table = pa.Table.from_batches([batch]) if isinstance(batch, pa.RecordBatch) else batch
        result = validate_batch(table)
        issues.update(result["issues"])
        comparisons.update(result["count_audit"])
        for name, rows in result["stock_count_examples"].items():
            count_examples[name].extend(rows[:max(0, 3 - len(count_examples[name]))])
        for name, rows in result["examples"].items():
            examples.setdefault(name, []).extend(rows[:max(0, 3 - len(examples.get(name, [])))])
        row_count += len(table)
        maximum_difference = max(maximum_difference, result["maximum_sales_difference"])
        columns = KEY_COLUMNS + HIERARCHY_COLUMNS + ["dt", "sale_amount"]
        scalar_rows = table.select(columns).to_pandas().itertuples(index=False, name=None)
        flags = hourly_values(table["hours_stock_status"])
        valid_flags = np.isin(flags, [0, 1]).all(axis=1)
        stock_totals = flags.sum(axis=1)
        for index, row in enumerate(scalar_rows):
            city, store, product, management, first, second, third, day_text, sale = row
            if any(pd.isna(value) for value in row):
                continue
            day = day_lookup.get(day_text)
            if day is None:
                issues["unexpected_date"] += 1
                continue
            key = (int(city), int(store), int(product))
            hierarchy = (int(management), int(first), int(second), int(third))
            mappings = [(int(store), int(city)), (int(product), hierarchy),
                        (int(third), hierarchy[:3]), (int(second), hierarchy[:2]),
                        (int(first), int(management))]
            hierarchy_bad = False
            for mapping, (child, parent) in zip(hierarchy_maps, mappings):
                if child in mapping and mapping[child] != parent:
                    hierarchy_bad = True
                else:
                    mapping[child] = parent
            if hierarchy_bad:
                issues["invalid_hierarchy_mapping"] += 1
                examples.setdefault("invalid_hierarchy_mapping", [])
                if len(examples["invalid_hierarchy_mapping"]) < 3:
                    examples["invalid_hierarchy_mapping"].append(dict(zip(columns, row)))
            bit = 1 << (day - 1)
            if key not in series:
                series[key] = {"seen": 0, "last_day": 0, "rows": 0, "invalid": False,
                               "hierarchy": hierarchy, "volume": 0.0, "stockout_hours": 0}
            state = series[key]
            if state["seen"] & bit:
                issues["duplicate_series_date"] += 1
                state["invalid"] = True
                if len(examples.setdefault("duplicate_series_date", [])) < 3:
                    examples["duplicate_series_date"].append({"key": list(key), "dt": day_text})
            if day <= state["last_day"]:
                issues["nonchronological_rows"] += 1
                state["invalid"] = True
            if state["hierarchy"] != hierarchy:
                issues["inconsistent_series_hierarchy"] += 1
                state["invalid"] = True
            state["seen"] |= bit
            state["last_day"] = day
            state["rows"] += 1
            state["invalid"] |= bool(result["invalid_rows"][index]) or hierarchy_bad
            if day <= 14:
                state["volume"] += float(sale)
            if day <= 14 and valid_flags[index]:
                state["stockout_hours"] += int(stock_totals[index])
    expected = (1 << len(dates)) - 1
    missing_records = sum(len(dates) - state["seen"].bit_count() for state in series.values())
    missing_series = [key for key, state in series.items() if state["seen"] != expected]
    if missing_records:
        issues["missing_series_dates"] = missing_records
        examples["missing_series_dates"] = [list(key) for key in missing_series[:3]]
    report = {
        "rows": row_count, "series": len(series), "dates": dates,
        "issues": dict(issues), "examples": examples,
        "series_with_missing_dates": len(missing_series), "stock_count_audit": dict(comparisons),
        "stock_count_examples": count_examples,
        "maximum_absolute_sales_difference": maximum_difference,
        "sales_tolerance": {"atol": SALES_ATOL, "rtol": SALES_RTOL},
    }
    return report, series


def quantile_groups(values):
    edges = np.unique(np.quantile(values, [0, 1 / 3, 2 / 3, 1]))
    if len(edges) == 1:
        return np.zeros(len(values), dtype=np.int8), edges.tolist()
    return np.searchsorted(edges[1:-1], values, side="left").astype(np.int8), edges.tolist()


def chronological_splits(dates):
    parsed = [date.fromisoformat(value) for value in dates]
    if len(parsed) != 90 or any(b - a != timedelta(days=1) for a, b in zip(parsed, parsed[1:])):
        raise ValueError("Expected 90 consecutive training dates")
    all_dates = dates + [(parsed[-1] + timedelta(days=offset)).isoformat() for offset in range(1, 8)]

    def period(first, last):
        return {"first_day": first, "last_day": last, "start": all_dates[first - 1], "end": all_dates[last - 1]}

    def windows(origins):
        return [{"origin_day": origin, "origin_date": all_dates[origin - 1],
                 "target": period(origin + 1, origin + 7)} for origin in origins]

    return {"warmup": period(1, 14), "development": period(1, 69),
            "calibration": period(70, 90), "final_evaluation_expected": period(91, 97),
            "development_windows": windows([41, 48, 55, 62]),
            "calibration_windows": windows([69, 76, 83]), "final_window_expected": windows([90])[0],
            "evaluation_dates_verified": False}


def daily_table(table, dates):
    table = table.select(REQUIRED_COLUMNS).replace_schema_metadata(None)
    flags = hourly_values(table["hours_stock_status"])
    stockout_hours = flags.sum(axis=1).astype(np.int8)
    day_lookup = {value: index + 1 for index, value in enumerate(dates)}
    table = table.set_column(table.schema.get_field_index("dt"), "dt", pc.cast(table["dt"], pa.date32()))
    table = table.append_column("day_number", pa.array([day_lookup[value.isoformat()] for value in table["dt"].to_pylist()], type=pa.int16()))
    for name in KEY_COLUMNS + HIERARCHY_COLUMNS:
        table = table.set_column(table.schema.get_field_index(name), name, pc.cast(table[name], pa.int32(), safe=True))
    table = table.set_column(table.schema.get_field_index("hours_stock_status"), "hours_stock_status", pc.cast(table["hours_stock_status"], pa.list_(pa.int8())))
    table = table.append_column("unavailable_hours", pa.array(stockout_hours))
    table = table.append_column("fully_available", pa.array(stockout_hours == 0))
    table = table.append_column("fully_unavailable", pa.array(stockout_hours == 24))
    return table.append_column("unavailable_fraction", pa.array(stockout_hours / 24, type=pa.float32()))


def hourly_table(daily):
    repeated = pc.take(daily.select(KEY_COLUMNS + HIERARCHY_COLUMNS + ["dt", "day_number"] + CONTEXT_COLUMNS),
                       pa.array(np.repeat(np.arange(len(daily)), 24)))
    repeated = repeated.append_column("hour", pa.array(np.tile(np.arange(24, dtype=np.int8), len(daily))))
    repeated = repeated.append_column("sales", pc.list_flatten(daily["hours_sale"]))
    return repeated.append_column("is_stockout", pc.list_flatten(daily["hours_stock_status"]))


def peak_process_memory():
    if sys.platform == "win32":
        import ctypes
        import struct

        counters = ctypes.create_string_buffer(8 + 8 * struct.calcsize("P"))
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not query(kernel.GetCurrentProcess(), counters, len(counters)):
            raise OSError(ctypes.get_last_error(), "Cannot measure process memory")
        # PROCESS_MEMORY_COUNTERS stores PeakWorkingSetSize after its two DWORD fields.
        return struct.unpack_from("P", counters, 8)[0]
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def ratio(numerator, denominator):
    return {"numerator": int(numerator), "denominator": int(denominator),
            "percent": 100 * int(numerator) / int(denominator) if denominator else None}


def iter_daily_batches(path, batch_size=16384, series_keys=None, through_day=90):
    """Read processed training rows in source order, without opening evaluation data."""
    path = Path(path).resolve()
    if path.name != "daily_sales.parquet":
        raise ValueError("Expected processed daily_sales.parquet")
    if batch_size < 1 or through_day not in range(1, 91):
        raise ValueError("Use a positive batch size and a training cutoff from 1 to 90")
    keys = None if series_keys is None else set(map(tuple, series_keys))
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        mask = table["day_number"].to_numpy() <= through_day
        if keys is not None:
            identifiers = table.select(KEY_COLUMNS).to_pandas().itertuples(index=False, name=None)
            mask &= np.fromiter((key in keys for key in identifiers), dtype=bool, count=len(table))
        if mask.any():
            yield table.filter(pa.array(mask))


def iter_hourly_batches(path, batch_size=98304, series_keys=None, through_day=90):
    """Expand every requested observation on demand, with at most batch_size hourly rows."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    for daily in iter_daily_batches(path, max(1, batch_size // 24), series_keys, through_day):
        hourly = hourly_table(daily)
        for start in range(0, len(hourly), batch_size):
            yield hourly.slice(start, batch_size)


def iter_series_histories(path, batch_size=16384, series_keys=None, through_day=90):
    """Yield complete chronological histories, carrying the final series across batches."""
    pending = None
    completed = set()

    def check_history(history):
        key = tuple(history[name][0].as_py() for name in KEY_COLUMNS)
        if key in completed:
            raise ValueError("A series reappeared after its history ended")
        if not np.array_equal(history["day_number"].to_numpy(), np.arange(1, through_day + 1)):
            raise ValueError("Incomplete or duplicated series history")
        completed.add(key)
        return key, history

    for batch in iter_daily_batches(path, batch_size, series_keys, through_day):
        table = batch if pending is None else pa.concat_tables([pending, batch])
        identifiers = np.column_stack([table[name].to_numpy() for name in KEY_COLUMNS])
        changes = np.flatnonzero(np.any(identifiers[1:] != identifiers[:-1], axis=1)) + 1
        start = 0
        for end in changes:
            yield check_history(table.slice(start, int(end) - start))
            start = int(end)
        pending = table.slice(start)
    if pending is not None:
        yield check_history(pending)
    if series_keys is not None and completed != set(map(tuple, series_keys)):
        raise ValueError("Requested series missing from processed data")


def series_inventory(states):
    rows = []
    for key, state in sorted(states.items()):
        rows.append(dict(zip(KEY_COLUMNS + HIERARCHY_COLUMNS, key + state["hierarchy"])) | {
            "initial_sales_volume": state["volume"],
            "initial_stockout_fraction": state["stockout_hours"] / (14 * 24),
        })
    inventory = pd.DataFrame(rows)
    inventory["volume_group"], volume_edges = quantile_groups(inventory["initial_sales_volume"])
    inventory["stockout_group"], stockout_edges = quantile_groups(inventory["initial_stockout_fraction"])
    return inventory, {"history_days": [1, 14], "volume_edges": volume_edges,
                       "stockout_edges": stockout_edges, "purpose": "descriptive coverage groups only",
                       "ties": "duplicate tertile edges removed; ties remain in the lower group"}


def write_daily(source, output, dates, batch_size):
    writer = None
    rows = 0
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_size, columns=REQUIRED_COLUMNS):
            table = daily_table(pa.Table.from_batches([batch]), dates)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(table)
    finally:
        if writer is not None:
            writer.close()
    return rows


def verify_processed(source, daily_path, expected_series):
    from itertools import zip_longest

    daily_rows = 0
    raw_batches = pq.ParquetFile(source).iter_batches(batch_size=4093, columns=REQUIRED_COLUMNS)
    processed_batches = pq.ParquetFile(daily_path).iter_batches(batch_size=4093)
    for raw, processed in zip_longest(raw_batches, processed_batches):
        if raw is None or processed is None:
            raise ValueError("Source and processed batch coverage differs")
        original = pa.Table.from_batches([raw])
        restored = pa.Table.from_batches([processed]).select(REQUIRED_COLUMNS).cast(original.schema)
        if not original.equals(restored):
            raise ValueError("Processed values differ from the original training records")
        daily_rows += len(original)
    histories = sum(1 for _ in iter_series_histories(daily_path, batch_size=4093))
    if histories != expected_series or daily_rows != expected_series * 90:
        raise ValueError("Complete training coverage was not preserved")
    hourly_rows = 0
    raw_batches = pq.ParquetFile(source).iter_batches(batch_size=4096, columns=KEY_COLUMNS + ["hours_sale", "hours_stock_status"])
    for raw, hourly in zip_longest(raw_batches, iter_hourly_batches(daily_path, batch_size=4096 * 24)):
        if raw is None or hourly is None or len(hourly) != len(raw) * 24:
            raise ValueError("Hourly batch coverage differs from the source")
        raw = pa.Table.from_batches([raw])
        if not np.array_equal(hourly["hour"].to_numpy(), np.tile(np.arange(24), len(raw))):
            raise ValueError("Missing or duplicated hourly positions")
        for name in KEY_COLUMNS:
            if not np.array_equal(hourly[name].to_numpy(), np.repeat(raw[name].to_numpy(), 24)):
                raise ValueError("Hourly identifiers do not match source histories")
        if not np.array_equal(hourly["sales"].to_numpy(), hourly_values(raw["hours_sale"]).reshape(-1)):
            raise ValueError("Hourly sales differ from original observations")
        if not np.array_equal(hourly["is_stockout"].to_numpy(), hourly_values(raw["hours_stock_status"]).reshape(-1)):
            raise ValueError("Hourly stockout flags differ from the source")
        hourly_rows += len(hourly)
    if hourly_rows != daily_rows * 24:
        raise ValueError("Hourly observation coverage is incomplete")
    return {"daily_rows": daily_rows, "hourly_rows": hourly_rows, "series": histories,
            "source_value_parity": "exact", "unique_daily_keys": daily_rows,
            "unique_hourly_keys": hourly_rows, "complete_history_check": "passed"}


def update_counts(counts, values):
    unique, frequencies = np.unique(values, return_counts=True)
    counts.update({float(value): int(count) for value, count in zip(unique, frequencies)})


def distribution(counts):
    if not counts:
        return {"count": 0}
    values = np.array(sorted(counts), dtype=np.float64)
    frequencies = np.array([counts[value] for value in values], dtype=np.int64)
    cumulative = np.cumsum(frequencies)
    n = int(cumulative[-1])
    percentiles = {}
    for percentile in [5, 25, 50, 75, 90, 95, 99]:
        rank = (n - 1) * percentile / 100
        lower, upper = math.floor(rank), math.ceil(rank)
        low_value = values[np.searchsorted(cumulative, lower + 1)]
        high_value = values[np.searchsorted(cumulative, upper + 1)]
        percentiles[str(percentile)] = float(low_value + (high_value - low_value) * (rank - lower))
    return {"count": n, "mean": math.fsum(float(value) * int(count) for value, count in zip(values, frequencies)) / n,
            "minimum": float(values[0]), "maximum": float(values[-1]), "percentiles": percentiles}


def stockout_runs(flags, identifiers, days, state, counts):
    boundary = np.r_[True, np.any(identifiers[1:] != identifiers[:-1], axis=1)]
    boundary[1:] |= np.diff(days) != 1
    if state.get("last_key") == tuple(identifiers[0]) and state["last_day"] + 1 == days[0]:
        boundary[0] = False
    flat = flags.reshape(-1).astype(bool)
    breaks = np.zeros(len(flat), dtype=bool)
    breaks[np.flatnonzero(boundary) * 24] = True
    starts = np.flatnonzero(flat & (~np.r_[False, flat[:-1]] | breaks))
    ends = np.flatnonzero(flat & (~np.r_[flat[1:], False] | np.r_[breaks[1:], True]))
    durations = ends - starts + 1
    pending = state.get("open_run", 0)
    if pending:
        if not boundary[0] and flat[0]:
            durations[0] += pending
        else:
            counts[pending] += 1
    state["open_run"] = int(durations[-1]) if flat[-1] else 0
    if flat[-1]:
        durations = durations[:-1]
    for duration, count in zip(*np.unique(durations, return_counts=True)):
        counts[int(duration)] += int(count)
    state["last_key"] = tuple(identifiers[-1])
    state["last_day"] = int(days[-1])


def update_coverage(frame, coverage):
    targets = frame[frame["day_number"].between(42, 69)].copy()
    if targets.empty:
        return
    targets["origin"] = 41 + ((targets["day_number"] - 42) // 7) * 7
    targets["horizon"] = targets["day_number"] - targets["origin"]
    targets["all"] = "all"
    targets["horizon_stockout_volume"] = targets[["horizon", "stockout_group", "volume_group"]].astype(str).agg(":".join, axis=1)
    for dimension in ["all", "horizon", "stockout_group", "volume_group", "third_category_id", "horizon_stockout_volume"]:
        grouped = targets.groupby(["origin", dimension], sort=True)["fully_available"].agg(["sum", "size"])
        for (origin, group), row in grouped.iterrows():
            key = (int(origin), dimension, str(group))
            counts = coverage.setdefault(key, [0, 0])
            counts[0] += int(row["sum"])
            counts[1] += int(row["size"])


def development_analysis(daily_path, inventory, output, batch_size=16384):
    totals = Counter()
    daily_counts, hourly_counts, run_counts = Counter(), Counter(), Counter()
    hour_counts = [Counter() for _ in range(24)]
    available_counts = np.zeros(24, dtype=np.int64)
    unavailable_counts = np.zeros(24, dtype=np.int64)
    weekday_counts = {"weekday": Counter(), "weekend": Counter()}
    weekday_hours = {"weekday": [0, 0.0], "weekend": [0, 0.0]}
    state, coverage, categories = {}, {}, {}
    series = inventory.copy()
    for name in ["history_days", "available_days", "unavailable_hours"]:
        series[name] = np.zeros(len(series), dtype=np.int32)
    series["total_sales"] = 0.0
    lookup = {key: index for index, key in enumerate(series[KEY_COLUMNS].itertuples(index=False, name=None))}
    history_days = np.zeros(len(series), dtype=np.int32)
    available_days = np.zeros(len(series), dtype=np.int32)
    available_target_days = np.zeros(len(series), dtype=np.int32)
    unavailable_hours = np.zeros(len(series), dtype=np.int32)
    total_sales = np.zeros(len(series), dtype=np.float64)
    for table in iter_daily_batches(daily_path, batch_size=batch_size, through_day=69):
        sales = hourly_values(table["hours_sale"])
        flags = hourly_values(table["hours_stock_status"])
        available = flags == 0
        frame = table.drop(["hours_sale", "hours_stock_status"]).to_pandas()
        frame = frame.merge(inventory[KEY_COLUMNS + ["volume_group", "stockout_group"]], on=KEY_COLUMNS, validate="many_to_one", sort=False)
        identifiers = np.column_stack([frame[name].to_numpy() for name in KEY_COLUMNS])
        indexes = np.fromiter((lookup[tuple(key)] for key in identifiers), dtype=np.int64, count=len(frame))
        days = frame["day_number"].to_numpy()
        fully_available = frame["fully_available"].to_numpy()
        stock_totals = flags.sum(axis=1)
        np.add.at(history_days, indexes, 1)
        np.add.at(available_days, indexes, fully_available)
        np.add.at(unavailable_hours, indexes, stock_totals)
        np.add.at(total_sales, indexes, frame["sale_amount"].to_numpy())
        totals["daily_rows"] += len(table)
        totals["fully_available_days"] += int(fully_available.sum())
        totals["fully_unavailable_days"] += int((stock_totals == 24).sum())
        totals["positive_stockout_sales"] += int(((sales > 0) & ~available).sum())
        totals["zero_available_sales"] += int(((sales == 0) & available).sum())
        totals["available_days_11_14"] += int(fully_available[(days >= 11) & (days <= 14)].sum())
        totals["available_hours_1_10"] += int(available[days <= 10].sum())
        totals["available_hours_1_14"] += int(available[days <= 14].sum())
        window = (days >= 42) & (days <= 69)
        np.add.at(available_target_days, indexes, fully_available & window)
        totals["window_targets"] += int(window.sum())
        totals["eligible_window_targets"] += int(fully_available[window].sum())
        update_counts(daily_counts, frame["sale_amount"].to_numpy())
        update_counts(hourly_counts, sales)
        available_counts += available.sum(axis=0)
        unavailable_counts += flags.sum(axis=0)
        for hour in range(24):
            update_counts(hour_counts[hour], sales[:, hour][available[:, hour]])
        stockout_runs(flags, identifiers, days, state, run_counts)
        update_coverage(frame, coverage)
        grouped = frame.groupby(HIERARCHY_COLUMNS, sort=True).agg(rows=("sale_amount", "size"), sales=("sale_amount", "sum"), available=("fully_available", "sum"))
        for key, row in grouped.iterrows():
            counts = categories.setdefault(key, [0, 0.0, 0])
            counts[0] += int(row["rows"])
            counts[1] += float(row["sales"])
            counts[2] += int(row["available"])
        weekend = pd.to_datetime(frame["dt"]).dt.day_of_week.to_numpy() >= 5
        for label, mask in [("weekday", ~weekend), ("weekend", weekend)]:
            update_counts(weekday_counts[label], frame.loc[mask, "sale_amount"].to_numpy())
            weekday_hours[label][0] += int(available[mask].sum())
            weekday_hours[label][1] += float(sales[mask][available[mask]].sum())
    if state.get("open_run", 0):
        run_counts[state["open_run"]] += 1
    n = totals["daily_rows"]
    unavailable = int(unavailable_counts.sum())
    available = int(available_counts.sum())
    if sum(hours * count for hours, count in run_counts.items()) != unavailable:
        raise ValueError("Stockout runs do not reconcile with unavailable hours")
    if not np.all(history_days == 69) or n != len(inventory) * 69:
        raise ValueError("Development analysis omitted or duplicated history")
    for name, values in [("history_days", history_days), ("available_days", available_days),
                         ("available_target_days", available_target_days),
                         ("unavailable_hours", unavailable_hours), ("total_sales", total_sales)]:
        series[name] = values
    series["available_hours"] = history_days * 24 - unavailable_hours
    series.to_parquet(output / "series_coverage.parquet", index=False)
    available_histogram = Counter()
    hour_rows = []
    for hour in range(24):
        summary = distribution(hour_counts[hour])
        available_histogram.update(hour_counts[hour])
        hour_rows.append({"hour": hour, "available_count": int(available_counts[hour]), "denominator": n,
                          "unavailable_count": int(unavailable_counts[hour]), "unavailable_percent": 100 * unavailable_counts[hour] / n,
                          "available_sales_mean": summary.get("mean"),
                          "available_sales_median": summary.get("percentiles", {}).get("50")})
    hour_frame = pd.DataFrame(hour_rows)
    hour_frame.to_csv(output / "hourly_patterns.csv", index=False)
    peak_hours = hour_frame.sort_values(["available_sales_mean", "hour"], ascending=[False, True]).head(4)["hour"].tolist()
    coverage_rows = [{"origin_day": key[0], "dimension": key[1], "group": key[2], "eligible_count": value[0],
                      "denominator": value[1], "eligible_percent": 100 * value[0] / value[1]}
                     for key, value in sorted(coverage.items())]
    pd.DataFrame(coverage_rows).to_csv(output / "target_coverage.csv", index=False)
    pd.DataFrame([dict(zip(HIERARCHY_COLUMNS, key)) | {"daily_count": value[0], "daily_denominator": n,
                  "recorded_sales": value[1], "fully_available_days": value[2]}
                  for key, value in sorted(categories.items())]).to_csv(output / "category_distribution.csv", index=False)
    pd.DataFrame([{"hours": hours, "runs": count, "total_runs": sum(run_counts.values())}
                  for hours, count in sorted(run_counts.items())]).to_csv(output / "stockout_durations.csv", index=False)
    pd.DataFrame([{"group": label, "daily_count": sum(weekday_counts[label].values()),
                  "recorded_daily_mean": distribution(weekday_counts[label]).get("mean"),
                  "available_hourly_count": weekday_hours[label][0],
                  "available_hourly_mean": weekday_hours[label][1] / weekday_hours[label][0] if weekday_hours[label][0] else None}
                  for label in weekday_counts]).to_csv(output / "weekday_patterns.csv", index=False)
    return {
        "analysis_days": [1, 69], "series": len(inventory),
        "identifier_counts": {name: int(inventory[name].nunique()) for name in KEY_COLUMNS + HIERARCHY_COLUMNS},
        "daily_rows": n, "hourly_rows": n * 24,
        "unavailable_hours": ratio(unavailable, n * 24),
        "fully_available_days": ratio(totals["fully_available_days"], n),
        "partially_unavailable_days": ratio(n - totals["fully_available_days"] - totals["fully_unavailable_days"], n),
        "fully_unavailable_days": ratio(totals["fully_unavailable_days"], n),
        "positive_sales_during_stockout": ratio(totals["positive_stockout_sales"], unavailable),
        "zero_sales_while_available": ratio(totals["zero_available_sales"], available),
        "peak_hours": peak_hours, "peak_hour_stockouts": ratio(unavailable_counts[peak_hours].sum(), n * len(peak_hours)),
        "stockout_run_lengths": distribution(run_counts), "run_boundaries": "continuous within series; censored at development boundaries",
        "daily_recorded_sales": distribution(daily_counts), "hourly_recorded_sales": distribution(hourly_counts),
        "available_hourly_sales": distribution(available_histogram), "eligible_recovery_hours": available,
        "available_hours_days_1_10": totals["available_hours_1_10"], "available_hours_days_1_14": totals["available_hours_1_14"],
        "fully_available_days_11_14": ratio(totals["available_days_11_14"], len(inventory) * 4),
        "development_window_targets": ratio(totals["eligible_window_targets"], totals["window_targets"]),
        "series_without_fully_available_days": ratio((available_days == 0).sum(), len(inventory)),
        "series_without_fully_available_window_targets": ratio((available_target_days == 0).sum(), len(inventory)),
        "history_days_per_series": distribution(Counter(history_days)),
        "available_days_per_series": distribution(Counter(available_days)),
        "available_hours_per_series": distribution(Counter(series["available_hours"])),
        "series_total_sales": distribution(Counter(total_sales)),
        "history_group_definition": "tertiles of days 1-14 across every series, for descriptive coverage only",
        "interpretation": "recorded normalized sales; stockout-affected observations are not latent demand; no causal claims",
    }


def prepare_data(workspace, threads=4, batch_size=16384):
    start = perf_counter()
    if threads not in range(1, 5) or batch_size < 1:
        raise ValueError("Use one to four threads and a positive batch size")
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    workspace = Path(workspace).resolve()
    repository = Path(__file__).resolve().parents[1]
    if workspace == repository or repository in workspace.parents:
        raise ValueError("Private workspace must be outside the repository")
    disk_before = shutil.disk_usage(workspace).free
    source = training_path(workspace)
    source_hash = file_hash(source)
    if source_hash != TRAIN_SHA256:
        raise ValueError("Training source hash differs from the approved dataset")
    require_columns(pa.Table.from_batches([], schema=pq.ParquetFile(source).schema_arrow))
    output = workspace / "data" / "processed"
    notes = workspace / "notes" / "data_preparation"
    output.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)
    if (output / "sample_manifest.json").exists() or (output / "hourly_sales.parquet").exists():
        raise ValueError("Isolate obsolete sampled outputs before running complete-data preparation")
    dates = ordered_training_dates(source, batch_size)
    report, states = audit_batches(pq.ParquetFile(source).iter_batches(batch_size=batch_size, columns=REQUIRED_COLUMNS), dates)
    report["source_schema"] = str(pq.ParquetFile(source).schema_arrow)
    save_json(notes / "training_validation.json", report)
    if report["issues"] or report["rows"] != EXPECTED_SERIES * 90 or report["series"] != EXPECTED_SERIES:
        raise ValueError("Training quality or coverage issue requires review; no records were excluded")
    counts = report["stock_count_audit"]
    if counts["agreement_6_21"] != counts["eligible_rows"]:
        raise ValueError("Published stockout count differs from hours 6-21; review required")
    inventory, groups = series_inventory(states)
    del states
    inventory.to_parquet(output / "series_inventory.parquet", index=False)
    save_json(output / "split_definitions.json", chronological_splits(dates))
    print("Complete training validation passed", flush=True)
    daily_path = output / "daily_sales.parquet"
    daily_rows = write_daily(source, daily_path, dates, batch_size)
    print("Daily records written", flush=True)
    verification = verify_processed(source, daily_path, EXPECTED_SERIES)
    print("Complete history and hourly batch verification passed", flush=True)
    analysis = development_analysis(daily_path, inventory, notes, batch_size)
    save_json(notes / "development_analysis.json", analysis)
    if file_hash(source) != source_hash:
        raise ValueError("Training source changed during preparation")
    artifacts = {name: {"sha256": file_hash(output / name), "bytes": (output / name).stat().st_size}
                 for name in ["daily_sales.parquet", "series_inventory.parquet", "split_definitions.json"]}
    manifest = {"source_sha256": source_hash, "coverage": verification, "training_dates": dates,
                "hourly_storage": "generated on demand by iter_hourly_batches; no expanded hourly file",
                "history_interface": "iter_series_histories; complete histories carried across batches",
                "descriptive_groups": groups, "artifacts": artifacts, "evaluation_file_accessed": False}
    save_json(output / "data_manifest.json", manifest)
    result = {"series": EXPECTED_SERIES, "daily_rows": daily_rows, "hourly_rows": verification["hourly_rows"],
              "source_sha256": source_hash, "threads": threads, "batch_size": batch_size,
              "elapsed_seconds": perf_counter() - start, "peak_process_memory_bytes": peak_process_memory(),
              "disk_free_before_bytes": disk_before, "disk_free_after_bytes": shutil.disk_usage(workspace).free,
              "processed_bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
              "artifacts": artifacts, "verification": verification,
              "evaluation_file_accessed": False, "python": sys.version}
    save_json(notes / "preparation_run.json", result)
    return result
