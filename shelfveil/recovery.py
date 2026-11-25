import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from shelfveil.data import (
    KEY_COLUMNS, TRAIN_SHA256, chronological_splits, development_analysis,
    file_hash, hourly_values, iter_daily_batches, iter_series_histories,
    peak_process_memory, save_json, validate_batch,
)


METHODS = ["no_recovery", "availability_scaling", "profile"]
FAMILIES = ["random", "continuous", "peak", "stress"]
LEVELS = ["unsupported", "series_hour_weekday", "series_hour", "product_hour",
          "third_category_hour", "global_hour", "pooled"]
THRESHOLDS = [3, 5, 10, 10, 10, 1]


def new_history(inventory):
    n = len(inventory)
    product_codes, products = pd.factorize(inventory["product_id"], sort=True)
    category_codes, categories = pd.factorize(inventory["third_category_id"], sort=True)
    return {
        "day": 0, "product_codes": product_codes, "category_codes": category_codes,
        "sales_ring": np.zeros((28, n, 24)), "available_ring": np.zeros((28, n, 24), dtype=bool),
        "weekday_ring": np.full(28, -1, dtype=np.int8),
        "series_sum": np.zeros((n, 24)), "series_count": np.zeros((n, 24), dtype=np.int16),
        "weekday_sum": np.zeros((n, 7, 24)), "weekday_count": np.zeros((n, 7, 24), dtype=np.int16),
        "product_sum": np.zeros((len(products), 24)), "product_count": np.zeros((len(products), 24), dtype=np.int32),
        "category_sum": np.zeros((len(categories), 24)), "category_count": np.zeros((len(categories), 24), dtype=np.int32),
        "global_sum": np.zeros(24), "global_count": np.zeros(24, dtype=np.int32),
        "duration_ring": np.zeros((28, 25), dtype=np.int64), "run_ring": np.zeros((28, 25), dtype=np.int64),
        "duration_counts": np.zeros(25, dtype=np.int64), "run_counts": np.zeros(25, dtype=np.int64),
    }


def profile_predictions(history, weekday):
    shape = history["series_sum"].shape
    predictions = np.full(shape, np.nan)
    levels = np.zeros(shape, dtype=np.int8)
    supports = np.zeros(shape, dtype=np.int32)
    pooled_count = int(history["global_count"].sum())
    candidates = [
        (history["weekday_sum"][:, weekday], history["weekday_count"][:, weekday]),
        (history["series_sum"], history["series_count"]),
        (history["product_sum"][history["product_codes"]], history["product_count"][history["product_codes"]]),
        (history["category_sum"][history["category_codes"]], history["category_count"][history["category_codes"]]),
        (np.broadcast_to(history["global_sum"], shape), np.broadcast_to(history["global_count"], shape)),
        (np.broadcast_to(history["global_sum"].sum(), shape), np.broadcast_to(pooled_count, shape)),
    ]
    for level, ((sums, counts), threshold) in enumerate(zip(candidates, THRESHOLDS), start=1):
        use = (levels == 0) & (counts >= threshold)
        predictions[use] = np.maximum(0, sums[use] / counts[use])
        levels[use] = level
        supports[use] = counts[use]
    return predictions, levels, supports


def update_history(history, sales, flags, weekday, day_number):
    if day_number != history["day"] + 1 or sales.shape != history["series_sum"].shape:
        raise ValueError("History must advance once per complete date")
    slot = (day_number - 1) % 28
    available = flags == 0
    observed = np.where(available, sales, 0)
    old_sales = history["sales_ring"][slot]
    old_available = history["available_ring"][slot]
    delta_sales = observed - old_sales
    delta_counts = available.astype(np.int16) - old_available.astype(np.int16)
    history["series_sum"] += delta_sales
    history["series_count"] += delta_counts
    old_weekday = int(history["weekday_ring"][slot])
    if old_weekday >= 0:
        history["weekday_sum"][:, old_weekday] -= old_sales
        history["weekday_count"][:, old_weekday] -= old_available
    history["weekday_sum"][:, weekday] += observed
    history["weekday_count"][:, weekday] += available
    for group in ["product", "category"]:
        np.add.at(history[group + "_sum"], history[group + "_codes"], delta_sales)
        np.add.at(history[group + "_count"], history[group + "_codes"], delta_counts)
    history["global_sum"] += delta_sales.sum(axis=0)
    history["global_count"] += delta_counts.sum(axis=0, dtype=np.int32)
    duration_counts = np.bincount(flags.sum(axis=1).astype(int), minlength=25)
    transitions = np.diff(np.pad(flags.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    start_rows, starts = np.where(transitions == 1)
    end_rows, ends = np.where(transitions == -1)
    if not np.array_equal(start_rows, end_rows):
        raise ValueError("Stockout run accounting failed")
    run_counts = np.bincount(ends - starts, minlength=25)
    history["duration_counts"] += duration_counts - history["duration_ring"][slot]
    history["run_counts"] += run_counts - history["run_ring"][slot]
    history["duration_ring"][slot] = duration_counts
    history["run_ring"][slot] = run_counts
    history["sales_ring"][slot] = observed
    history["available_ring"][slot] = available
    history["weekday_ring"][slot] = weekday
    history["day"] = day_number


def recover_day(observed, flags, predictions, recorded_daily=None):
    if observed.shape != flags.shape or observed.shape != predictions.shape:
        raise ValueError("Hourly inputs must have identical shapes")
    if not np.isfinite(observed).all() or (observed < 0).any() or not np.isin(flags, [0, 1]).all():
        raise ValueError("Invalid recorded sales or stockout flags")
    recorded = observed.sum(axis=1) if recorded_daily is None else np.asarray(recorded_daily)
    available = flags == 0
    available_count = available.sum(axis=1)
    estimates = np.where(np.isfinite(predictions), np.maximum(0, predictions), observed)
    recovered = np.where(available, observed, np.maximum(observed, estimates))
    profile_daily = np.maximum(recorded, recovered.sum(axis=1))
    profile_daily[available_count == 24] = recorded[available_count == 24]
    scaled = np.divide(24 * np.where(available, observed, 0).sum(axis=1), available_count,
                       out=profile_daily.copy(), where=available_count > 0)
    scaled = np.maximum(recorded, scaled)
    scaled[available_count == 24] = recorded[available_count == 24]
    return {"no_recovery": recorded.copy(), "availability_scaling": scaled, "profile": profile_daily,
            "profile_hourly": recovered, "scaling_fallback": available_count == 0}


def mix_uint64(values):
    with np.errstate(over="ignore"):
        values = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> 27)) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> 31)


def case_seeds(identifiers, day_number, family, seed=42):
    values = np.full(len(identifiers), seed, dtype=np.uint64)
    for column in range(3):
        values = mix_uint64(values ^ np.asarray(identifiers[:, column], dtype=np.uint64))
    return mix_uint64(mix_uint64(values ^ np.uint64(day_number)) ^ np.uint64(family + 1))


def historical_peak_order(history):
    means = np.divide(history["global_sum"], history["global_count"],
                      out=np.full(24, -np.inf), where=history["global_count"] > 0)
    return np.argsort(-means, kind="stable")


def make_mask(identifiers, day_number, family, history, truth=None, seed=42):
    values = case_seeds(identifiers, day_number, family, seed)
    lengths = np.arange(2, 7) if family == 1 else np.arange(1, 5)
    weights = history["run_counts"][lengths] if family == 1 else history["duration_counts"][lengths]
    if weights.sum() == 0:
        weights = np.ones(len(lengths), dtype=np.int64)
    uniforms = (values >> 11).astype(np.float64) / 2**53
    durations = lengths[np.searchsorted(np.cumsum(weights), uniforms * weights.sum(), side="right")]
    hours = np.arange(24)
    if family == 1:
        starts = (mix_uint64(values) % (25 - durations).astype(np.uint64)).astype(int)
        mask = (hours >= starts[:, None]) & (hours < (starts + durations)[:, None])
    else:
        if family == 0:
            priorities = mix_uint64(values[:, None] ^ mix_uint64(hours.astype(np.uint64)))
            order = np.argsort(priorities, axis=1, kind="stable")
        elif family == 2:
            order = np.broadcast_to(historical_peak_order(history), (len(values), 24))
        elif family == 3:
            if truth is None or truth.shape != (len(values), 24):
                raise ValueError("Stress masks require scoring truth inside the mask generator")
            order = np.argsort(-truth, axis=1, kind="stable")
        else:
            raise ValueError("Unknown mask family")
        ranks = np.argsort(order, axis=1)
        mask = ranks < durations[:, None]
    return mask, durations.astype(np.int8)


def masked_inputs(observed, mask):
    sales = np.where(mask, 0, observed)
    flags = mask.astype(np.int8)
    unavailable = flags.sum(axis=1).astype(np.int8)
    return {"sales": sales, "flags": flags, "recorded_daily": sales.sum(axis=1),
            "stock_hour6_22_cnt": flags[:, 6:22].sum(axis=1).astype(np.int8),
            "unavailable_hours": unavailable, "fully_available": unavailable == 0,
            "fully_unavailable": unavailable == 24, "unavailable_fraction": unavailable / 24}


def metric_sums(actual, predicted, uplift, groups):
    groups, inverse = np.unique(groups, return_inverse=True)
    error = predicted - actual
    values = [np.ones(len(actual)), actual, np.abs(error), error, error > 0, error < 0,
              actual == 0, predicted, uplift]
    sums = np.column_stack([np.bincount(inverse, weights=value, minlength=len(groups)) for value in values])
    return groups, sums


def add_metrics(accumulator, bank, target, dimension, groups, method, actual, predicted, observed):
    labels, sums = metric_sums(actual, predicted, predicted - observed, groups)
    for label, values in zip(labels, sums):
        key = (bank, target, dimension, str(label), method)
        if key not in accumulator:
            accumulator[key] = np.zeros(9)
        accumulator[key] += values


def metric_rows(accumulator):
    rows = []
    for key, values in sorted(accumulator.items()):
        bank, target, dimension, group, method = key
        n, actual_sum, absolute, signed, over, under, zero, predicted, uplift = values
        row = dict(zip(["bank", "target", "dimension", "group", "method"], key))
        row.update({"count": int(n), "actual_sum": actual_sum, "absolute_error_sum": absolute,
                    "signed_error_sum": signed, "mae": absolute / n, "wape": 100 * absolute / actual_sum if actual_sum > 0 else None,
                    "signed_wpe": 100 * signed / actual_sum if actual_sum > 0 else None,
                    "zero_denominator": actual_sum == 0, "overestimates": int(over), "underestimates": int(under),
                    "zero_actual_count": int(zero), "predicted_sum": predicted, "estimated_missing_sum": uplift})
        for baseline in ["no_recovery", "availability_scaling"]:
            base = accumulator.get((bank, target, dimension, group, baseline))
            row["relative_error_reduction_vs_" + baseline] = 100 * (base[2] - absolute) / base[2] if base is not None and base[2] > 0 else None
            row["wape_reduction_points_vs_" + baseline] = 100 * (base[2] - absolute) / actual_sum if base is not None and actual_sum > 0 else None
        rows.append(row)
    return rows


def add_cluster(clusters, bank, dimension, group, store_codes, actual, predictions, store_count):
    key = (bank, dimension, str(group))
    values = clusters.setdefault(key, np.zeros((store_count, 5)))
    columns = [actual] + [np.abs(predictions[name] - actual) for name in METHODS] + [np.ones(len(actual))]
    for column, weights in enumerate(columns):
        values[:, column] += np.bincount(store_codes, weights=weights, minlength=store_count)


def bootstrap_rows(clusters, replicates=1000, seed=42):
    rows = []
    for key, values in sorted(clusters.items()):
        values = values[values[:, 4] > 0]
        rng = np.random.default_rng(seed)
        sums = values[rng.integers(0, len(values), size=(replicates, len(values)))].sum(axis=1)
        valid = sums[:, 0] > 0
        for method, baseline in [("profile", "no_recovery"), ("profile", "availability_scaling"), ("availability_scaling", "no_recovery")]:
            column = METHODS.index(method) + 1
            base_column = METHODS.index(baseline) + 1
            differences = 100 * (sums[valid, column] - sums[valid, base_column]) / sums[valid, 0]
            point = 100 * (values[:, column].sum() - values[:, base_column].sum()) / values[:, 0].sum() if values[:, 0].sum() > 0 else None
            interval = np.percentile(differences, [2.5, 97.5]) if len(differences) and len(values) >= 2 else [None, None]
            rows.append({"bank": key[0], "dimension": key[1], "group": key[2], "method": method, "baseline": baseline,
                         "wape_difference_points": point, "lower_95": interval[0], "upper_95": interval[1],
                         "stores": len(values), "valid_replicates": int(valid.sum()), "requested_replicates": replicates,
                         "interpretation": "cross-store variability conditional on evaluated dates"})
    return rows


def write_table(writers, path, columns):
    table = pa.table(columns)
    if path not in writers:
        writers[path] = pq.ParquetWriter(path, table.schema, compression="zstd")
    writers[path].write_table(table)


def list_values(values, mask, value_type):
    lengths = mask.sum(axis=1)
    offsets = pa.array(np.r_[0, np.cumsum(lengths)], type=pa.int32())
    return pa.ListArray.from_arrays(offsets, pa.array(values[mask], type=value_type))


def verify_inputs(workspace):
    processed = workspace / "data" / "processed"
    manifest = json.loads((processed / "data_manifest.json").read_text())
    source = workspace / "data" / "raw" / "train.parquet"
    if file_hash(source) != TRAIN_SHA256 or manifest["source_sha256"] != TRAIN_SHA256:
        raise ValueError("Training source hash mismatch")
    for name, artifact in manifest["artifacts"].items():
        if Path(name).name != name or name == "eval.parquet":
            raise ValueError("Invalid training artifact path")
        if file_hash(processed / name) != artifact["sha256"]:
            raise ValueError(f"Processed artifact hash mismatch: {name}")
    inventory = pq.read_table(processed / "series_inventory.parquet").to_pandas().sort_values(KEY_COLUMNS).reset_index(drop=True)
    if len(inventory) != 50000 or inventory.duplicated(KEY_COLUMNS).any():
        raise ValueError("Expected 50000 unique series")
    daily = processed / "daily_sales.parquet"
    if pq.ParquetFile(daily).metadata.num_rows != 4500000:
        raise ValueError("Expected 4500000 daily training rows")
    dates = manifest["training_dates"]
    if json.loads((processed / "split_definitions.json").read_text()) != chronological_splits(dates):
        raise ValueError("Chronological split definitions differ")
    rows = 0
    for table in iter_daily_batches(daily):
        validated = table.set_column(table.schema.get_field_index("dt"), "dt", pc.cast(table["dt"], pa.string()))
        result = validate_batch(validated)
        if result["issues"] or result["count_audit"]["agreement_6_21"] != len(table):
            raise ValueError("Processed training validation failed")
        rows += len(table)
    histories = sum(1 for _ in iter_series_histories(daily))
    if rows != 4500000 or histories != 50000:
        raise ValueError("Incomplete chronological training histories")
    return daily, inventory, dates, manifest


def date_partitions(daily_path, output, batch_size):
    output.mkdir(parents=True, exist_ok=True)
    writers = {}
    columns = KEY_COLUMNS + ["dt", "day_number", "sale_amount", "hours_sale", "hours_stock_status"]
    try:
        for table in iter_daily_batches(daily_path, batch_size, through_day=69):
            table = table.select(columns)
            days = table["day_number"].to_numpy()
            for day_number in np.unique(days):
                path = output / f"day_{day_number:03d}.parquet"
                selected = table.filter(pa.array(days == day_number))
                write_table(writers, path, {name: selected[name] for name in columns})
    finally:
        for writer in writers.values():
            writer.close()


def load_date(path, inventory):
    table = pq.read_table(path).sort_by([(name, "ascending") for name in KEY_COLUMNS])
    frame = table.select(KEY_COLUMNS).to_pandas()
    if len(frame) != len(inventory) or not frame.equals(inventory[KEY_COLUMNS].astype(frame.dtypes.to_dict())):
        raise ValueError("Date partition has missing, duplicated or incorrectly ordered series")
    return table, hourly_values(table["hours_sale"]), hourly_values(table["hours_stock_status"])


def natural_history(table, sales, flags, estimates, levels, supports, day_number):
    recorded = table["sale_amount"].to_numpy()
    results = recover_day(sales, flags, estimates, recorded)
    missing = flags == 1
    unavailable = missing.sum(axis=1)
    fallback_counts = np.column_stack([(missing & (levels == level)).sum(axis=1) for level in range(7)])
    support_minimum = np.where(missing, supports, np.iinfo(np.int32).max).min(axis=1)
    support_minimum[unavailable == 0] = 0
    columns = {name: table[name] for name in KEY_COLUMNS + ["dt", "day_number"]}
    columns.update({"recorded_daily_sales": recorded, "availability_scaling_demand": results["availability_scaling"],
                    "profile_demand": results["profile"], "scaling_missing_demand": results["availability_scaling"] - recorded,
                    "profile_missing_demand": results["profile"] - recorded, "unavailable_hours": unavailable.astype(np.int8),
                    "profile_supported_hours": (unavailable - fallback_counts[:, 0]).astype(np.int8),
                    "profile_unsupported_hours": fallback_counts[:, 0].astype(np.int8),
                    "profile_support_minimum": support_minimum.astype(np.int32),
                    "profile_support_sum": np.where(missing, supports, 0).sum(axis=1),
                    "profile_fallback_counts": pa.FixedSizeListArray.from_arrays(pa.array(fallback_counts.astype(np.int8).reshape(-1)), 7),
                    "scaling_profile_fallback": results["scaling_fallback"],
                    "history_cutoff_day": np.full(len(table), day_number - 1, dtype=np.int16),
                    "history_start_day": np.full(len(table), max(1, day_number - 28), dtype=np.int16)})
    summary = {"day_number": day_number, "daily_rows": len(table), "recorded_sales": float(recorded.sum()),
               "scaling_missing_demand": float(columns["scaling_missing_demand"].sum()),
               "profile_missing_demand": float(columns["profile_missing_demand"].sum()),
               "censored_days": int((unavailable > 0).sum()), "unavailable_hours": int(unavailable.sum()),
               "profile_unsupported_hours": int(fallback_counts[:, 0].sum()),
               "scaling_fallback_days": int(results["scaling_fallback"].sum())}
    summary.update({"hours_" + name: int(fallback_counts[:, level].sum()) for level, name in enumerate(LEVELS)})
    return columns, summary


def evaluate_masks(table, sales, estimates, levels, supports, inventory, history, day_number,
                   bank, window, store_codes, store_count, writers, output, metrics, clusters, fallback):
    identifiers = inventory[KEY_COLUMNS].to_numpy()
    count = len(sales)
    peak = np.zeros(24, dtype=bool)
    peak[historical_peak_order(history)[:4]] = True
    for family, family_name in enumerate(FAMILIES):
        mask, duration = make_mask(identifiers, day_number, family, history, truth=sales if family == 3 else None)
        inputs = masked_inputs(sales, mask)
        predictions = recover_day(inputs["sales"], inputs["flags"], estimates)
        keys = {name: table[name] for name in KEY_COLUMNS + ["dt", "day_number"]}
        keys.update({"bank": pa.array([bank] * count), "family": pa.array([family_name] * count)})
        bits = (mask.astype(np.uint32) * (np.uint32(1) << np.arange(24, dtype=np.uint32))).sum(axis=1, dtype=np.uint32)
        write_table(writers, output / "mask_bank.parquet", keys | {"mask_bits": bits, "duration": duration,
                    "masked_recorded_daily": inputs["recorded_daily"], "stock_hour6_22_cnt": inputs["stock_hour6_22_cnt"],
                    "unavailable_hours": inputs["unavailable_hours"], "fully_available": inputs["fully_available"],
                    "fully_unavailable": inputs["fully_unavailable"], "unavailable_fraction": inputs["unavailable_fraction"]})
        write_table(writers, output / "mask_predictions.parquet", keys | {
            **{name + "_daily": predictions[name] for name in METHODS},
            "profile_hidden_predictions": list_values(predictions["profile_hourly"], mask, pa.float64()),
            "profile_hidden_levels": list_values(levels, mask, pa.int8()),
            "profile_hidden_supports": list_values(supports, mask, pa.int32())})
        # Scoring truth is written separately and never passed into recovery functions.
        actual = table["sale_amount"].to_numpy()
        write_table(writers, output / "mask_labels.parquet", keys | {
            "actual_daily_sales": actual, "actual_hidden_sales": list_values(sales, mask, pa.float64())})
        daily_groups = {"overall": np.zeros(count, dtype=np.int8), "family": np.full(count, family),
                        "duration": duration, "family_duration": family * 10 + duration,
                        "window": np.full(count, window), "family_window": np.full(count, family * 100 + window),
                        "volume_group": inventory["volume_group"].to_numpy(),
                        "stockout_group": inventory["stockout_group"].to_numpy(),
                        "third_category_id": inventory["third_category_id"].to_numpy(),
                        "fully_supported": ((levels > 0) | ~mask).all(axis=1).astype(np.int8)}
        for dimension, groups in daily_groups.items():
            for method in METHODS:
                add_metrics(metrics, bank, "daily", dimension, groups, method, actual, predictions[method], inputs["recorded_daily"])
        actual_hidden = sales[mask]
        hidden_groups = {"overall": np.zeros(len(actual_hidden), dtype=np.int8),
                         "family": np.full(len(actual_hidden), family),
                         "peak_hour": np.broadcast_to(peak, mask.shape)[mask].astype(np.int8),
                         "positive_actual": (actual_hidden > 0).astype(np.int8),
                         "profile_fallback_level": levels[mask], "window": np.full(len(actual_hidden), window),
                         "family_duration": np.repeat(family * 10 + duration, duration)}
        for dimension, groups in hidden_groups.items():
            for method, values in [("no_recovery", np.zeros(len(actual_hidden))), ("profile", predictions["profile_hourly"][mask])]:
                add_metrics(metrics, bank, "hidden", dimension, groups, method, actual_hidden, values, np.zeros(len(actual_hidden)))
        for level in range(7):
            fallback[(bank, family, level)] += int((levels[mask] == level).sum())
        for dimension, group in [("overall", "all"), ("family", family_name), ("window", window)]:
            add_cluster(clusters, bank, dimension, group, store_codes, actual, predictions, store_count)


def verify_recovery_outputs(output, inventory, expected_cases):
    rows = 0
    for day_number, batch in enumerate(pq.ParquetFile(output / "recovered_daily.parquet").iter_batches(batch_size=len(inventory)), 1):
        table = pa.Table.from_batches([batch])
        keys = table.select(KEY_COLUMNS).to_pandas()
        if len(table) != len(inventory) or not keys.equals(inventory[KEY_COLUMNS].astype(keys.dtypes.to_dict())):
            raise ValueError("Recovered history coverage differs from the complete inventory")
        if not np.all(table["day_number"].to_numpy() == day_number) or not np.all(table["history_cutoff_day"].to_numpy() == day_number - 1):
            raise ValueError("Recovered historical cutoff is invalid")
        recorded = table["recorded_daily_sales"].to_numpy()
        for name in ["profile_demand", "availability_scaling_demand"]:
            values = table[name].to_numpy()
            if not np.isfinite(values).all() or (values < recorded).any():
                raise ValueError("Recovered demand violated its observed lower bound")
        rows += len(table)
    if rows != len(inventory) * 69:
        raise ValueError("Incomplete recovered historical population")
    readers = [pq.ParquetFile(output / name) for name in ["mask_bank.parquet", "mask_predictions.parquet", "mask_labels.parquet"]]
    if any(reader.metadata.num_rows != expected_cases for reader in readers):
        raise ValueError("Mask bank, predictions and labels have different coverage")
    count = 0
    sums = {method: np.zeros(4) for method in METHODS}
    from itertools import zip_longest

    for batches in zip_longest(*(reader.iter_batches(batch_size=8192) for reader in readers)):
        if any(batch is None for batch in batches):
            raise ValueError("Unmatched mask evidence batches")
        bank, predictions, labels = [pa.Table.from_batches([batch]) for batch in batches]
        key_columns = KEY_COLUMNS + ["dt", "day_number", "bank", "family"]
        if not bank.select(key_columns).equals(predictions.select(key_columns)) or not bank.select(key_columns).equals(labels.select(key_columns)):
            raise ValueError("Mask scoring keys do not align")
        bits = bank["mask_bits"].to_numpy()
        mask = ((bits[:, None] >> np.arange(24)) & 1).astype(bool)
        if not np.array_equal(mask.sum(axis=1), bank["duration"].to_numpy()):
            raise ValueError("Mask duration differs from its hidden hours")
        if not np.array_equal(mask[:, 6:22].sum(axis=1), bank["stock_hour6_22_cnt"].to_numpy()):
            raise ValueError("Masked stockout count was not recomputed")
        actual = labels["actual_daily_sales"].to_numpy()
        observed = bank["masked_recorded_daily"].to_numpy()
        hidden = np.array([sum(values) for values in labels["actual_hidden_sales"].to_pylist()])
        if not np.allclose(actual, observed + hidden, atol=1e-8, rtol=1e-6):
            raise ValueError("Masked inputs do not reconcile with separate scoring labels")
        for method in METHODS:
            values = predictions[method + "_daily"].to_numpy()
            if (values < observed).any() or not np.isfinite(values).all():
                raise ValueError("Invalid masked recovery output")
            sums[method] += [len(actual), actual.sum(), np.abs(values - actual).sum(), (values - actual).sum()]
        count += len(bank)
    if count != expected_cases:
        raise ValueError("Scoring verification omitted cases")
    return {"historical_rows": rows, "mask_cases": count,
            "rescored_daily_sums": {name: values.tolist() for name, values in sums.items()}}


def run_recovery(workspace, threads=4, batch_size=65536):
    start = perf_counter()
    workspace = Path(workspace).resolve()
    repository = Path(__file__).resolve().parents[1]
    if workspace == repository or repository in workspace.parents:
        raise ValueError("Private workspace must be outside the repository")
    if threads not in range(1, 5) or batch_size < 1:
        raise ValueError("Use one to four threads and a positive batch size")
    try:
        code_commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                              text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = None
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    output = workspace / "data" / "recovery"
    notes = workspace / "notes" / "statistical_recovery"
    output.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)
    protocol_path = notes / "protocol.json"
    if not protocol_path.exists():
        save_json(protocol_path, {"seed": 42, "window_days": 28, "support_thresholds": dict(zip(LEVELS[1:], THRESHOLDS)),
                  "mask_durations": "prior-28-day real count frequencies, 1-4 hours; real within-day run frequencies, 2-6 hours; uniform if unsupported",
                  "mask_order": "stateless SplitMix64 random, uniform continuous start, prior global peak, synthetic truth-ranked stress",
                  "eligibility": "every fully available day in 11-14 and 42-69; every family, no cap"})
    protocol = json.loads(protocol_path.read_text())
    if protocol["seed"] != 42 or protocol["window_days"] != 28 or protocol["support_thresholds"] != dict(zip(LEVELS[1:], THRESHOLDS)):
        raise ValueError("Recorded recovery protocol differs from implementation")
    disk_before = shutil.disk_usage(workspace).free
    daily_path, inventory, dates, input_manifest = verify_inputs(workspace)
    print("Input hashes and complete chronological coverage verified", flush=True)
    analysis_notes = workspace / "notes" / "data_preparation"
    correction = development_analysis(daily_path, inventory, analysis_notes)
    save_json(analysis_notes / "development_analysis.json", correction)
    save_json(notes / "coverage_correction.json", {key: correction[key] for key in [
        "series_without_fully_available_days", "series_without_fully_available_window_targets"]})
    partitions = output / "date_partitions"
    date_partitions(daily_path, partitions, batch_size)
    print("Development date partitions prepared", flush=True)
    history = new_history(inventory)
    cluster_keys = pd.MultiIndex.from_frame(inventory[["city_id", "store_id"]])
    store_codes, stores = pd.factorize(cluster_keys, sort=True)
    metrics, clusters, writers = {}, {}, {}
    fallback = Counter()
    natural_rows, mask_counts, duration_records = [], Counter(), []
    try:
        for day_number in range(1, 70):
            table, sales, flags = load_date(partitions / f"day_{day_number:03d}.parquet", inventory)
            weekday = date.fromisoformat(dates[day_number - 1]).weekday()
            if history["day"] != day_number - 1:
                raise ValueError("Recovery history cutoff differs from prediction date")
            estimates, levels, supports = profile_predictions(history, weekday)
            columns, summary = natural_history(table, sales, flags, estimates, levels, supports, day_number)
            write_table(writers, output / "recovered_daily.parquet", columns)
            natural_rows.append(summary)
            if 11 <= day_number <= 14 or 42 <= day_number <= 69:
                eligible = np.flatnonzero((flags == 0).all(axis=1))
                bank = "early" if day_number <= 14 else "development"
                window = 10 if bank == "early" else 41 + ((day_number - 42) // 7) * 7
                mask_counts[bank] += len(eligible) * 4
                duration_records.append({"day_number": day_number, "history_cutoff": day_number - 1,
                                         "duration_weights": history["duration_counts"][1:5].tolist(),
                                         "continuous_weights": history["run_counts"][2:7].tolist(),
                                         "peak_order": historical_peak_order(history).tolist()})
                evaluate_masks(table.take(pa.array(eligible)), sales[eligible], estimates[eligible], levels[eligible], supports[eligible],
                               inventory.iloc[eligible].reset_index(drop=True), history, day_number, bank, window,
                               store_codes[eligible], len(stores), writers, output, metrics, clusters, fallback)
            # Update all shared statistics only after every prediction for this date.
            update_history(history, sales, flags, weekday, day_number)
            if day_number % 7 == 0 or day_number == 69:
                print(f"Historical days processed: {day_number}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    del history
    expected = {"early": 41786 * 4, "development": 623963 * 4}
    if dict(mask_counts) != expected:
        raise ValueError(f"Controlled eligibility coverage differs: {dict(mask_counts)}")
    rows = metric_rows(metrics)
    pd.DataFrame(rows).to_csv(notes / "recovery_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows(clusters)).to_csv(notes / "paired_bootstrap.csv", index=False)
    pd.DataFrame(natural_rows).to_csv(notes / "natural_recovery_by_day.csv", index=False)
    fallback_rows = []
    for (bank, family, level), count in sorted(fallback.items()):
        denominator = sum(fallback[(bank, family, index)] for index in range(7))
        fallback_rows.append({"bank": bank, "family": FAMILIES[family], "level": level, "fallback": LEVELS[level],
                              "hours": count, "denominator": denominator, "percent": 100 * count / denominator if denominator else None})
    pd.DataFrame(fallback_rows).to_csv(notes / "fallback_frequencies.csv", index=False)
    save_json(notes / "mask_construction_history.json", duration_records)
    cluster_rows = []
    for key, values in sorted(clusters.items()):
        for store, row in enumerate(values):
            if row[4] > 0:
                cluster_rows.append({"bank": key[0], "dimension": key[1], "group": key[2], "city_id": int(stores[store][0]),
                                     "store_id": int(stores[store][1]), "actual_sum": row[0], "no_recovery_error": row[1],
                                     "scaling_error": row[2], "profile_error": row[3], "cases": int(row[4])})
    pd.DataFrame(cluster_rows).to_parquet(notes / "store_error_sums.parquet", index=False)
    verification = verify_recovery_outputs(output, inventory, sum(mask_counts.values()))
    for method in METHODS:
        combined = sum((values[:4] for key, values in metrics.items() if key[1:4] == ("daily", "overall", "0") and key[4] == method), start=np.zeros(4))
        if not np.allclose(combined, verification["rescored_daily_sums"][method], rtol=1e-10, atol=1e-8):
            raise ValueError("Independent rescoring does not match pooled metrics")
    artifacts = {path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size,
                           "schema": str(pq.ParquetFile(path).schema_arrow)} for path in output.glob("*.parquet")}
    result = {"code_commit": code_commit, "input_source_sha256": input_manifest["source_sha256"],
              "input_artifacts": input_manifest["artifacts"], "python": sys.version,
              "numpy_version": np.__version__, "pandas_version": pd.__version__, "pyarrow_version": pa.__version__,
              "protocol_sha256": file_hash(protocol_path), "recovery_source_sha256": file_hash(__file__),
              "historical_rows": len(inventory) * 69, "mask_cases": sum(mask_counts.values()), "bank_cases": dict(mask_counts),
              "elapsed_seconds": perf_counter() - start, "peak_process_memory_bytes": peak_process_memory(),
              "threads": threads, "batch_size": batch_size, "disk_free_before_bytes": disk_before,
              "disk_free_after_bytes": shutil.disk_usage(workspace).free,
              "recovery_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
              "artifacts": artifacts, "verification": verification, "evaluation_file_accessed": False}
    save_json(notes / "recovery_run.json", result)
    return result
