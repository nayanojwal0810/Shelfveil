import builtins
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from shelfveil import data


def daily_row(day="2024-03-28", product=1, city=0, store=0):
    return {
        "city_id": city, "store_id": store, "product_id": product,
        "management_group_id": 0, "first_category_id": 1,
        "second_category_id": 2, "third_category_id": 3,
        "dt": day, "sale_amount": 2.0, "hours_sale": [0.0, 2.0] + [0.0] * 22,
        "hours_stock_status": [0] * 24, "stock_hour6_22_cnt": 0,
        "discount": 1.0, "holiday_flag": 0, "activity_flag": 0,
        "precpt": 0.0, "avg_temperature": 15.0, "avg_humidity": 50.0, "avg_wind_level": 1.0,
    }


def table_from_rows(rows):
    return pa.Table.from_pylist(rows)


def test_missing_column():
    table = table_from_rows([daily_row()]).drop(["city_id"])
    with pytest.raises(ValueError, match="Missing required columns: city_id"):
        data.validate_batch(table)


@pytest.mark.parametrize("name", ["hours_sale", "hours_stock_status"])
def test_array_lengths(name):
    row = daily_row()
    row[name] = row[name][:-1]
    result = data.validate_batch(table_from_rows([row]))
    assert result["issues"][f"invalid_length_{name}"] == 1


@pytest.mark.parametrize("flag", [2, -1, None])
def test_invalid_flags(flag):
    row = daily_row()
    row["hours_stock_status"][5] = flag
    result = data.validate_batch(table_from_rows([row]))
    assert result["issues"]["invalid_stock_flags"] == 1
    assert result["invalid_rows"].tolist() == [True]


def test_null_array_and_context():
    rows = [daily_row(), daily_row(product=2)]
    rows[0]["hours_stock_status"] = None
    rows[0]["avg_temperature"] = None
    issues = data.validate_batch(table_from_rows(rows))["issues"]
    assert issues["null_hours_stock_status"] == 1
    assert issues["nonfinite_avg_temperature"] == 1


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_invalid_sales(value):
    row = daily_row()
    row["sale_amount"] = value
    row["hours_sale"][0] = value
    issues = data.validate_batch(table_from_rows([row]))["issues"]
    assert issues["invalid_daily_sales"] == 1
    assert issues["invalid_hourly_sales"] == 1


def test_sales_reconciliation_and_tolerance():
    row = daily_row()
    row["sale_amount"] += 1e-10
    assert not data.validate_batch(table_from_rows([row]))["issues"]
    row["sale_amount"] += 0.01
    assert data.validate_batch(table_from_rows([row]))["issues"]["daily_hourly_mismatch"] == 1


def test_zeros_and_positive_stockout_sales_are_preserved():
    row = daily_row()
    row["hours_stock_status"][1] = 1
    table = table_from_rows([row])
    assert not data.validate_batch(table)["issues"]
    daily = data.daily_table(table, [row["dt"]])
    hourly = data.hourly_table(daily)
    assert daily["sale_amount"].to_pylist() == [2.0]
    assert daily["fully_available"].to_pylist() == [False]
    assert hourly["sales"].to_pylist() == row["hours_sale"]
    assert hourly["is_stockout"][1].as_py() == 1
    assert hourly["sales"][1].as_py() == 2.0


def test_stock_count_hour_convention():
    row = daily_row()
    row["hours_stock_status"][21:23] = [1, 1]
    row["stock_hour6_22_cnt"] = 1
    audit = data.validate_batch(table_from_rows([row]))["count_audit"]
    assert audit == {"eligible_rows": 1, "agreement_6_21": 1, "agreement_6_22": 0}


def test_duplicates_across_batches_and_missing_dates():
    first = table_from_rows([daily_row()])
    duplicate = table_from_rows([daily_row()])
    report, _ = data.audit_batches([first, duplicate], ["2024-03-28", "2024-03-29"])
    assert report["issues"]["duplicate_series_date"] == 1
    assert report["issues"]["missing_series_dates"] == 1
    assert report["examples"]["duplicate_series_date"]


def test_city_is_part_of_series_key():
    rows = [daily_row(city=0, store=10), daily_row(city=1, store=10)]
    report, states = data.audit_batches([table_from_rows(rows)], ["2024-03-28"])
    assert report["series"] == 2
    assert set(states) == {(0, 10, 1), (1, 10, 1)}
    assert report["issues"]["invalid_hierarchy_mapping"] == 1


def test_hierarchy_and_order_validation():
    rows = [daily_row("2024-03-29"), daily_row()]
    rows[1]["first_category_id"] = 9
    report, _ = data.audit_batches([table_from_rows(rows)], ["2024-03-28", "2024-03-29"])
    assert report["issues"]["invalid_hierarchy_mapping"] == 1
    assert report["issues"]["inconsistent_series_hierarchy"] == 1
    assert report["issues"]["nonchronological_rows"] == 1


def test_hourly_expansion_identifiers_and_values():
    rows = [daily_row(product=1), daily_row(product=2)]
    rows[1]["hours_sale"] = [float(hour) for hour in range(24)]
    rows[1]["sale_amount"] = 276.0
    hourly = data.hourly_table(data.daily_table(table_from_rows(rows), [rows[0]["dt"]]))
    assert len(hourly) == 48
    assert hourly["hour"].to_pylist() == list(range(24)) * 2
    assert hourly["product_id"].to_pylist() == [1] * 24 + [2] * 24
    assert hourly["sales"].to_pylist()[24:] == list(map(float, range(24)))
    assert hourly["sales"].type == pa.float64()


def test_split_boundaries():
    dates = pd.date_range("2024-03-28", periods=90).strftime("%Y-%m-%d").tolist()
    splits = data.chronological_splits(dates)
    assert splits["development"]["end"] == "2024-06-04"
    assert splits["calibration"]["start"] == "2024-06-05"
    assert splits["final_evaluation_expected"]["start"] == "2024-06-26"
    assert splits["final_evaluation_expected"]["end"] == "2024-07-02"
    for window in splits["development_windows"] + splits["calibration_windows"]:
        assert window["target"]["first_day"] == window["origin_day"] + 1
        assert window["target"]["last_day"] == window["origin_day"] + 7
    assert not splits["evaluation_dates_verified"]
    with pytest.raises(ValueError, match="90 consecutive"):
        data.chronological_splits(dates[:-1])


def test_zero_daily_sales_are_valid():
    row = daily_row()
    row["sale_amount"] = 0.0
    row["hours_sale"] = [0.0] * 24
    table = table_from_rows([row])
    assert not data.validate_batch(table)["issues"]
    daily = data.daily_table(table, [row["dt"]])
    assert daily["sale_amount"].to_pylist() == [0.0]
    assert daily["fully_available"].to_pylist() == [True]


def test_training_hash_mismatch_stops_before_processing(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "train.parquet").write_bytes(b"unexpected dataset")
    with pytest.raises(ValueError, match="source hash differs"):
        data.prepare_data(tmp_path)
    assert not (tmp_path / "data" / "processed").exists()


@pytest.fixture
def training_workspace(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    dates = pd.date_range("2024-03-28", periods=90).strftime("%Y-%m-%d").tolist()
    rows = [daily_row(day, product=product) for product in range(3) for day in dates]
    for index, row in enumerate(rows):
        row["sale_amount"] = float(1 + index % 7)
        row["hours_sale"][1] = row["sale_amount"]
    rows[0]["hours_stock_status"][22:] = [1, 1]
    rows[1]["hours_stock_status"][:3] = [1, 1, 1]
    rows[182]["hours_stock_status"] = [1] * 24
    rows[182]["stock_hour6_22_cnt"] = 16
    source = raw / "train.parquet"
    pq.write_table(table_from_rows(rows), source, row_group_size=13)
    (raw / "eval.parquet").write_bytes(b"sealed evaluation sentinel")
    monkeypatch.setattr(data, "TRAIN_SHA256", data.file_hash(source))
    monkeypatch.setattr(data, "EXPECTED_SERIES", 3)
    return tmp_path, dates, rows


def test_complete_pipeline_and_evaluation_protection(training_workspace, monkeypatch):
    workspace, _, _ = training_workspace
    accesses = []

    def guard(function):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                assert Path(path).name != "eval.parquet", "Evaluation file accessed"
                accesses.append(str(path))
            return function(path, *args, **kwargs)
        return call

    monkeypatch.setattr(pq, "ParquetFile", guard(pq.ParquetFile))
    monkeypatch.setattr(pq, "read_table", guard(pq.read_table))
    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(Path, "open", guard(Path.open))
    result = data.prepare_data(workspace, threads=1, batch_size=31)
    assert result["daily_rows"] == 270
    assert result["hourly_rows"] == 6480
    assert result["verification"]["unique_hourly_keys"] == 6480
    assert result["verification"]["series"] == 3
    assert result["verification"]["source_value_parity"] == "exact"
    assert not result["evaluation_file_accessed"]
    assert not (workspace / "data" / "processed" / "hourly_sales.parquet").exists()
    assert data.file_hash(workspace / "data" / "raw" / "train.parquet") == data.TRAIN_SHA256
    assert accesses
    analysis = json.loads((workspace / "notes" / "data_preparation" / "development_analysis.json").read_text())
    assert analysis["daily_rows"] == 207
    assert analysis["unavailable_hours"]["numerator"] == 29
    assert analysis["stockout_run_lengths"]["count"] == 2
    assert analysis["stockout_run_lengths"]["maximum"] == 24
    assert analysis["fully_available_days"]["numerator"] == 204
    assert (workspace / "data" / "raw" / "eval.parquet").stat().st_size == len(b"sealed evaluation sentinel")


def test_batch_size_equivalence_and_global_aggregation(training_workspace):
    workspace, _, rows = training_workspace
    data.prepare_data(workspace, threads=1, batch_size=17)
    output = workspace / "data" / "processed"
    first = pq.read_table(output / "daily_sales.parquet")
    evidence = workspace / "notes" / "data_preparation"
    analysis_first = json.loads((evidence / "development_analysis.json").read_text())
    coverage_first = pd.read_csv(evidence / "target_coverage.csv")
    data.prepare_data(workspace, threads=1, batch_size=97)
    assert first.equals(pq.read_table(output / "daily_sales.parquet"))
    analysis_second = json.loads((evidence / "development_analysis.json").read_text())
    assert analysis_first == analysis_second
    pd.testing.assert_frame_equal(coverage_first, pd.read_csv(evidence / "target_coverage.csv"))
    expected_sales = [row["sale_amount"] for row in rows if row["dt"] <= "2024-06-04"]
    assert analysis_second["daily_recorded_sales"]["mean"] == pytest.approx(np.mean(expected_sales))
    assert analysis_second["daily_recorded_sales"]["percentiles"]["95"] == pytest.approx(np.percentile(expected_sales, 95))


@pytest.mark.parametrize("batch_size", [7, 25, 113])
def test_hourly_batches_retain_every_observation(training_workspace, batch_size):
    workspace, dates, rows = training_workspace
    output = workspace / "daily_sales.parquet"
    data.write_daily(workspace / "data" / "raw" / "train.parquet", output, dates, batch_size=23)
    batches = list(data.iter_hourly_batches(output, batch_size=batch_size))
    assert all(len(batch) <= batch_size for batch in batches)
    hourly = pa.concat_tables(batches).to_pandas()
    assert len(hourly) == len(rows) * 24
    assert not hourly.duplicated(data.KEY_COLUMNS + ["dt", "hour"]).any()
    assert hourly.sales.tolist() == [value for row in rows for value in row["hours_sale"]]
    assert hourly.is_stockout.tolist() == [value for row in rows for value in row["hours_stock_status"]]


def test_complete_histories_across_boundaries(training_workspace):
    workspace, dates, _ = training_workspace
    output = workspace / "daily_sales.parquet"
    data.write_daily(workspace / "data" / "raw" / "train.parquet", output, dates, batch_size=11)
    for batch_size in [1, 89, 91]:
        histories = list(data.iter_series_histories(output, batch_size=batch_size))
        assert len(histories) == 3
        assert all(history["day_number"].to_pylist() == list(range(1, 91)) for _, history in histories)
    history = list(data.iter_series_histories(output, batch_size=7, series_keys=[(0, 0, 1)], through_day=69))
    assert len(history) == 1
    assert history[0][0] == (0, 0, 1)
    assert len(history[0][1]) == 69


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "reappearing"])
def test_history_retrieval_detects_loss_and_duplication(training_workspace, corruption):
    workspace, dates, rows = training_workspace
    if corruption == "missing":
        rows.pop(40)
    elif corruption == "duplicate":
        rows[40] = rows[39].copy()
    else:
        rows.extend(rows[:90])
    output = workspace / "daily_sales.parquet"
    pq.write_table(data.daily_table(table_from_rows(rows), dates), output)
    with pytest.raises(ValueError, match="history|reappeared"):
        list(data.iter_series_histories(output, batch_size=23))


def test_verification_detects_changed_source_values(training_workspace):
    workspace, dates, rows = training_workspace
    rows[0]["hours_sale"][1] += 1
    rows[0]["sale_amount"] += 1
    output = workspace / "daily_sales.parquet"
    pq.write_table(data.daily_table(table_from_rows(rows), dates), output)
    with pytest.raises(ValueError, match="differ from the original"):
        data.verify_processed(workspace / "data" / "raw" / "train.parquet", output, 3)


def test_distribution_uses_global_observation_counts():
    counts = Counter()
    data.update_counts(counts, np.array([100.0]))
    data.update_counts(counts, np.zeros(99))
    summary = data.distribution(counts)
    assert summary["count"] == 100
    assert summary["mean"] == 1
    assert summary["percentiles"]["50"] == 0
    assert summary["percentiles"]["99"] == pytest.approx(1)


def test_old_outputs_are_rejected(training_workspace):
    workspace, _, _ = training_workspace
    output = workspace / "data" / "processed"
    output.mkdir()
    (output / "sample_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="obsolete sampled outputs"):
        data.prepare_data(workspace, threads=1)


def test_hourly_interface_rejects_evaluation_file(tmp_path):
    evaluation = tmp_path / "eval.parquet"
    with pytest.raises(ValueError, match="processed daily_sales"):
        list(data.iter_hourly_batches(evaluation))


def test_reject_workspace_inside_repository():
    with pytest.raises(ValueError, match="outside the repository"):
        data.prepare_data(Path(data.__file__).resolve().parents[1])
