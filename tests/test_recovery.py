import copy
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from shelfveil import data, recovery


def inventory():
    return pd.DataFrame({"city_id": [0, 1], "store_id": [0, 1], "product_id": [5, 5],
                         "third_category_id": [7, 7], "volume_group": [0, 1], "stockout_group": [1, 0]})


def test_common_recovery_lower_bounds_and_no_double_counting():
    observed = np.zeros((1, 24))
    observed[0, :5] = [5, 8, 2, 0, 6]
    flags = np.zeros((1, 24), dtype=np.int8)
    flags[0, 2:4] = 1
    predictions = np.zeros((1, 24))
    predictions[0, 2:4] = [12, 10]
    result = recovery.recover_day(observed, flags, predictions)
    assert result["no_recovery"][0] == 21
    assert result["profile"][0] == 41
    assert result["profile"][0] - result["no_recovery"][0] == 20
    assert np.array_equal(result["profile_hourly"][flags == 0], observed[flags == 0])
    predictions[:] = -1
    assert np.array_equal(recovery.recover_day(observed, flags, predictions)["profile_hourly"], observed)


def test_scaling_and_all_unavailable_fallback():
    observed = np.zeros((1, 24))
    observed[0, :2] = [5, 8]
    flags = np.ones((1, 24), dtype=np.int8)
    flags[0, :2] = 0
    result = recovery.recover_day(observed, flags, np.full((1, 24), 2.0))
    assert result["availability_scaling"][0] == 156
    assert not result["scaling_fallback"][0]
    flags[:] = 1
    result = recovery.recover_day(observed, flags, np.full((1, 24), 2.0))
    assert result["scaling_fallback"][0]
    assert result["availability_scaling"][0] == result["profile"][0] == 57
    unsupported = recovery.recover_day(observed, flags, np.full((1, 24), np.nan))
    assert unsupported["profile"][0] == unsupported["availability_scaling"][0] == 13


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5, 6])
def test_each_fallback_and_support_boundary(level):
    state = recovery.new_history(inventory())
    threshold = recovery.THRESHOLDS[level - 1]
    if level == 1:
        sums, counts = state["weekday_sum"][:, 0], state["weekday_count"][:, 0]
    elif level == 2:
        sums, counts = state["series_sum"], state["series_count"]
    elif level in [3, 4]:
        name = "product" if level == 3 else "category"
        sums, counts = state[name + "_sum"], state[name + "_count"]
    else:
        sums, counts = state["global_sum"], state["global_count"]
    sums[...] = 2 * threshold
    counts[...] = threshold
    predictions, levels, supports = recovery.profile_predictions(state, 0)
    assert (predictions == 2).all()
    assert (levels == level).all()
    expected = threshold * 24 if level == 6 else threshold
    assert (supports == expected).all()
    if level < 6:
        counts[...] = threshold - 1
        _, below_levels, _ = recovery.profile_predictions(state, 0)
        assert not (below_levels == level).any()


def test_available_zeros_included_unavailable_positive_sales_excluded():
    state = recovery.new_history(inventory())
    sales = np.zeros((2, 24))
    sales[1] = 100
    flags = np.zeros((2, 24), dtype=np.int8)
    flags[1] = 1
    for day in range(1, 6):
        recovery.update_history(state, sales, flags, (day - 1) % 7, day)
    predictions, levels, supports = recovery.profile_predictions(state, 5)
    assert (state["series_count"][0] == 5).all()
    assert (state["series_count"][1] == 0).all()
    assert (state["global_sum"] == 0).all()
    assert (predictions == 0).all()
    assert (levels[0] == 2).all()
    assert (supports[0] == 5).all()


def test_rolling_expiry_and_same_date_update_ordering():
    state = recovery.new_history(inventory())
    predictions, levels, _ = recovery.profile_predictions(state, 0)
    assert np.isnan(predictions).all() and (levels == 0).all()
    flags = np.zeros((2, 24), dtype=np.int8)
    for day in range(1, 29):
        sales = np.full((2, 24), 100.0 if day == 1 else 1.0)
        recovery.update_history(state, sales, flags, (day - 1) % 7, day)
    assert (state["series_sum"] == 127).all()
    recovery.update_history(state, np.ones((2, 24)), flags, 0, 29)
    assert (state["series_sum"] == 28).all()
    assert (state["series_count"] == 28).all()
    predictions, _, _ = recovery.profile_predictions(state, 1)
    assert (predictions == 1).all()
    with pytest.raises(ValueError, match="once per complete date"):
        recovery.update_history(state, np.ones((2, 24)), flags, 0, 29)


def test_shared_statistics_do_not_see_another_series_future():
    state = recovery.new_history(inventory())
    before = recovery.profile_predictions(state, 0)[0]
    sales = np.full((2, 24), 1000.0)
    flags = np.ones((2, 24), dtype=np.int8)
    flags[0] = 0
    assert np.isnan(before).all()
    recovery.update_history(state, sales, flags, 0, 1)
    after = recovery.profile_predictions(state, 1)[0]
    assert np.isfinite(after).all()
    assert np.isnan(before).all()
    assert (state["global_count"] == 1).all()


@pytest.mark.parametrize("family", range(4))
def test_deterministic_masks_and_batch_independence(family):
    state = recovery.new_history(inventory())
    state["global_sum"][:] = np.arange(24)
    state["global_count"][:] = 1
    identifiers = inventory()[data.KEY_COLUMNS].to_numpy()
    truth = np.tile(np.arange(24, dtype=float), (2, 1))
    full, duration = recovery.make_mask(identifiers, 11, family, state, truth=truth)
    repeated, _ = recovery.make_mask(identifiers, 11, family, state, truth=truth)
    assert np.array_equal(full, repeated)
    split = [recovery.make_mask(identifiers[index:index + 1], 11, family, state, truth=truth[index:index + 1])[0] for index in range(2)]
    assert np.array_equal(full, np.concatenate(split))
    assert np.array_equal(full.sum(axis=1), duration)
    assert (duration >= (2 if family == 1 else 1)).all()
    assert (duration <= (6 if family == 1 else 4)).all()
    if family == 1:
        for row in full:
            assert np.diff(np.flatnonzero(row)).max(initial=0) <= 1
    if family in [2, 3]:
        for row, length in zip(full, duration):
            assert row[24 - length:].all()


def test_masked_summaries_and_hidden_label_mutation_invariance():
    state = recovery.new_history(inventory())
    for day in range(1, 6):
        recovery.update_history(state, np.ones((2, 24)), np.zeros((2, 24), dtype=np.int8), day % 7, day)
    original = np.tile(np.arange(24, dtype=float), (2, 1))
    mask = np.zeros((2, 24), dtype=bool)
    mask[:, [7, 22]] = True
    inputs = recovery.masked_inputs(original, mask)
    assert (inputs["stock_hour6_22_cnt"] == 1).all()
    assert (inputs["unavailable_hours"] == 2).all()
    assert (inputs["recorded_daily"] == original.sum(axis=1) - 29).all()
    assert not inputs["fully_available"].any()
    snapshot = copy.deepcopy(state)
    frozen_inputs = {key: value.copy() for key, value in inputs.items()}
    estimates = recovery.profile_predictions(state, 5)[0]
    first = recovery.recover_day(inputs["sales"], inputs["flags"], estimates)
    scoring_labels = original
    scoring_labels[mask] = 1e9
    second = recovery.recover_day(inputs["sales"], inputs["flags"], recovery.profile_predictions(state, 5)[0])
    assert np.array_equal(first["profile_hourly"], second["profile_hourly"])
    assert np.array_equal(first["availability_scaling"], second["availability_scaling"])
    for key in state:
        if isinstance(state[key], np.ndarray):
            assert np.array_equal(state[key], snapshot[key])
    assert (scoring_labels[mask] == 1e9).all()
    for key in inputs:
        assert np.array_equal(inputs[key], frozen_inputs[key])


def test_pooled_metrics_and_zero_denominators():
    metrics = {}
    for actual, predicted in [(np.array([1.0]), np.array([2.0])), (np.array([99.0]), np.array([99.0]))]:
        recovery.add_metrics(metrics, "development", "daily", "overall", np.zeros(1), "profile", actual, predicted, actual)
    row = recovery.metric_rows(metrics)[0]
    assert row["wape"] == 1
    assert row["mae"] == 0.5
    zero = {}
    recovery.add_metrics(zero, "early", "hidden", "overall", np.zeros(2), "profile", np.zeros(2), np.array([1., 2.]), np.zeros(2))
    row = recovery.metric_rows(zero)[0]
    assert row["wape"] is None and row["signed_wpe"] is None
    assert row["zero_denominator"] and row["mae"] == 1.5


def test_store_clustered_bootstrap():
    values = np.array([[100, 10, 9, 8, 1], [100, 10, 9, 8, 1]], dtype=float)
    rows = recovery.bootstrap_rows({("development", "overall", "all"): values})
    first = rows[0]
    assert first["wape_difference_points"] == -2
    assert first["lower_95"] == first["upper_95"] == -2
    assert first["stores"] == 2 and first["valid_replicates"] == 1000
    values[:, 0] = 0
    assert recovery.bootstrap_rows({("early", "overall", "all"): values})[0]["lower_95"] is None


def test_partitioned_recovery_coverage_and_evaluation_protection(tmp_path, monkeypatch):
    dates = pd.date_range("2024-03-28", periods=90).strftime("%Y-%m-%d").tolist()
    rows = []
    inv = inventory()
    for city, store, product in inv[data.KEY_COLUMNS].itertuples(index=False, name=None):
        for day_number, day in enumerate(dates, 1):
            row = {"city_id": city, "store_id": store, "product_id": product, "management_group_id": 0,
                   "first_category_id": 1, "second_category_id": 2, "third_category_id": 7, "dt": day,
                   "sale_amount": 24.0, "hours_sale": [1.] * 24, "hours_stock_status": [0] * 24,
                   "stock_hour6_22_cnt": 0, "discount": 1., "holiday_flag": 0, "activity_flag": 0,
                   "precpt": 0., "avg_temperature": 15., "avg_humidity": 50., "avg_wind_level": 1.}
            rows.append(row)
    daily = tmp_path / "daily_sales.parquet"
    pq.write_table(data.daily_table(pa.Table.from_pylist(rows), dates), daily)
    original_read = pq.read_table

    def guarded_read(path, *args, **kwargs):
        assert "eval.parquet" not in str(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", guarded_read)
    outputs = []
    for batch_size in [17, 101]:
        output = tmp_path / str(batch_size)
        recovery.date_partitions(daily, output / "dates", batch_size)
        state = recovery.new_history(inv)
        writers, metrics, clusters, fallback = {}, {}, {}, Counter()
        for day_number in range(1, 70):
            table, sales, flags = recovery.load_date(output / "dates" / f"day_{day_number:03d}.parquet", inv)
            estimates, levels, supports = recovery.profile_predictions(state, (day_number - 1) % 7)
            columns, _ = recovery.natural_history(table, sales, flags, estimates, levels, supports, day_number)
            recovery.write_table(writers, output / "recovered_daily.parquet", columns)
            if 11 <= day_number <= 14 or 42 <= day_number <= 69:
                recovery.evaluate_masks(table, sales, estimates, levels, supports, inv, state, day_number,
                                        "early" if day_number <= 14 else "development", 10 if day_number <= 14 else 41 + ((day_number - 42) // 7) * 7,
                                        np.array([0, 1]), 2, writers, output, metrics, clusters, fallback)
            recovery.update_history(state, sales, flags, (day_number - 1) % 7, day_number)
        for writer in writers.values():
            writer.close()
        checked = recovery.verify_recovery_outputs(output, inv, 2 * 32 * 4)
        assert checked["historical_rows"] == 138
        outputs.append(output)
    for name in ["recovered_daily.parquet", "mask_bank.parquet", "mask_predictions.parquet", "mask_labels.parquet"]:
        assert pq.read_table(outputs[0] / name).equals(pq.read_table(outputs[1] / name))
