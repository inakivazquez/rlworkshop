import pytest
from rlworkshop import RewardTracker


def test_aggregation_counts_and_total():
    tracker = RewardTracker(["success", "step_penalty"])
    assert tracker.add("step_penalty", -0.1) == -0.1
    tracker.add("step_penalty", -0.1)
    tracker.add("other", 1.0)
    summary = tracker.snapshot()
    assert summary["reward_components"]["success"] == 0
    assert summary["reward_counts"]["success"] == 0
    assert summary["reward_components"]["step_penalty"] == pytest.approx(-0.2)
    assert summary["reward_counts"]["step_penalty"] == 2
    assert tracker.total == pytest.approx(0.8)


def test_snapshot_is_independent_and_reset():
    tracker = RewardTracker(["goal"])
    info = tracker.add_to_info({"custom": "ok"})
    tracker.add("goal", 2)
    assert info["reward_components"]["goal"] == 0
    assert info["custom"] == "ok"
    tracker.reset()
    assert tracker.total == 0
    assert tracker.counts["goal"] == 0


def test_invalid_component_name():
    with pytest.raises(ValueError):
        RewardTracker().add("", 2)
