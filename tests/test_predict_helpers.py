"""Unit tests for pure helper functions in api/routes/predict.py."""
import math
import pytest
from unittest.mock import MagicMock, patch

from api.routes.predict import _sanitize, _compute_slot_stats, _save_inference_stats


# ── _sanitize ────────────────────────────────────────────────────────────────

class TestSanitize:
    def test_nan_returns_none(self):
        assert _sanitize(float("nan")) is None

    def test_pos_inf_returns_none(self):
        assert _sanitize(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert _sanitize(float("-inf")) is None

    def test_valid_float_unchanged(self):
        assert _sanitize(42.5) == 42.5

    def test_int_unchanged(self):
        assert _sanitize(10) == 10

    def test_string_unchanged(self):
        assert _sanitize("hello") == "hello"

    def test_none_unchanged(self):
        assert _sanitize(None) is None

    def test_zero_unchanged(self):
        assert _sanitize(0.0) == 0.0


# ── _compute_slot_stats ───────────────────────────────────────────────────────

# errors: +10, -10, +10  →  MAE=10, mean=10/3, n_over=2, n_under=1
PREDS = [
    {"timestamp": "2024-01-01T00:00:00Z", "p50": 100.0},
    {"timestamp": "2024-01-01T01:00:00Z", "p50": 120.0},
    {"timestamp": "2024-01-01T02:00:00Z", "p50": 80.0},
]
ACTUAL = [
    {"timestamp": "2024-01-01T00:00:00Z", "price": 90.0},
    {"timestamp": "2024-01-01T01:00:00Z", "price": 130.0},
    {"timestamp": "2024-01-01T02:00:00Z", "price": 70.0},
]


class TestComputeSlotStats:
    def test_mae(self):
        s = _compute_slot_stats(PREDS, ACTUAL)
        assert s["mae"] == 10.0

    def test_mean_error(self):
        s = _compute_slot_stats(PREDS, ACTUAL)
        assert s["mean_error"] == round(10 / 3, 2)

    def test_n_over_under(self):
        s = _compute_slot_stats(PREDS, ACTUAL)
        assert s["n_over"] == 2
        assert s["n_under"] == 1
        assert s["n"] == 3

    def test_no_overlap_returns_none(self):
        other = [{"timestamp": "2099-01-01T00:00:00Z", "price": 50.0}]
        assert _compute_slot_stats(PREDS, other) is None

    def test_nan_p50_is_skipped(self):
        preds = [
            {"timestamp": "2024-01-01T00:00:00Z", "p50": float("nan")},
            {"timestamp": "2024-01-01T01:00:00Z", "p50": 120.0},
        ]
        actual = [
            {"timestamp": "2024-01-01T00:00:00Z", "price": 90.0},
            {"timestamp": "2024-01-01T01:00:00Z", "price": 100.0},
        ]
        s = _compute_slot_stats(preds, actual)
        assert s["n"] == 1
        assert s["mae"] == 20.0

    def test_empty_preds_returns_none(self):
        assert _compute_slot_stats([], ACTUAL) is None

    def test_empty_actual_returns_none(self):
        assert _compute_slot_stats(PREDS, []) is None


# ── _save_inference_stats ─────────────────────────────────────────────────────

SAVE_PREDS = [
    {"timestamp": "2024-01-01T06:00:00Z", "p50": 100.0},
    {"timestamp": "2024-01-01T07:00:00Z", "p50": 120.0},
]
SAVE_ACTUAL = [
    {"timestamp": "2024-01-01T06:00:00Z", "price": 90.0},
    {"timestamp": "2024-01-01T07:00:00Z", "price": 130.0},
]


@pytest.fixture
def fake_model():
    m = MagicMock()
    m.run_id = "run_2024-01-01_000000"
    return m


class TestSaveInferenceStats:
    def test_upload_called_with_correct_args(self, fake_model):
        with patch("infra.s3.upload_inference_stats") as mock_upload:
            _save_inference_stats(
                "FR", fake_model, "2024-01-01_06-00-00",
                "2024-01-01T06:00:00", SAVE_PREDS, SAVE_ACTUAL,
            )
            mock_upload.assert_called_once()
            zone, iid, stats = mock_upload.call_args[0]
            assert zone == "FR"
            assert iid == "2024-01-01_06-00-00"
            assert stats["zone"] == "FR"
            assert stats["inference_id"] == "2024-01-01_06-00-00"
            assert stats["training_run_id"] == fake_model.run_id

    def test_error_records_are_correct(self, fake_model):
        with patch("infra.s3.upload_inference_stats") as mock_upload:
            _save_inference_stats(
                "FR", fake_model, "id", "2024-01-01T06:00:00",
                SAVE_PREDS, SAVE_ACTUAL,
            )
            stats = mock_upload.call_args[0][2]
            assert len(stats["errors"]) == 2
            assert stats["errors"][0]["error"] == 10.0   # 100 - 90
            assert stats["errors"][1]["error"] == -10.0  # 120 - 130

    def test_summary_stats_are_correct(self, fake_model):
        with patch("infra.s3.upload_inference_stats") as mock_upload:
            _save_inference_stats(
                "FR", fake_model, "id", "2024-01-01T06:00:00",
                SAVE_PREDS, SAVE_ACTUAL,
            )
            summary = mock_upload.call_args[0][2]["summary"]
            assert summary["n"] == 2
            assert summary["mae"] == 10.0
            assert summary["n_over"] == 1
            assert summary["n_under"] == 1

    def test_s3_error_is_swallowed(self, fake_model):
        with patch("infra.s3.upload_inference_stats", side_effect=Exception("S3 down")):
            # must not raise
            _save_inference_stats(
                "FR", fake_model, "id", "2024-01-01T06:00:00",
                SAVE_PREDS, SAVE_ACTUAL,
            )

    def test_no_matching_timestamps_skips_upload(self, fake_model):
        with patch("infra.s3.upload_inference_stats") as mock_upload:
            _save_inference_stats(
                "FR", fake_model, "id", "2024-01-01T06:00:00",
                SAVE_PREDS, [],  # no actual prices
            )
            mock_upload.assert_not_called()
