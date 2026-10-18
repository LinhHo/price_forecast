"""Unit tests for forecasting/model/services/model_registry.py."""
import time
import pytest
from pathlib import Path
from unittest.mock import patch

import forecasting.model.services.model_registry as registry


@pytest.fixture(autouse=True)
def clear_caches():
    """Reset module-level caches before and after each test."""
    registry._MODEL_CACHE.clear()
    registry._S3_RUN_CACHE.clear()
    yield
    registry._MODEL_CACHE.clear()
    registry._S3_RUN_CACHE.clear()


# ── _resolve_latest_local_run ─────────────────────────────────────────────────

class TestResolveLatestLocalRun:
    def test_returns_lexicographically_last_run(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "run_2024-01-01").mkdir()
        (runs / "run_2024-01-02").mkdir()
        (runs / "run_2024-01-03").mkdir()
        assert registry._resolve_latest_local_run(tmp_path) == "run_2024-01-03"

    def test_raises_when_no_runs_dir(self, tmp_path):
        with pytest.raises(RuntimeError, match="No local runs found"):
            registry._resolve_latest_local_run(tmp_path)

    def test_raises_when_runs_dir_is_empty(self, tmp_path):
        (tmp_path / "runs").mkdir()
        with pytest.raises(RuntimeError, match="No trained runs found"):
            registry._resolve_latest_local_run(tmp_path)

    def test_single_run_is_returned(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "run_only").mkdir()
        assert registry._resolve_latest_local_run(tmp_path) == "run_only"


# ── _latest_s3_run ────────────────────────────────────────────────────────────

class TestLatestS3Run:
    def test_returns_last_sorted_run(self):
        with patch("infra.s3.list_runs", return_value=["run_2024-01-01", "run_2024-01-02"]):
            assert registry._latest_s3_run("FR") == "run_2024-01-02"

    def test_returns_none_when_no_runs(self):
        with patch("infra.s3.list_runs", return_value=[]):
            assert registry._latest_s3_run("FR") is None

    def test_cache_hit_skips_s3_call(self):
        registry._S3_RUN_CACHE["FR"] = ("run_cached", time.time())  # fresh entry
        with patch("infra.s3.list_runs") as mock_list:
            result = registry._latest_s3_run("FR")
        assert result == "run_cached"
        mock_list.assert_not_called()

    def test_expired_cache_calls_s3(self):
        registry._S3_RUN_CACHE["FR"] = ("run_stale", 0.0)  # expired (timestamp=epoch)
        with patch("infra.s3.list_runs", return_value=["run_new"]) as mock_list:
            result = registry._latest_s3_run("FR")
        assert result == "run_new"
        mock_list.assert_called_once()

    def test_s3_error_returns_stale_value(self):
        registry._S3_RUN_CACHE["FR"] = ("run_stale", 0.0)  # expired
        with patch("infra.s3.list_runs", side_effect=Exception("no credentials")):
            result = registry._latest_s3_run("FR")
        assert result == "run_stale"

    def test_s3_error_resets_ttl(self):
        """After an S3 error the TTL is reset so we don't hammer S3 every call."""
        registry._S3_RUN_CACHE["FR"] = ("run_stale", 0.0)
        with patch("infra.s3.list_runs", side_effect=Exception("no credentials")):
            registry._latest_s3_run("FR")
        _, checked_at = registry._S3_RUN_CACHE["FR"]
        assert time.time() - checked_at < 5  # TTL was just reset
