"""Unit tests for the /stats/{zone} aggregation route."""
import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.stats import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# Two inference records:
#   record A: errors +10, -5
#   record B: error  +5
# Combined: err_vals = [10, -5, 5]
#   n=3, mae=(10+5+5)/3=6.67, rmse=sqrt((100+25+25)/3)=sqrt(50)=7.07
#   mean=(10-5+5)/3=3.33, n_over=2, n_under=1, over_rate=0.667
STATS_A = {"errors": [
    {"error": 10.0, "hour_utc": 6, "timestamp": "2024-01-01T06:00:00Z", "p50": 100.0, "actual": 90.0},
    {"error": -5.0, "hour_utc": 7, "timestamp": "2024-01-01T07:00:00Z", "p50": 125.0, "actual": 130.0},
]}
STATS_B = {"errors": [
    {"error": 5.0,  "hour_utc": 8, "timestamp": "2024-01-01T08:00:00Z", "p50": 105.0, "actual": 100.0},
]}


def _mock_get(zone, iid):
    return {"2024-01-01": STATS_A, "2024-01-02": STATS_B}[iid]


class TestZoneStatsEmpty:
    def test_empty_zone_returns_zero_response(self):
        with patch("infra.s3.list_inferences", return_value=[]):
            resp = client.get("/FR")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_inferences"] == 0
        assert data["mae"] is None
        assert data["errors"] == []

    def test_s3_list_error_returns_500(self):
        with patch("infra.s3.list_inferences", side_effect=Exception("no creds")):
            resp = client.get("/FR")
        assert resp.status_code == 500


class TestZoneStatsAggregation:
    def test_n_inferences(self):
        with patch("infra.s3.list_inferences", return_value=["2024-01-01", "2024-01-02"]), \
             patch("infra.s3.get_inference_stats", side_effect=_mock_get):
            data = client.get("/FR").json()
        assert data["n_inferences"] == 2

    def test_n_error_hours(self):
        with patch("infra.s3.list_inferences", return_value=["2024-01-01", "2024-01-02"]), \
             patch("infra.s3.get_inference_stats", side_effect=_mock_get):
            data = client.get("/FR").json()
        assert data["n_error_hours"] == 3

    def test_mae(self):
        with patch("infra.s3.list_inferences", return_value=["2024-01-01", "2024-01-02"]), \
             patch("infra.s3.get_inference_stats", side_effect=_mock_get):
            data = client.get("/FR").json()
        assert data["mae"] == round((10 + 5 + 5) / 3, 2)

    def test_mean_error(self):
        with patch("infra.s3.list_inferences", return_value=["2024-01-01", "2024-01-02"]), \
             patch("infra.s3.get_inference_stats", side_effect=_mock_get):
            data = client.get("/FR").json()
        assert data["mean_error"] == round((10 - 5 + 5) / 3, 2)

    def test_over_rate(self):
        with patch("infra.s3.list_inferences", return_value=["2024-01-01", "2024-01-02"]), \
             patch("infra.s3.get_inference_stats", side_effect=_mock_get):
            data = client.get("/FR").json()
        assert data["n_over"] == 2
        assert data["n_under"] == 1
        assert data["over_rate"] == round(2 / 3, 3)

    def test_errors_list_contains_all_records(self):
        with patch("infra.s3.list_inferences", return_value=["2024-01-01", "2024-01-02"]), \
             patch("infra.s3.get_inference_stats", side_effect=_mock_get):
            data = client.get("/FR").json()
        assert len(data["errors"]) == 3

    def test_bad_inference_is_skipped(self):
        """A single failing inference should not break the whole response."""
        def mock_get_with_error(zone, iid):
            if iid == "2024-01-01":
                raise Exception("corrupted")
            return STATS_B

        with patch("infra.s3.list_inferences", return_value=["2024-01-01", "2024-01-02"]), \
             patch("infra.s3.get_inference_stats", side_effect=mock_get_with_error):
            data = client.get("/FR").json()
        assert data["n_inferences"] == 2
        assert data["n_error_hours"] == 1  # only STATS_B loaded
