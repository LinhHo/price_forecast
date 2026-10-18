"""Unit tests for infra/s3.py — all S3 calls are mocked via monkeypatch."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, call

import infra.s3 as s3_mod
from config import S3_BUCKET_NAME


@pytest.fixture(autouse=True)
def mock_s3_client(monkeypatch):
    """Replace the module-level boto3 client with a MagicMock for every test."""
    mock = MagicMock()
    monkeypatch.setattr(s3_mod, "_s3", mock)
    return mock


# ── upload_inference_stats ────────────────────────────────────────────────────

class TestUploadInferenceStats:
    def test_correct_s3_key(self, mock_s3_client):
        s3_mod.upload_inference_stats("FR", "2024-01-01", {"zone": "FR"})
        kw = mock_s3_client.put_object.call_args[1]
        assert kw["Key"] == "FR/inferences/2024-01-01/stats.json"
        assert kw["Bucket"] == S3_BUCKET_NAME

    def test_body_is_valid_json(self, mock_s3_client):
        payload = {"zone": "FR", "errors": []}
        s3_mod.upload_inference_stats("FR", "run1", payload)
        body = mock_s3_client.put_object.call_args[1]["Body"]
        assert json.loads(body) == payload

    def test_content_type_is_json(self, mock_s3_client):
        s3_mod.upload_inference_stats("FR", "run1", {})
        kw = mock_s3_client.put_object.call_args[1]
        assert kw["ContentType"] == "application/json"


# ── list_inferences ───────────────────────────────────────────────────────────

class TestListInferences:
    def test_returns_sorted_ids(self, mock_s3_client):
        mock_s3_client.get_paginator.return_value.paginate.return_value = [{
            "CommonPrefixes": [
                {"Prefix": "FR/inferences/2024-01-02/"},
                {"Prefix": "FR/inferences/2024-01-01/"},
            ]
        }]
        assert s3_mod.list_inferences("FR") == ["2024-01-01", "2024-01-02"]

    def test_empty_bucket_returns_empty_list(self, mock_s3_client):
        mock_s3_client.get_paginator.return_value.paginate.return_value = [{}]
        assert s3_mod.list_inferences("FR") == []

    def test_uses_correct_s3_prefix(self, mock_s3_client):
        mock_s3_client.get_paginator.return_value.paginate.return_value = [{}]
        s3_mod.list_inferences("DE")
        paginate_kwargs = mock_s3_client.get_paginator.return_value.paginate.call_args[1]
        assert paginate_kwargs["Prefix"] == "DE/inferences/"


# ── get_inference_stats ───────────────────────────────────────────────────────

class TestGetInferenceStats:
    def test_parses_json_body(self, mock_s3_client):
        payload = {"zone": "FR", "errors": [{"error": 5.0}]}
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=json.dumps(payload).encode()))
        }
        result = s3_mod.get_inference_stats("FR", "run1")
        assert result == payload

    def test_uses_correct_s3_key(self, mock_s3_client):
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b"{}"))
        }
        s3_mod.get_inference_stats("FR", "2024-01-01")
        kw = mock_s3_client.get_object.call_args[1]
        assert kw["Key"] == "FR/inferences/2024-01-01/stats.json"
        assert kw["Bucket"] == S3_BUCKET_NAME


# ── list_runs ─────────────────────────────────────────────────────────────────

class TestListRuns:
    def test_returns_sorted_runs(self, mock_s3_client):
        mock_s3_client.get_paginator.return_value.paginate.return_value = [{
            "CommonPrefixes": [
                {"Prefix": "FR/runs/run_2024-01-02/"},
                {"Prefix": "FR/runs/run_2024-01-01/"},
            ]
        }]
        assert s3_mod.list_runs("FR") == ["run_2024-01-01", "run_2024-01-02"]

    def test_empty_returns_empty_list(self, mock_s3_client):
        mock_s3_client.get_paginator.return_value.paginate.return_value = [{}]
        assert s3_mod.list_runs("FR") == []


# ── list_zones ────────────────────────────────────────────────────────────────

class TestListZones:
    def test_returns_sorted_zones(self, mock_s3_client):
        mock_s3_client.get_paginator.return_value.paginate.return_value = [{
            "CommonPrefixes": [
                {"Prefix": "NL/"},
                {"Prefix": "FR/"},
            ]
        }]
        assert s3_mod.list_zones() == ["FR", "NL"]

    def test_empty_bucket_returns_empty_list(self, mock_s3_client):
        mock_s3_client.get_paginator.return_value.paginate.return_value = [{}]
        assert s3_mod.list_zones() == []


# ── upload_dir ────────────────────────────────────────────────────────────────

class TestUploadDir:
    def test_uploads_all_files(self, mock_s3_client, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "b.txt").write_text("b")

        s3_mod.upload_dir(tmp_path, "FR/runs/run1")

        assert mock_s3_client.upload_file.call_count == 2
        uploaded_keys = {c[0][2] for c in mock_s3_client.upload_file.call_args_list}
        assert "FR/runs/run1/a.txt" in uploaded_keys
        assert "FR/runs/run1/sub/b.txt" in uploaded_keys

    def test_empty_directory_uploads_nothing(self, mock_s3_client, tmp_path):
        s3_mod.upload_dir(tmp_path, "FR/runs/run1")
        mock_s3_client.upload_file.assert_not_called()
