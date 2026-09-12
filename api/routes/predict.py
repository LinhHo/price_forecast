from fastapi import APIRouter, HTTPException
from forecasting.model.services.model_registry import get_model
from forecasting.data.entsoe import load_prices
import pandas as pd
import math
import traceback
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def _sanitize(v):
    """Replace NaN/Inf floats with None so FastAPI can JSON-encode them."""
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except TypeError:
        pass
    return v


def _compute_slot_stats(preds: list[dict], actual: list[dict]) -> dict | None:
    """Compute error summary for a single forecast slot given actual prices."""
    actual_map = {r["timestamp"]: r["price"] for r in actual}
    errors = [
        row["p50"] - actual_map[row["timestamp"]]
        for row in preds
        if row.get("timestamp") in actual_map
        and row.get("p50") is not None
        and not math.isnan(row["p50"])
    ]
    if not errors:
        return None
    n = len(errors)
    return {
        "n":          n,
        "mae":        round(sum(abs(e) for e in errors) / n, 2),
        "mean_error": round(sum(errors) / n, 2),
        "n_over":     sum(1 for e in errors if e > 0),
        "n_under":    sum(1 for e in errors if e < 0),
    }


def _save_inference_stats(
    zone: str, model, inference_id: str, forecast_start_iso: str,
    preds: list[dict], actual: list[dict],
):
    """Build per-hour error records and upload stats JSON to S3. Logs on failure."""
    try:
        from infra.s3 import upload_inference_stats
        actual_map = {r["timestamp"]: r["price"] for r in actual}
        errors = []
        for row in preds:
            ts  = row.get("timestamp")
            p50 = row.get("p50")
            if ts in actual_map and p50 is not None and not math.isnan(p50):
                err = p50 - actual_map[ts]
                errors.append({
                    "timestamp": ts,
                    "hour_utc":  int(ts[11:13]),
                    "p50":       round(p50, 2),
                    "actual":    round(actual_map[ts], 2),
                    "error":     round(err, 2),
                })
        if not errors:
            return
        errs = [e["error"] for e in errors]
        n    = len(errs)
        stats = {
            "zone":            zone,
            "inference_id":    inference_id,
            "training_run_id": model.run_id,
            "forecast_start":  forecast_start_iso,
            "errors":          errors,
            "summary": {
                "n":          n,
                "mae":        round(sum(abs(e) for e in errs) / n, 2),
                "rmse":       round((sum(e**2 for e in errs) / n) ** 0.5, 2),
                "mean_error": round(sum(errs) / n, 2),
                "n_over":     sum(1 for e in errs if e > 0),
                "n_under":    sum(1 for e in errs if e < 0),
            },
        }
        upload_inference_stats(zone, inference_id, stats)
        logger.info("Saved inference stats zone=%s inference=%s", zone, inference_id)
    except Exception as e:
        logger.warning("Could not save inference stats: %s", e)


@router.get("/{zone}")
def predict(zone: str, date_to_predict: str | None = None):
    try:
        model = get_model(zone)
        preds  = model.predict(date_to_predict)
        run_id = model.run_id
        base_url = f"/artifacts/{zone}/runs/{run_id}/predictions"

        actual         = None
        forecast_start = None

        if date_to_predict:
            forecast_start = pd.Timestamp(date_to_predict)
            if forecast_start.tzinfo is None:
                forecast_start = forecast_start.tz_localize("UTC")
            forecast_end = forecast_start + pd.Timedelta(hours=24)

            if forecast_end < pd.Timestamp.now(tz="UTC"):
                try:
                    entsoe_start = forecast_start.normalize()
                    entsoe_end   = forecast_end.normalize() + pd.Timedelta(days=1)
                    df_actual    = load_prices(zone, entsoe_start, entsoe_end)
                    df_actual    = df_actual.loc[forecast_start : forecast_end - pd.Timedelta(hours=1)]
                    actual = [
                        {
                            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "price":     round(float(row["price_eur_per_mwh"]), 2),
                        }
                        for ts, row in df_actual.iterrows()
                    ]
                except Exception as e:
                    logger.warning("Could not fetch actual ENTSOE prices: %s", e)

        clean_preds  = [{k: _sanitize(v) for k, v in row.items()} for row in preds]
        clean_actual = (
            [{k: _sanitize(v) for k, v in row.items()} for row in actual]
            if actual is not None else None
        )

        slot_stats = None
        if clean_actual and forecast_start:
            slot_stats   = _compute_slot_stats(clean_preds, clean_actual)
            inference_id = forecast_start.strftime("%Y-%m-%d_%H-%M-%S")
            _save_inference_stats(zone, model, inference_id, forecast_start.isoformat(), clean_preds, clean_actual)

        return {
            "zone":           zone,
            "run_id":         run_id,
            "csv":            f"{base_url}/forecast.csv",
            "png":            f"{base_url}/forecast.png",
            "weather_source": model.weather_source,
            "data":           clean_preds,
            "actual":         clean_actual,
            "slot_stats":     slot_stats,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
