from fastapi import APIRouter, HTTPException
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{zone}")
def zone_stats(zone: str):
    """
    Aggregate inference accuracy stats for a zone from S3.
    Returns summary numbers + all raw per-hour errors for client-side plotting.
    """
    try:
        from infra.s3 import list_inferences, get_inference_stats
        ids = list_inferences(zone)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not ids:
        return {
            "zone": zone, "n_inferences": 0,
            "mae": None, "rmse": None, "mean_error": None, "over_rate": None,
            "errors": [],
        }

    all_errors: list[dict] = []
    for iid in ids:
        try:
            stats = get_inference_stats(zone, iid)
            all_errors.extend(stats.get("errors", []))
        except Exception as e:
            logger.warning("Could not load stats zone=%s inference=%s: %s", zone, iid, e)

    err_vals = [e["error"] for e in all_errors if e.get("error") is not None]

    if not err_vals:
        return {
            "zone": zone, "n_inferences": len(ids),
            "mae": None, "rmse": None, "mean_error": None, "over_rate": None,
            "errors": [],
        }

    n      = len(err_vals)
    mae    = sum(abs(e) for e in err_vals) / n
    rmse   = (sum(e**2 for e in err_vals) / n) ** 0.5
    mean_e = sum(err_vals) / n
    n_over = sum(1 for e in err_vals if e > 0)

    return {
        "zone":          zone,
        "n_inferences":  len(ids),
        "n_error_hours": n,
        "mae":           round(mae, 2),
        "rmse":          round(rmse, 2),
        "mean_error":    round(mean_e, 2),
        "n_over":        n_over,
        "n_under":       n - n_over,
        "over_rate":     round(n_over / n, 3),
        "errors":        all_errors,   # all per-hour records for client-side charts
    }
