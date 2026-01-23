from fastapi import APIRouter
from forecasting.services.model_registry import get_model

router = APIRouter()


@router.get("/{zone}")
def predict(zone: str, date: str):
    model = get_model(zone)  # cached, fast
    df = model.predict(date)

    return {
        "zone": zone,
        "date": date,
        "forecast": df.to_dict(orient="records"),
    }
