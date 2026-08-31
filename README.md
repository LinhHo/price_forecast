This module forecasts electricity price for a selected zone using Temporal Fusion Transformer (TFT) model, using weather data.

## Web UI

Start the API server:

```bash
uvicorn api.app:app --reload
```

Then open **http://localhost:8000/ui/** in your browser.

### How to use

1. **Forecast Window** — pick a date and a start hour (UTC). The model predicts the 24 h window beginning at that hour.
2. **Model Selection** — choose a country/zone. The list is pulled from AWS S3, showing only zones with a trained model.
3. Press **Run Forecast →**.

The chart shows:
- Orange line + shaded band — forecast P50 with P10–P90 uncertainty range
- Blue line — actual ENTSOE day-ahead prices (only shown when the selected date is in the past)

## Setup

Put your API tokens in `.env`:

```
ENTSOE_TOKEN=...
ERA5_TOKEN=...
```

AWS credentials must be configured (via `~/.aws/credentials` or environment variables) so the app can read models from S3.

## Training

Training runs in Colab (not via the API). See `training/train_colab.py`.