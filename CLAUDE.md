# Price forecast

Forecast electricity price with a machine-learning model to inform when to charge the user's EV.

**Rule:** keep code strictly to the structure below. Ask when unclear.

## Model

- **Method:** Temporal Fusion Transformer (TFT), PyTorch-forecasting.
- **Resolution:** hourly.

## Data

- **Meteorological:** train on ERA5 reanalysis; infer with open-meteo.
- **Target:** day-ahead electricity price (ENTSO-E).
- **Extra features:** temporal — day of week, week of year, month of year.

## Output

Shown in a dashboard served via the API (`/dashboard`).

- Predict price 24h ahead of the selected time, with an uncertainty band, as a timeseries plot:
  - **Blue:** prediction, include the uncertainty band in light blue.
  - **Orange:** ENTSO-E actual price, including the previous 7 days, and if available, the actual price of 24h from the selected time (i.e. selected time is >24h in the past).
- *Planned:* statistics plot — MAE distribution (requires saving past-prediction results).
- *Planned:* return two options for when to charge the EV; input = time needed to fully charge.

## Process

- Training runs on Google Colab GPU; trained models saved to AWS S3.
- FastAPI: user selects a time/date and a country (from those with pre-trained models on S3).

## Structure

```
forecasting/
  data/                # data flow: load, preprocess
  model/tft_model.py   # TFT definition, train & save
training/
  train_colab.py       # training on Google Colab GPU
infra/
  s3.py                # save/load models on AWS S3; saved predictions, metrics, and figures (needed for the MAE step)
api/                   # FastAPI: select time + country/bidding zone
dashboard/             # output plots served via the API
```

---

## Common structure in a scientific programming project

- **Dockerfile** — handle the environment, keep the deployment consistent 
- **README.md** — purpose, install steps, how to run.
- **requirements.txt** — pin dependencies
- **config.py** — paths, parameters, bidding zones in a config file (e.g. YAML), separate from code; use `logging` over `print` for the pipeline and API.
- **tests/** — unit tests for data handling and model I/O; run before merging.
- **results/** —.
- **.env + .gitignore** — keep AWS credentials and secrets in `.env`; never commit them, data, or model weights.
- **CITATION.cff / LICENSE** — how to cite the work and terms of use.