# from forecasting.model.tft_model import TFTPriceModel
# from infra.s3 import upload_run

# def train_zone(zone: str):
#     model = TFTPriceModel(zone)
#     model.train(
#         start="2023-01-01",
#         end="2024-01-01",
#         max_epochs=30,
#         batch_size=64,
#     )

#     upload_run(zone, model.run_dir)


from forecasting.model.tft_model import TFTPriceModel
from infra.s3 import upload_dir


def train_and_upload(zone: str):
    model = TFTPriceModel(zone)

    model.train(
        start="2023-01-01",
        end="2024-01-01",
        max_epochs=5,  # 30,
        batch_size=64,
    )

    s3_prefix = f"{zone}/runs/{model.run_id}"
    upload_dir(model.run_dir, s3_prefix)

    print(f"Uploaded to s3://price-forecast-tft-model/{s3_prefix}")
