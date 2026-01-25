from forecasting.model.tft_model import TFTPriceModel
from infra.s3 import upload_file


def train_and_upload(zone: str):
    model = TFTPriceModel(zone)

    model.train(
        start="2023-01-01",
        end="2024-01-01",
        max_epochs=5,  # 30,
        batch_size=64,
    )

    # artifacts
    run_dir = model.run_dir

    artifacts = [
        ("model/tft.ckpt", run_dir / "model" / "tft.ckpt"),
        ("training_dataset.pt", run_dir / "data" / "training_dataset.pt"),
        ("meta.json", run_dir / "meta.json"),
    ]

    for key, path in artifacts:
        s3_key = f"{zone}/runs/{model.run_id}/{key}"
        upload_file(local_path=path, s3_key=s3_key)

    print(f"Uploaded run {model.run_id} to S3")


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


# from forecasting.model.tft_model import TFTPriceModel
# from infra.s3 import upload_file


# def train_and_upload(zone: str):
#     model = TFTPriceModel(zone)

#     model.train(
#         start="2023-01-01",
#         end="2024-01-01",
#         max_epochs=5,  # 30,
#         batch_size=64,
#     )

#     # s3_prefix = f"{zone}/runs/{model.run_id}"
#     # upload_file(model.run_dir, s3_prefix)

#     s3_prefix = f"{zone}/runs/{model.run_id}/model/tft.ckpt"
#     upload_file(
#         local_path=model.run_dir / "model" / "tft.ckpt",
#         s3_key=s3_prefix,
#     )

#     print(f"Uploaded to s3://price-forecast-tft-model/{s3_prefix}")
