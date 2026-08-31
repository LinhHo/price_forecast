from forecasting.model.tft_model import TFTPriceModel
from infra.s3 import upload_dir

# from infra.s3 import upload_file


# def train_and_upload(zone: str, start: str, end: str, max_epochs: int = 5, batch_size: int = 64):
#     model = TFTPriceModel(zone)

#     model.train(
#         start=start,
#         end=end,
#         max_epochs=max_epochs,
#         batch_size=batch_size,
#     )


def train_and_upload(
    zone: str,
    start: str | None = None,
    end: str | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
):
    model = TFTPriceModel(zone)

    model.train(
        start=start,
        end=end,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )

    # Upload entire run directory (model, dataset, meta, metrics, figures)
    s3_prefix = f"{zone}/runs/{model.run_id}"
    upload_dir(model.run_dir, s3_prefix)

    # Previously only uploaded 3 artifacts explicitly:
    # run_dir = model.run_dir
    # artifacts = [
    #     ("model/tft.ckpt", run_dir / "model" / "tft.ckpt"),
    #     ("training_dataset.pt", run_dir / "training_dataset.pt"),
    #     ("meta.json", run_dir / "meta.json"),
    # ]
    # for key, path in artifacts:
    #     s3_key = f"{zone}/runs/{model.run_id}/{key}"
    #     upload_file(local_path=path, s3_key=s3_key)

    print(f"Uploaded run {model.run_id} to S3 under {s3_prefix}/")
