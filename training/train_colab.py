from forecasting.model.tft_model import TFTPriceModel
import shutil
from infra.s3 import upload_file
from config import setup_logging

setup_logging()


def train_and_upload(zone: str):
    model = TFTPriceModel(zone)

    try:
        model.train(
            start="2023-01-01",
            end="2024-01-01",
            max_epochs=5,  # 30,
            batch_size=64,
        )

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

    except Exception as e:
        run_dir = model.run_dir
        print("Training failed, cleaning up run_dir ", run_dir)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        raise e
