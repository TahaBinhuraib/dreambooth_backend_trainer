import os

from app.trainer import dreambooth

MODEL_ID = os.environ["model_id"]
S3_FOLDER = os.environ["s3_folder"]
STEPS = os.environ.get("steps", 2000)
UNIQUE_IDENTIFIER = os.environ["unique_identifier"]
BUCKET_NAME = os.environ["bucket_name"]


def main() -> None:
    dreambooth_trainer = dreambooth.DreamBooth(
        unique_identifier=UNIQUE_IDENTIFIER,
        images=S3_FOLDER,
        steps=STEPS,
        model_id=MODEL_ID,
        bucket_name=BUCKET_NAME,
    )
    dreambooth_trainer.train()


if __name__ == "__main__":
    main()
