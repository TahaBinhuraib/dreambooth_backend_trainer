import pydantic

from app.storage import storage
from app.trainer.train_dreambooth import DreamboothTrainer


class DreamBooth(pydantic.BaseModel):
    """Dreambooth trainer class blueprint. train method will handle all tasks related to training.

    Args:
    seed: Random seed to set for training
    steps: Number of training steps
    images: S3 links to training images
    unique_identifier: binds a unique identifier with the specific subject(Training images)
    """

    unique_identifier: str
    images: str
    steps: int
    model_id: str
    bucket_name: str

    def train(self) -> dict:
        """Dreambooth Trainer. Steps to complete:
            1. Download images
            2. Call DreamboothTrainer and train model
            3. Upload model zip to S3

        Returns:
            dict: on complete or failure post a request to dreambooth_lambda_url with {"info": "failed | completed"}
        """
        s3 = storage.S3Storage()
        s3.download_s3_folder(
            bucket_name=self.bucket_name,
            s3_folder=self.images,
        )

        trainer = DreamboothTrainer(
            pretrained_model_path="./v1_4_fp16",
            instance_data_dir="./images",
            instance_prompt=self.unique_identifier,
            output_dir="./test-model",
        )
        trainer.main_trainer()

        s3.zip_and_upload(
            folder_path="./images/",
            bucket_name="dreambooth-api",
            key=f"models/model-{self.model_id}.zip",
        )
