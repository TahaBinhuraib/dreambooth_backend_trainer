import os


class Config:

    AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", default="us-east-1")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", default=None)
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", default=None)
    AWS_S3_MODELS_BUCKET = os.getenv("AWS_S3_MODELS_BUCKET", default="dreambooth-api")
