from app.storage import storage


def main() -> None:
    s3 = storage.S3Storage()
    s3.download_s3_file(bucket_name="dreambooth-api", key="models/model_id_1.zip")


if __name__ == "__main__":
    main()
