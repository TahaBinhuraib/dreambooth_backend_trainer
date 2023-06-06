import os
import shutil
from typing import List

import boto3

from app.config.config import Config


class S3Storage:
    def __init__(self) -> None:
        self.s3 = boto3.resource(
            "s3",
            region_name="us-east-2",
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
        )

    def zip_and_upload(self, folder_path: str, bucket_name: str, key: str) -> None:
        """Zip a folder and upload it to S3.

        Args:
            folder_path (str): The path of the folder to zip.
            bucket_name (str): The name of the S3 bucket.
            key (str): The key of the S3 object.
        """

        zip_file_path = "./test.zip"

        try:
            shutil.make_archive("test", "zip", folder_path)
            print(key)
            self.s3.meta.client.upload_file(zip_file_path, bucket_name, key)
            # self.s3.upload_file(zip_file_path, bucket_name, key)
        except Exception as e:
            print(f"error while uploading {folder_path} to s3: {e}")

    def download_s3_file(self, bucket_name: str, key: str) -> None:
        """Downloads a file from s3

        Args:
            bucket_name (str): Bucket name in s3
            key (str): s3 key(file name) could be a path if file is in a folder.
        """
        self.s3.Bucket(bucket_name).download_file(key, "v_1_4.zip")

    def download_s3_folder(self, bucket_name, s3_folder, local_dir="./images/") -> None:
        """
        Download the contents of a folder directory
        Args:
            bucket_name: the name of the s3 bucket
            s3_folder: the folder path in the s3 bucket
            local_dir: a relative or absolute directory path in the local file system
        """
        bucket = self.s3.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=s3_folder):
            target = (
                obj.key
                if local_dir is None
                else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
            )
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == "/":
                continue
            bucket.download_file(obj.key, target)
