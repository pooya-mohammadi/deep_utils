import logging
import os
from typing import Tuple

import boto3


class Boto3Utils:
    def __init__(self,

                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
                 endpoint_url: str = None,
                 aws_region: str = None,
                 bucket_name: str = None,
                 credential_txt_path: str = None,
                 querystring_expire: int = 12 * 60 * 60,
                 querystring_auth: bool = True) -> None:
        """
            Initializes an S3Uploader instance.
            If credential_txt_path is provided the credentials are taken from the text file.
            Text File's format:
            AWS_ACCESS_KEY_ID=<your-key>
            AWS_SECRET_ACCESS_KEY=<your-secret-key>
            AWS_REGION=<your-region>
            :param aws_access_key_id: The AWS access key ID to use for authentication.
            :param aws_secret_access_key: The AWS secret access key to use for authentication.
            :param aws_region: The AWS region to use.
            :param credential_txt_path: str = None
            :param querystring_auth: bool = True - If True, the querystring will be signed.
            :param querystring_expire: int = 3600 - The number of seconds the querystring will be valid for.
        """
        if credential_txt_path and (aws_secret_access_key is None or aws_access_key_id is None):
            aws_access_key_id, aws_secret_access_key, region_name = self._get_credentials_from_txt(
                credential_txt_path)

        if aws_secret_access_key is None or aws_access_key_id is None:
            aws_access_key_id = aws_access_key_id or os.getenv('aws_access_key_id') or os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = aws_secret_access_key or os.getenv('aws_secret_access_key') or os.getenv(
                'AWS_SECRET_ACCESS_KEY')
            aws_region = aws_region or os.getenv("aws_region") or os.getenv("AWS_REGION")

        assert aws_access_key_id, "aws_access_key_id is not provided"
        assert aws_secret_access_key, "aws_secret_access_key is not provided"
        assert aws_region or endpoint_url, "aws_region or endpoint_url is not provided"
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = aws_region
        self.bucket_name = bucket_name
        self.con_s3, self.con_bucket = self.connect_s3(bucket_name)
        self.querystring_expire = querystring_expire
        self.querystring_auth = querystring_auth

    def put_presigned_url(self, object_path: str, parameters=None, expire=None, bucket_name=None):
        # Preserve the trailing slash after normalizing the path.
        params = parameters.copy() if parameters else {}
        if expire is None:
            expire = 3600

        params["Bucket"] = self.bucket_name if bucket_name is None else bucket_name
        params["Key"] = object_path.lstrip("/")
        con_bucket = self.con_bucket if self.con_bucket is not None else self._get_bucket_connection(params["Bucket"])

        url = con_bucket.meta.client.generate_presigned_url(
            "put_object", Params=params, ExpiresIn=expire
        )

        return url

    def get_presigned_url(self, object_path: str, parameters=None, expire=None, bucket_name=None):
        # Preserve the trailing slash after normalizing the path.
        params = parameters.copy() if parameters else {}
        if expire is None:
            expire = 3600

        params["Bucket"] = self.bucket_name if bucket_name is None else bucket_name
        params["Key"] = object_path.lstrip("/")
        con_bucket = self.con_bucket if self.con_bucket is not None else self._get_bucket_connection(params["Bucket"])

        url = con_bucket.meta.client.generate_presigned_url(
            "get_object", Params=params, ExpiresIn=expire
        )
        return url

    @staticmethod
    def _get_credentials_from_txt(credential_txt_path: str) -> tuple:
        """
        get credentials from txt file. Text File's format:
            AWS_ACCESS_KEY_ID=<your-key>
            AWS_SECRET_ACCESS_KEY=<your-secret-key>
            AWS_REGION=<your-region>
        :param credential_txt_path:
        :return:
        """
        credentials = dict()
        with open(credential_txt_path, 'r') as f:
            for line in f.readlines():
                key, value = line.strip().split('=')
                credentials[key] = value

        return credentials['AWS_ACCESS_KEY_ID'], credentials['AWS_SECRET_ACCESS_KEY'], credentials['AWS_REGION']

    def _get_bucket_connection(self, bucket_name):
        """
        Creates bucket connection. First it tries to connect using user provided bucket_name, otherwise it uses the one
        provided in constructor. If both are None, then it raises ValueError.
        :param bucket_name:
        :return:
        """
        if bucket_name is None and self.bucket_name is None:
            raise ValueError('bucket_name must be provided if it is not set in the constructor')
        con_bucket = self.con_s3.Bucket(bucket_name) if bucket_name else self.con_bucket
        return con_bucket

    def connect_s3(self, bucket_name: str = None) -> Tuple[boto3.resource, boto3.client]:
        """
        Connects to an S3 bucket.

        :param bucket_name: The name of the S3 bucket to connect to, if None, the base s3 is returned.
        :return: The S3 bucket resource.
        """
        try:
            s3 = boto3.resource(
                endpoint_url=self.endpoint_url,
                service_name='s3',
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            logging.info(f'Connection created Successfully! ["Status":]')
            if bucket_name:
                return s3, s3.Bucket(bucket_name)
            else:
                return s3, None
        except Exception as error:
            logging.error(f'in Creating connection ! {error}')
            raise error

    def get_boto_client(self):
        return boto3.client('s3',
                            endpoint_url=self.endpoint_url,
                            region_name=self.region_name,
                            aws_access_key_id=self.aws_access_key_id,
                            aws_secret_access_key=self.aws_secret_access_key
                            )

    def upload_file(self, file_path: str, file_name: str, bucket_name: str = None) -> None:
        """
           Uploads a file to an S3 bucket.
           :param file_path: The path to the file to upload.
           :param file_name: The name to give the uploaded file in the S3 bucket.
           :param bucket_name: The name of the S3 bucket to upload to.
        """
        con_bucket = self._get_bucket_connection(bucket_name)
        con_bucket.upload_file(Filename=file_path, Key=file_name)


    def download_file(self, remote_key: str, local_download_path: str, bucket_name: str = None,
                      check_file_exists=False) -> str:
        """
        downloads a file from s3/minio
        :param remote_key:
        :param local_download_path:
        :param check_file_exists: if True, then check if file exists in local path
        :param bucket_name: if None, then use the default bucket.
        :return:
        """
        con_bucket = self._get_bucket_connection(bucket_name)
        if check_file_exists and os.path.exists(local_download_path):
            return local_download_path
        dir_name = os.path.dirname(local_download_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        con_bucket.download_file(Key=remote_key, Filename=local_download_path)
        return local_download_path