import logging
import os
import posixpath
from os.path import exists, join, basename, dirname
from typing import Optional, Dict, Any, Tuple
from urllib import parse
from urllib.parse import parse_qsl
from urllib.parse import urlsplit

import boto3


class Boto3Utils:
    def __init__(self,
                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
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
        assert aws_region, "aws_region is not provided"

        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = aws_region
        self.bucket_name = bucket_name
        self.con_s3, self.con_bucket = self.connect_s3(bucket_name)
        self.querystring_expire = querystring_expire
        self.querystring_auth = querystring_auth

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
