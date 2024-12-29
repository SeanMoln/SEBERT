import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

try:
    s3.upload_file('path/to/your/file.wav', 'moln9110', 'audio-file.wav')
    print("File uploaded successfully.")
except ClientError as e:
    print(f"Error occurred: {e}")
