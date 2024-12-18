# test.py
import os
from dotenv import load_dotenv
import boto3
import tensorflow as tf

def test_setup():
    # Test environment variables
    load_dotenv()
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')
    
    print("AWS Credentials Check:")
    print(f"Access Key exists: {'Yes' if aws_key else 'No'}")
    print(f"Secret Key exists: {'Yes' if aws_secret else 'No'}")
    print(f"Region configured: {aws_region}")
    
    # Test AWS connection
    try:
        session = boto3.Session(
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=aws_region
        )
        client = session.client('cloudwatch')
        print("\nAWS Connection: Success!")
    except Exception as e:
        print("\nAWS Connection Error:", str(e))
    
    # Test TensorFlow
    print("\nTensorFlow version:", tf.__version__)

if __name__ == "__main__":
    test_setup()