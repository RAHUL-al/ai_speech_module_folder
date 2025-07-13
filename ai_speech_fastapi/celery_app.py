from celery import Celery
import os
from dotenv import load_dotenv
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")



redis_broker_url = f"redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
redis_backend_url = f"redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/1"

print(redis_broker_url)

celery = Celery(
    "tasks",
    broker=redis_broker_url,
    backend=redis_backend_url
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"]
)

