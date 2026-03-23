import os
from sqlalchemy import create_engine

def get_engine():
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}"
        f":{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}"
        f":{os.environ['POSTGRES_PORT']}"
        f"/{os.environ['POSTGRES_DB']}"
    )
    return create_engine(url)
