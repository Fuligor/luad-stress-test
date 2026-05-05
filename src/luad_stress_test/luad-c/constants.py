from dotenv import load_dotenv
from os import getenv
from pathlib import Path

load_dotenv()

ADN_DIR = Path(getenv("ADN_DIR")) # type: ignore
IMG_DIR = Path(getenv("IMG_DIR")) # type: ignore
PROCESSED_DIR = Path(getenv("PROCESSED_DIR")) # type: ignore