from pathlib import Path
import json
import requests
import pandas as pd
from datetime import datetime, timezone


#Configuration
BASE = "https://services.swpc.noaa.gov/json/solar-cycle"
FILENAME = "sunspots.json"
URL = f"{BASE}/{FILENAME}"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
METADATA_FILE = DATA_DIR / "metadata.json"
PARQUET_FILE = DATA_DIR / "sunspots.parquet"
 
