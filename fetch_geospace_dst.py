from pathlib import Path
import json
import requests
import pandas as pd
from datetime import datetime, timezone

# NOAA SWPC API
BASE = "https://services.swpc.noaa.gov/json/geospace"
FILENAME_1H = "geospace_dst_1_hour.json"
URL_1H = f"{BASE}/{FILENAME_1H}"

# directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

# metadata (last fetch time)
METADATA_1H = DATA_DIR / "metadata_dst_1h.json"

# processed dataset
PARQUET_1H = DATA_DIR / "dst_1h.parquet"

# NOAA SWPC API
FILENAME_7D = "geospace_dst_7_day.json"
URL_7D = f"{BASE}/{FILENAME_7D}"

# metadata
METADATA_7D = DATA_DIR / "metadata_dst_7d.json"

# processed dataset
PARQUET_7D = DATA_DIR / "dst_7d.parquet"