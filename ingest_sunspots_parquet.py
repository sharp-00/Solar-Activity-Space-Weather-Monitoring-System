from pathlib import Path
import json
import requests
import pandas as pd
from datetime import datetime, timezone


#importing the data
BASE = "https://services.swpc.noaa.gov/json/solar-cycle"
FILENAME = "sunspots.json"
URL = f"{BASE}/{FILENAME}"

#if files are missing
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

#to store last fetch time
METADATA_FILE = DATA_DIR / "metadata.json"

#time-indexed datasheet
PARQUET_FILE = DATA_DIR / "sunspots.parquet"
 
