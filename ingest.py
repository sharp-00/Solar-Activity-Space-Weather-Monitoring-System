"""
ingest.py  —  v5
-----------------
Downloads raw solar and space weather data from authoritative sources.

All URLs verified against NCEI SWPC Products and Data documentation:
https://www.ncei.noaa.gov/products/space-weather/partners/swpc-products-and-data

Sources:
    1. SILSO       — Daily Sunspot Number (SSN)
    2. NOAA NCEI   — Solar flare events via Daily Solar Data (DSD) + event reports
    3. GFZ API     — Kp 3-hourly index
    4. NOAA NCEI   — Daily Geomagnetic Data (DGD) for Kp cross-check
    5. NASA OMNI   — Dst index hourly
    6. NOAA NCEI   — Daily Solar Data (DSD) for F10.7 historical series

Architecture change from v3:
    Flares 2017+: NOAA does NOT publish yearly combined flare files after 2016.
    Post-2016 flares are in daily event report files (YYYYMMDDevents.txt).
    Fetching ~3000 individual daily files is impractical for an ingest script.

    SOLUTION: Use NCEI's Daily Solar Data (DSD) yearly files which include
    a daily flare count (total optical + X-ray flares per day). This gives us
    the daily flare activity time series for the full 1996-present range from
    a single confirmed URL pattern. For individual flare event-level data
    (class, time, region), we supplement with the SWPC 7-day live JSON feed
    and flag the limitation explicitly in the data limitations section.

    This is also more honest analytically — daily counts align better with
    our daily resolution grid than raw event lists anyway.

Usage:
    python ingest.py
"""

import time
import logging
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")

START_YEAR   = 1986
CURRENT_YEAR = datetime.now(timezone.utc).year
CURRENT_QTR  = (datetime.now(timezone.utc).month - 1) // 3 + 1

TIMEOUT        = 90
RETRY_ATTEMPTS = 3
RETRY_BACKOFF  = 5

# Year where NCEI switched from annual to quarterly filenames
QUARTERLY_START_YEAR = 2019

# Confirmed base URLs from NCEI documentation
NCEI_DAILY_REPORTS = (
    "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/annual_reports/"
    "daily_solar_indices_summaries/"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Raw data directory: {RAW_DIR.resolve()}")


def _download(url: str, dest: Path, label: str) -> Path:
    """HTTP GET with retry. Streams to disk. Returns dest on success."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            log.info(f"[{label}] Attempt {attempt}/{RETRY_ATTEMPTS} — {url}")
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            r.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=16384):
                    f.write(chunk)

            size_kb = dest.stat().st_size / 1024
            log.info(f"[{label}] ✓ {dest.name} ({size_kb:.1f} KB)")
            return dest

        except requests.exceptions.HTTPError as e:
            log.warning(f"[{label}] HTTP {e.response.status_code}: {e}")
            if e.response.status_code == 404:
                raise RuntimeError(f"[{label}] 404 Not Found (Permanent). No retries.")
        except requests.exceptions.ConnectionError as e:
            log.warning(f"[{label}] Connection error: {e}")
        except requests.exceptions.Timeout:
            log.warning(f"[{label}] Timed out after {TIMEOUT}s")

        if attempt < RETRY_ATTEMPTS:
            log.info(f"[{label}] Retrying in {RETRY_BACKOFF}s ...")
            time.sleep(RETRY_BACKOFF)

    raise RuntimeError(f"[{label}] All {RETRY_ATTEMPTS} attempts failed. URL: {url}")


def _quarters_for_year(year: int) -> List[int]:
    """Return the list of quarters to fetch for a given year.
    For the current year, only return quarters up to the current one.
    """
    if year < CURRENT_YEAR:
        return [1, 2, 3, 4]
    else:  # year == CURRENT_YEAR
        return list(range(1, CURRENT_QTR + 1))


def _fetch_ncei_yearly_files(
    base_url: str,
    file_suffix: str,
    label_prefix: str,
    start_year: int = START_YEAR,
    end_year: int | None = None,
) -> List[Path]:
    """
    Generic fetcher for NCEI yearly/quarterly indexed files (DSD or DGD).

    NCEI naming convention:
      - 1994–2018: annual files  →  {YYYY}_{suffix}.txt
      - 2019+:     quarterly     →  {YYYY}Q{1-4}_{suffix}.txt

    Returns the list of successfully downloaded file paths.
    """
    if end_year is None:
        end_year = CURRENT_YEAR

    downloaded: List[Path] = []

    for year in range(start_year, end_year + 1):
        if year < QUARTERLY_START_YEAR:
            # Annual file
            fname = f"{year}_{file_suffix}.txt"
            dest = RAW_DIR / f"{file_suffix.lower()}_{year}.txt"
            try:
                _download(
                    url=f"{base_url}{fname}",
                    dest=dest,
                    label=f"{label_prefix}/{year}",
                )
                downloaded.append(dest)
            except RuntimeError:
                log.warning(f"[{label_prefix}/{year}] Not found — gap noted.")
        else:
            # Quarterly files
            for q in _quarters_for_year(year):
                fname = f"{year}Q{q}_{file_suffix}.txt"
                dest = RAW_DIR / f"{file_suffix.lower()}_{year}q{q}.txt"
                try:
                    _download(
                        url=f"{base_url}{fname}",
                        dest=dest,
                        label=f"{label_prefix}/{year}Q{q}",
                    )
                    downloaded.append(dest)
                except RuntimeError:
                    log.warning(f"[{label_prefix}/{year}Q{q}] Not found — gap noted.")

    return downloaded


# ---------------------------------------------------------------------------
# Source 1: SILSO Daily Sunspot Number
# ---------------------------------------------------------------------------

def fetch_ssn() -> Path:
    """
    SILSO Daily Total Sunspot Number (Version 2.0).

    Format: year month day decimal_year SSN std_dev n_obs definitive_flag
    Coverage: 1818-present. Filtered to 1996+ in clean.py.
    Source: https://www.sidc.be/SILSO/datafiles
    """
    dest = RAW_DIR / "ssn_daily.txt"
    return _download(
        url="https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.txt",
        dest=dest,
        label="SSN/SILSO",
    )


# ---------------------------------------------------------------------------
# Source 2: NOAA GOES XRS Flares — legacy yearly (1996-2016) + DSD daily counts
# ---------------------------------------------------------------------------

def fetch_flares() -> Path:
    """
    Download solar flare data using a two-tier strategy.

    TIER 1 — Event-level data (2008-2016):
        NCEI/NGDC legacy XRS archive, confirmed working through 2016.
        URL: .../solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_{YEAR}.txt
        Each row = one flare event with class, time, peak flux, active region.

    TIER 2 — Daily flare counts (1996-present):
        NCEI Daily Solar Data (DSD) yearly files.
        URL confirmed: .../daily_solar_indices_summaries/daily_solar_data/{YYYY}_DSD.txt
        Contains daily total flare count (optical + X-ray), F10.7, SSN, X-ray background.
        Coverage: 1994 to present. Filename: yyyy_DSD.txt

    Why DSD for 2017+:
        Post-2016 NOAA flare data is stored as individual daily event report files
        (YYYYMMDDevents.txt), making bulk download impractical (~3300 files).
        DSD provides daily flare counts in a clean yearly format, which aligns
        perfectly with our daily resolution grid. Event-level detail for recent
        flares is supplemented by the SWPC 7-day live JSON feed.

    Data limitation (to document in report):
        Event-level flare class data (B/C/M/X breakdown) is only available
        for 2008-2016 from the legacy archive. For 2017+, we have daily counts
        only. This limits event-level analysis to the 2008-2016 window.
    """
    year_files_legacy = []

    # --- Tier 1: Legacy event-level XRS files (2008-2016) ---
    legacy_base = (
        "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
        "solar-features/solar-flares/x-rays/goes/xrs/"
    )
    log.info("[Flares] Tier 1: NGDC legacy XRS event files (1996-2016)")
    for year in range(START_YEAR, 2017):
        dest = RAW_DIR / f"flares_{year}.txt"
        try:
            _download(
                url=f"{legacy_base}goes-xrs-report_{year}.txt",
                dest=dest,
                label=f"Flares-XRS/{year}",
            )
            year_files_legacy.append(dest)
        except RuntimeError:
            log.warning(f"[Flares-XRS/{year}] Not found — gap noted.")

    # Concatenate legacy event files
    if year_files_legacy:
        concat = RAW_DIR / "flares_events_1996_2016.txt"
        with open(concat, "wb") as out:
            for yf in year_files_legacy:
                with open(yf, "rb") as inp:
                    out.write(inp.read())
                out.write(b"\n")
        log.info(f"[Flares] ✓ Legacy events concatenated → {concat}")

    # --- Tier 2: DSD yearly/quarterly files (2008-present, daily flare counts) ---
    dsd_base = f"{NCEI_DAILY_REPORTS}daily_solar_data/"
    log.info("[Flares/DSD] Tier 2: NCEI Daily Solar Data (1996-present)")
    log.info("[Flares/DSD] Note: annual files 1996-2018, quarterly 2019+")
    dsd_files = _fetch_ncei_yearly_files(
        base_url=dsd_base,
        file_suffix="DSD",
        label_prefix="DSD",
    )

    # Concatenate DSD files
    if dsd_files:
        dsd_concat = RAW_DIR / "dsd_1996_present.txt"
        with open(dsd_concat, "wb") as out:
            for yf in dsd_files:
                with open(yf, "rb") as inp:
                    out.write(inp.read())
                out.write(b"\n")
        log.info(f"[DSD] ✓ {len(dsd_files)} files concatenated → {dsd_concat}")

    # --- Supplement: SWPC 7-day live JSON feed ---
    dest_recent = RAW_DIR / "flares_recent_7day.json"
    try:
        _download(
            url="https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json",
            dest=dest_recent,
            label="Flares/SWPC-7day",
        )
    except RuntimeError as e:
        log.warning(f"[Flares] SWPC 7-day feed failed: {e}")

    if not year_files_legacy and not dsd_files:
        raise RuntimeError(
            "[Flares] No flare data downloaded from any source. "
            "Check network connectivity to www.ngdc.noaa.gov."
        )

    # Return primary output path (DSD is more complete)
    return RAW_DIR / "dsd_1996_present.txt" if dsd_files else RAW_DIR / "flares_events_1996_2016.txt"


# ---------------------------------------------------------------------------
# Source 3: GFZ Kp Index — JSON API
# ---------------------------------------------------------------------------

def fetch_kp() -> Path:
    """
    Kp/ap/Ap/SN/F10.7 from GFZ JSON API at kp.gfz.de.

    Confirmed endpoint (replaces dead fileadmin flat-file URL).
    Returns 3-hourly Kp + daily Ap, SN (cross-check), F10.7 (Fobs/Fadj).
    """
    dest = RAW_DIR / "kp_gfz_api.json"

    start_str = f"{START_YEAR}-01-01T00:00:00Z"
    end_str   = datetime.now(timezone.utc).strftime("%Y-%m-%dT23:59:59Z")

    url = (
        "https://kp.gfz.de/app/json/"
        f"?start={start_str}"
        f"&end={end_str}"
        "&index=Kp,ap,Ap,SN,Fobs,Fadj"
        "&status=def,nowcast"
    )
    return _download(url=url, dest=dest, label="Kp/GFZ-API")


# ---------------------------------------------------------------------------
# Source 4: NCEI Daily Geomagnetic Data (DGD) — Kp/Ap cross-check + daily Kp
# ---------------------------------------------------------------------------

def fetch_dgd() -> Path:
    """
    Download NCEI Daily Geomagnetic Data (DGD) yearly/quarterly files.

    Confirmed URL from NCEI documentation:
        Annual  (1994-2018): .../daily_geomagnetic_data/{YYYY}_DGD.txt
        Quarterly (2019+):   .../daily_geomagnetic_data/{YYYY}Q{1-4}_DGD.txt

    Coverage: 1994 to present.

    Contains:
        - Daily 24-hour A index (Fredericksburg + College + Estimated planetary)
        - 8 x 3-hourly K indices (Fredericksburg, College, Estimated planetary Kp)

    Used as:
        - Cross-check against GFZ Kp API values
        - Fallback daily Kp source if GFZ API fails
        - Source of Fredericksburg/College K-indices for regional context
    """
    dgd_base = f"{NCEI_DAILY_REPORTS}daily_geomagnetic_data/"

    log.info("[DGD] Fetching NCEI Daily Geomagnetic Data (1996-present)")
    log.info("[DGD] Note: annual files 1996-2018, quarterly 2019+")
    dgd_files = _fetch_ncei_yearly_files(
        base_url=dgd_base,
        file_suffix="DGD",
        label_prefix="DGD",
    )

    if not dgd_files:
        raise RuntimeError(
            "[DGD] No DGD files downloaded. "
            "Check: https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/"
            "annual_reports/daily_solar_indices_summaries/daily_geomagnetic_data/"
        )

    concat = RAW_DIR / "dgd_1996_present.txt"
    with open(concat, "wb") as out:
        for yf in dgd_files:
            with open(yf, "rb") as inp:
                out.write(inp.read())
            out.write(b"\n")

    log.info(f"[DGD] ✓ {len(dgd_files)} files → {concat}")
    return concat


# ---------------------------------------------------------------------------
# Source 5: Dst Index — NASA OMNI2 (1996-2004) + Kyoto WDC (2005+)
# ---------------------------------------------------------------------------

# Kyoto WDC only reliably serves .for.request files from ~2005 onwards.
# For 1996-2004 we use NASA SPDF OMNI2 yearly bulk files instead.
KYOTO_START_YEAR       = 2005
KYOTO_FINAL_LAST_YEAR  = 2020   # dst_final goes through 2020-12
KYOTO_PROV_FIRST_YEAR  = 2021   # dst_provisional starts 2021-01

# OMNI2 covers the historical gap
OMNI2_START_YEAR = START_YEAR   # 1996
OMNI2_END_YEAR   = 2004         # inclusive


def fetch_dst_omni2() -> List[Path]:
    """
    Download NASA SPDF OMNI2 hourly data files for 1996-2004.

    Each file contains all heliospheric variables at hourly resolution.
    Dst is column index 40 (0-based). ~1.5 MB per year.

    URL: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_YYYY.dat
    """
    omni_files: List[Path] = []
    base = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"

    log.info(f"[Dst/OMNI2] Fetching NASA SPDF OMNI2 yearly files ({OMNI2_START_YEAR}-{OMNI2_END_YEAR})")

    for year in range(OMNI2_START_YEAR, OMNI2_END_YEAR + 1):
        fname = f"omni2_{year}.dat"
        dest = RAW_DIR / fname
        try:
            _download(url=f"{base}{fname}", dest=dest, label=f"Dst/OMNI2/{year}")
            omni_files.append(dest)
        except RuntimeError:
            log.warning(f"[Dst/OMNI2] {year}: not available — gap noted.")

    log.info(f"[Dst/OMNI2] ✓ {len(omni_files)} yearly files downloaded")
    return omni_files


def _kyoto_dst_months() -> List[tuple]:
    """
    Generate (year, month, tier) tuples for Kyoto Dst WDC downloads.
    Only generates entries from KYOTO_START_YEAR (2005) onwards.
    """
    now = datetime.now(timezone.utc)
    entries = []

    for year in range(KYOTO_START_YEAR, now.year + 1):
        start_month = 1
        end_month = 12 if year < now.year else now.month

        for month in range(start_month, end_month + 1):
            if year <= KYOTO_FINAL_LAST_YEAR:
                tier = "dst_final"
            else:
                tier = "dst_provisional"
            entries.append((year, month, tier))

    return entries


def fetch_dst() -> Path:
    """
    Hourly Dst index — dual-source strategy:

    Phase 1 (1996-2004): NASA SPDF OMNI2 yearly bulk files.
        Reliable, fast (9 files), no 404 issues.

    Phase 2 (2005-present): Kyoto WDC monthly files.
        Final Dst (2005-2020):
            https://wdc.kugi.kyoto-u.ac.jp/dst_final/YYYYMM/dstYYMM.for.request
        Provisional Dst (2021-present):
            https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/YYYYMM/dstYYMM.for.request
    """
    # --- Phase 1: OMNI2 for 1996-2004 ---
    omni_files = fetch_dst_omni2()

    # --- Phase 2: Kyoto WDC for 2005+ ---
    dest = RAW_DIR / "dst_kyoto_hourly.txt"
    months = _kyoto_dst_months()
    kyoto_files: List[Path] = []

    log.info(f"[Dst/Kyoto] Fetching Kyoto WDC Dst (final 2005-2020, provisional 2021+)")
    log.info(f"[Dst/Kyoto] Total months to fetch: {len(months)}")

    for year, month, tier in months:
        ym = f"{year}{month:02d}"
        yy = f"{year % 100:02d}"
        mm = f"{month:02d}"
        fname = f"dst{yy}{mm}.for.request"
        url = f"https://wdc.kugi.kyoto-u.ac.jp/{tier}/{ym}/{fname}"
        month_dest = RAW_DIR / f"dst_kyoto_{ym}.txt"

        try:
            _download(url=url, dest=month_dest, label=f"Dst/{tier}/{ym}")
            kyoto_files.append(month_dest)
        except RuntimeError:
            log.warning(f"[Dst] {ym} ({tier}): not available — gap noted.")

        # Be polite to Kyoto servers
        time.sleep(0.5)

    if kyoto_files:
        with open(dest, "wb") as out:
            for yf in kyoto_files:
                with open(yf, "rb") as inp:
                    out.write(inp.read())
                out.write(b"\n")
        log.info(f"[Dst/Kyoto] ✓ {len(kyoto_files)}/{len(months)} months → {dest}")

    total = len(omni_files) + len(kyoto_files)
    if total == 0:
        raise RuntimeError("[Dst] All sources failed for both OMNI2 and Kyoto.")

    log.info(f"[Dst] ✓ Total coverage: {len(omni_files)} OMNI2 years + {len(kyoto_files)} Kyoto months")
    return dest


# ---------------------------------------------------------------------------
# Source 6: F10.7 — NCEI DSD (historical) + NOAA SWPC (recent supplement)
# ---------------------------------------------------------------------------

def fetch_f107() -> Path:
    """
    F10.7 solar flux.

    Historical (1996-present): extracted from DSD files downloaded in fetch_flares.
        DSD column 'Adjusted F10.7' is the standard adjusted-to-1AU value.
        Already saved as dsd_1996_present.txt — no extra download needed.

    Recent supplement (~45 days): NOAA SWPC JSON feed.
        Fills the gap between latest DSD file and today.

    harmonise.py merges both into a single daily F10.7 series.
    """
    dest = RAW_DIR / "f107_recent.json"
    return _download(
        url="https://services.swpc.noaa.gov/json/f107_cm_flux.json",
        dest=dest,
        label="F10.7/NOAA-SWPC",
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(results: dict) -> None:
    manifest_path = RAW_DIR / "MANIFEST.txt"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(manifest_path, "w") as f:
        f.write("Solar Activity Monitoring — Ingest Manifest\n")
        f.write(f"Generated : {ts}\n")
        f.write(f"Version   : v5 (multi-source Dst)\n")
        f.write(f"Period    : {START_YEAR}-present\n")
        f.write("=" * 60 + "\n\n")
        f.write("NOTE: Flare event-level data (class/time) only available\n")
        f.write("      for 1996-2016 from NGDC legacy XRS archive. Flare counts\n")
        f.write("      will be NaN from 1986-1993.\n\n")
        for source, info in results.items():
            status = "OK" if info["success"] else "FAILED"
            f.write(f"[{status}] {source}\n")
            if info["success"]:
                f.write(f"  Path : {info['path']}\n")
                f.write(f"  Size : {info['size_kb']:.1f} KB\n")
            else:
                f.write(f"  Error: {info['error']}\n")
            f.write("\n")

    log.info(f"Manifest written → {manifest_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all() -> dict:
    """Run all ingest functions. Returns per-source result dict."""
    ensure_dirs()

    fetchers = {
        "SSN (SILSO)"              : fetch_ssn,
        "Flares + DSD (NCEI)"      : fetch_flares,
        "Kp (GFZ API)"             : fetch_kp,
        "DGD/Kp cross-check (NCEI)": fetch_dgd,
        "Dst (NASA OMNI)"          : fetch_dst,
        "F10.7 recent (NOAA SWPC)" : fetch_f107,
    }

    results = {}
    log.info("=" * 60)
    log.info("Starting ingest pipeline  [v5 — multi-source Dst]")
    log.info("=" * 60)

    for name, fetcher in fetchers.items():
        try:
            path = fetcher()
            results[name] = {
                "success" : True,
                "path"    : str(path),
                "size_kb" : path.stat().st_size / 1024,
            }
        except Exception as e:
            log.error(f"[{name}] FAILED: {e}")
            results[name] = {"success": False, "error": str(e), "size_kb": 0}

    log.info("=" * 60)
    n_ok = sum(1 for r in results.values() if r["success"])
    log.info(f"Ingest complete: {n_ok}/{len(results)} sources successful")
    for name, info in results.items():
        log.info(f"  {'✓' if info['success'] else '✗'} {name}")
    log.info("=" * 60)

    write_manifest(results)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all()
