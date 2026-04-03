"""ingest.py — Solar/Space-Weather Data Downloader (consolidated)

Absorbs all download logic from the former download_data.py.
Nothing is cleaned or parsed here — raw bytes go to data/raw/.
clean.py handles all parsing, merging, and parquet output.

Modes
─────
python ingest.py            Full historical ingest. Skips files already on disk.
python ingest.py --latest   Incremental: only fast-changing recent feeds.

Sources
───────
1.  SILSO       — Daily sunspot number (space-sep .txt, modern)
2.  SILSO       — Daily sunspot number CSV (semicolon-sep, legacy format)
3.  SILSO       — Monthly sunspot number
4.  SILSO       — EISN recent (last few days)
5.  GFZ Potsdam — Kp flat file since 1932 (primary Kp source)
6.  NOAA SWPC   — Kp recent JSON (7-day supplement)
7.  NOAA NCEI   — Daily Solar Data (DSD) — F10.7 + flare counts
8.  NOAA NCEI   — Daily Geomagnetic Data (DGD) — Kp cross-check
9.  NASA SPDF   — OMNI2 hourly (Dst 1986-2004)
10. WDC Kyoto   — Monthly Dst (2005-present)
11. NOAA SWPC   — F10.7 recent JSON
12. NOAA SWPC   — X-ray flux 7-day JSON (flare supplement)
13. NOAA SWPC   — Geospace DST 7-day JSON (recent Dst supplement)
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import requests

RAW_DIR         = Path("data/raw")
START_YEAR      = 1986
TIMEOUT         = 90
RETRIES         = 3
BACKOFF         = 5
QUARTERLY_START = 2019

now_utc      = datetime.now(timezone.utc)
CURRENT_YEAR = now_utc.year
CURRENT_QTR  = (now_utc.month - 1) // 3 + 1

NCEI_BASE = (
    "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/"
    "annual_reports/daily_solar_indices_summaries/"
)

# ── URL registry ──────────────────────────────────────────────────────────────
URLS = {
    # SSN
    "silso_daily_txt":    "https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt",
    "silso_daily_csv":    "https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.csv",
    "silso_monthly":      "https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv",
    "silso_recent":       "https://www.sidc.be/silso/DATA/EISN/EISN_current.txt",
    # Kp
    "kp_gfz_flat_1":     "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt",
    "kp_gfz_flat_2":     "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt",
    "kp_noaa_recent":    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    # F10.7
    "f107_swpc_json":    "https://services.swpc.noaa.gov/json/f107_cm_flux.json",
    # Flares supplement
    "xray_7day_json":    "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
    # DST supplement
    "dst_7day_json":     "https://services.swpc.noaa.gov/json/geospace/geospace_dst_7_day.json",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


# ── Core download ─────────────────────────────────────────────────────────────

def _download(url: str, dest: Path, label: str,
              skip_existing: bool = True) -> Path:
    """Download url → dest with retry. Skips non-empty existing files."""
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        log.info(f"[{label}] on disk — skip")
        return dest
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        try:
            log.info(f"[{label}] {attempt}/{RETRIES} — {url}")
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(16384):
                    f.write(chunk)
            log.info(f"[{label}] saved {dest.name} ({dest.stat().st_size/1024:.1f} KB)")
            return dest
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                raise RuntimeError(f"[{label}] 404 permanent")
            log.warning(f"[{label}] HTTP {e.response.status_code}")
        except (requests.ConnectionError, requests.Timeout) as e:
            log.warning(f"[{label}] network: {e}")
        if attempt < RETRIES:
            time.sleep(BACKOFF)
    raise RuntimeError(f"[{label}] all {RETRIES} attempts failed")


def _try_mirrors(mirrors: List[str], dest: Path, label: str,
                 skip_existing: bool = True) -> Path:
    """Try multiple mirror URLs, return first success."""
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        log.info(f"[{label}] on disk — skip")
        return dest
    for url in mirrors:
        try:
            return _download(url, dest, label, skip_existing=False)
        except RuntimeError as e:
            log.warning(f"[{label}] mirror failed: {e}")
    raise RuntimeError(f"[{label}] all mirrors failed")


def _quarters(year: int) -> List[int]:
    return list(range(1, CURRENT_QTR + 1)) if year == CURRENT_YEAR else [1, 2, 3, 4]


def _fetch_ncei_yearly(base_url: str, suffix: str, label: str,
                        start: int = START_YEAR,
                        skip_existing: bool = True) -> List[Path]:
    """Fetch NCEI annual (pre-2019) or quarterly (2019+) files."""
    files = []
    for year in range(start, CURRENT_YEAR + 1):
        if year < QUARTERLY_START:
            fname = f"{year}_{suffix}.txt"
            try:
                files.append(_download(f"{base_url}{fname}", RAW_DIR / fname,
                                       f"{label}/{year}", skip_existing=skip_existing))
            except RuntimeError:
                log.warning(f"[{label}] {year} not available")
        else:
            for q in _quarters(year):
                fname = f"{year}Q{q}_{suffix}.txt"
                is_current = (year == CURRENT_YEAR)
                try:
                    files.append(_download(f"{base_url}{fname}", RAW_DIR / fname,
                                           f"{label}/{year}Q{q}",
                                           skip_existing=(skip_existing and not is_current)))
                except RuntimeError:
                    log.warning(f"[{label}] {year}Q{q} not available")
    return files


def _concat_files(files: List[Path], dest: Path) -> Path:
    with open(dest, "wb") as out:
        for f in files:
            with open(f, "rb") as inp:
                out.write(inp.read())
            out.write(b"\n")
    return dest


# ── Per-source fetchers ───────────────────────────────────────────────────────

def fetch_ssn(skip_existing: bool = False) -> List[Path]:
    """SILSO daily SSN in both formats (space-sep .txt + semicolon-sep .csv).
    Also fetches monthly and EISN recent.
    Returns list of paths saved."""
    saved = []
    # Primary: space-separated (used by clean.py parse_ssn)
    try:
        saved.append(_download(URLS["silso_daily_txt"],
                               RAW_DIR / "ssn_daily.txt", "SSN/SILSO-txt",
                               skip_existing=skip_existing))
    except RuntimeError as e:
        log.warning(f"SSN .txt failed: {e}")

    # Alternative: semicolon CSV (used by clean.py parse_ssn_csv)
    try:
        saved.append(_download(URLS["silso_daily_csv"],
                               RAW_DIR / "silso_sunspots_daily.csv", "SSN/SILSO-csv",
                               skip_existing=skip_existing))
    except RuntimeError as e:
        log.warning(f"SSN .csv failed: {e}")

    # Monthly
    try:
        saved.append(_download(URLS["silso_monthly"],
                               RAW_DIR / "silso_sunspots_monthly.csv", "SSN/monthly",
                               skip_existing=skip_existing))
    except RuntimeError as e:
        log.warning(f"SSN monthly failed: {e}")

    # EISN recent (last few days, highest fidelity)
    try:
        saved.append(_download(URLS["silso_recent"],
                               RAW_DIR / "silso_sunspots_recent.txt", "SSN/EISN",
                               skip_existing=False))   # always re-fetch
    except RuntimeError as e:
        log.warning(f"SSN EISN failed: {e}")

    return saved


def fetch_kp_flatfile(skip_existing: bool = False) -> Path:
    """GFZ Kp flat file — all Kp/ap/F10.7/SN since 1932 in one ~6 MB file."""
    return _try_mirrors(
        [URLS["kp_gfz_flat_1"], URLS["kp_gfz_flat_2"]],
        RAW_DIR / "kp_historic_gfz.txt",
        "Kp/GFZ-flat", skip_existing=skip_existing,
    )


def fetch_kp_noaa_recent() -> Path:
    """NOAA SWPC Kp JSON — last 7 days (always re-fetched)."""
    return _download(URLS["kp_noaa_recent"],
                     RAW_DIR / "noaa_kp_index.json",
                     "Kp/NOAA-SWPC", skip_existing=False)


def fetch_flares_ncei(skip_existing: bool = True) -> Path:
    """NCEI DSD files — daily F10.7 + flare counts (1986-present)."""
    files = _fetch_ncei_yearly(f"{NCEI_BASE}daily_solar_data/", "DSD", "DSD",
                                skip_existing=skip_existing)
    if not files:
        raise RuntimeError("[DSD] no files downloaded")
    return _concat_files(files, RAW_DIR / "dsd_1996_present.txt")


def fetch_xray_recent() -> Path:
    """NOAA SWPC X-ray flux 7-day JSON — supplement for recent flare data."""
    return _download(URLS["xray_7day_json"],
                     RAW_DIR / "noaa_xray_7day.json",
                     "Xray/NOAA-7day", skip_existing=False)


def fetch_dgd(skip_existing: bool = True) -> Path:
    """NCEI Daily Geomagnetic Data — Kp cross-check (1986-present)."""
    files = _fetch_ncei_yearly(f"{NCEI_BASE}daily_geomagnetic_data/", "DGD", "DGD",
                                skip_existing=skip_existing)
    if not files:
        raise RuntimeError("[DGD] no files downloaded")
    return _concat_files(files, RAW_DIR / "dgd_1996_present.txt")


def fetch_dst_omni2(skip_existing: bool = True) -> List[Path]:
    """NASA OMNI2 hourly Dst 1986-2004 (historical; skip if on disk)."""
    base  = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"
    files = []
    for yr in range(START_YEAR, 2005):
        fname = f"omni2_{yr}.dat"
        try:
            files.append(_download(f"{base}{fname}", RAW_DIR / fname,
                                   f"OMNI2/{yr}", skip_existing=skip_existing))
        except RuntimeError:
            log.warning(f"[OMNI2] {yr} not available")
    return files


def fetch_dst_kyoto(latest_only: bool = False,
                    skip_existing: bool = True) -> Path:
    """Kyoto WDC monthly Dst (2005-present)."""
    KYOTO_FINAL = 2020
    dest = RAW_DIR / "dst_kyoto_hourly.txt"
    kyoto_files: List[Path] = []

    start_year  = now_utc.year if latest_only else 2005
    start_month = max(1, now_utc.month - 3) if latest_only else 1

    for year in range(start_year, CURRENT_YEAR + 1):
        sm = start_month if year == start_year else 1
        em = 12 if year < CURRENT_YEAR else now_utc.month
        for month in range(sm, em + 1):
            tier  = "dst_final" if year <= KYOTO_FINAL else "dst_provisional"
            ym    = f"{year}{month:02d}"
            yy_mm = f"{year % 100:02d}{month:02d}"
            url   = f"https://wdc.kugi.kyoto-u.ac.jp/{tier}/{ym}/dst{yy_mm}.for.request"
            mdest = RAW_DIR / f"dst_kyoto_{ym}.txt"
            is_current = (year == CURRENT_YEAR and month == now_utc.month)
            try:
                kyoto_files.append(
                    _download(url, mdest, f"Dst/{ym}",
                              skip_existing=(skip_existing and not is_current)))
            except RuntimeError:
                log.warning(f"[Dst/Kyoto] {ym} not available")
            time.sleep(0.4)

    if kyoto_files:
        all_monthly = sorted(RAW_DIR.glob("dst_kyoto_????.txt"))
        if all_monthly:
            _concat_files(all_monthly, dest)
    return dest


def fetch_dst_noaa_recent() -> Path:
    """NOAA SWPC geospace DST 7-day JSON — recent DST supplement."""
    return _download(URLS["dst_7day_json"],
                     RAW_DIR / "noaa_dst_7day.json",
                     "Dst/NOAA-7day", skip_existing=False)


def fetch_f107(skip_existing: bool = False) -> Path:
    """NOAA SWPC F10.7 rolling JSON (~45 days). Always re-fetched."""
    return _download(URLS["f107_swpc_json"],
                     RAW_DIR / "f107_recent.json",
                     "F10.7/SWPC", skip_existing=skip_existing)


# ── Orchestrators ─────────────────────────────────────────────────────────────

def _run_fetchers(fetchers: dict) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, fn in fetchers.items():
        try:
            path = fn()
            paths = path if isinstance(path, list) else [path]
            size  = sum(Path(p).stat().st_size for p in paths
                        if p and Path(p).exists()) / 1024
            results[name] = {"success": True, "size_kb": size}
        except Exception as e:
            log.error(f"[{name}] FAILED: {e}")
            results[name] = {"success": False, "error": str(e)}

    ok = sum(1 for r in results.values() if r["success"])
    log.info(f"Complete: {ok}/{len(results)} sources OK")
    for name, r in results.items():
        log.info(f"  {'OK' if r['success'] else 'FAIL'} {name}")

    ts = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    with open(RAW_DIR / "MANIFEST.txt", "w") as f:
        f.write(f"Solar Ingest Manifest\nGenerated: {ts}\n{'='*40}\n")
        for name, r in results.items():
            f.write(f"[{'OK' if r['success'] else 'FAIL'}] {name}\n")
            if not r["success"]:
                f.write(f"  Error: {r['error']}\n")
    return results


# Full historical ingest — skip files already on disk
FETCHERS_FULL = {
    "SSN (SILSO all formats)":    lambda: fetch_ssn(skip_existing=True),
    "Kp flat file (GFZ)":         lambda: fetch_kp_flatfile(skip_existing=True),
    "Kp recent (NOAA)":           lambda: fetch_kp_noaa_recent(),
    "F10.7 + Flares DSD (NCEI)":  lambda: fetch_flares_ncei(skip_existing=True),
    "X-ray 7-day (NOAA)":         lambda: fetch_xray_recent(),
    "DGD/Kp (NCEI)":              lambda: fetch_dgd(skip_existing=True),
    "Dst OMNI2 (NASA)":           lambda: fetch_dst_omni2(skip_existing=True),
    "Dst Kyoto (WDC)":            lambda: fetch_dst_kyoto(latest_only=False, skip_existing=True),
    "Dst recent (NOAA)":          lambda: fetch_dst_noaa_recent(),
    "F10.7 JSON (NOAA)":          lambda: fetch_f107(),
}

# Incremental — only fast-changing / recent feeds
FETCHERS_LATEST = {
    "SSN (SILSO all formats)":    lambda: fetch_ssn(skip_existing=False),
    "Kp flat file (GFZ)":         lambda: fetch_kp_flatfile(skip_existing=False),
    "Kp recent (NOAA)":           lambda: fetch_kp_noaa_recent(),
    "DGD current quarter":        lambda: fetch_dgd(skip_existing=True),
    "Dst Kyoto last 3 months":    lambda: fetch_dst_kyoto(latest_only=True, skip_existing=True),
    "Dst recent (NOAA)":          lambda: fetch_dst_noaa_recent(),
    "F10.7 JSON (NOAA)":          lambda: fetch_f107(),
    "X-ray 7-day (NOAA)":         lambda: fetch_xray_recent(),
}


def run_all()     -> dict:
    log.info("Solar ingest — FULL"); return _run_fetchers(FETCHERS_FULL)

def fetch_latest() -> dict:
    log.info("Solar ingest — LATEST"); return _run_fetchers(FETCHERS_LATEST)


if __name__ == "__main__":
    fetch_latest() if "--latest" in sys.argv else run_all()
