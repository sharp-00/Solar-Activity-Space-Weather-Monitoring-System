"""clean.py — Data Cleaning & Harmonisation Pipeline (consolidated)

Absorbs all Kp cleaning and pipeline logic from the former download_data.py.
Reads raw files from data/raw/ and writes parquets to data/clean/.

Modes
─────
python clean.py           Full rebuild from all raw files → parquets
python clean.py --update  Merge recent raw data into existing parquets (fast)

Output parquets (data/clean/)
──────────────────────────────
solar_weather_daily.parquet   — master merged dataset
sunspots_daily_clean.parquet  — SSN (SILSO, multiple formats)
kp_daily_clean.parquet        — Kp/ap index (GFZ + NOAA SWPC + DGD fallback)
dst_daily_clean.parquet       — Dst (OMNI2 + Kyoto + NOAA 7-day)
f107_daily_clean.parquet      — F10.7 solar flux
flares_daily_clean.parquet    — Solar flare counts
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

RAW_DIR    = Path("data/raw")
CLEAN_DIR  = Path("data/clean")
START_DATE = "1986-01-01"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Generic helpers ───────────────────────────────────────────────────────────

def _resample_alias(freq: str) -> str:
    """Correct resample alias for installed pandas version."""
    _MAP = {"M": "ME", "Y": "YE", "Q": "QE", "A": "YE"}
    major, minor = (int(x) for x in pd.__version__.split(".")[:2])
    return _MAP.get(freq, freq) if (major, minor) >= (2, 2) else freq


def finalize(df: pd.DataFrame, keep: list) -> pd.DataFrame:
    """Build date index, filter, dedup, keep only requested columns."""
    if "year" in df.columns and "month" in df.columns and "day" in df.columns:
        df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    if "date" not in df.columns:
        return pd.DataFrame()
    df = (df.dropna(subset=["date"])
            .query("date >= @START_DATE")
            .drop_duplicates(subset=["date"], keep="last"))
    valid = [c for c in keep if c in df.columns and c != "date"]
    return df[["date"] + valid].set_index("date").sort_index()


def to_num(df: pd.DataFrame, cols: list, missing) -> pd.DataFrame:
    """Coerce cols to numeric; replace sentinel values with NaN."""
    present = [c for c in cols if c in df.columns]
    if present:
        df[present] = df[present].apply(pd.to_numeric, errors="coerce")
        df[present] = df[present].replace(missing, np.nan)
    return df


def _save(df: pd.DataFrame, stem: str) -> None:
    """Write DataFrame as parquet + CSV to CLEAN_DIR."""
    if df.empty:
        log.warning(f"[save] {stem} is empty — skipping")
        return
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_DIR / f"{stem}.parquet")
    df.to_csv(CLEAN_DIR / f"{stem}.csv", float_format="%.4f")
    log.info(f"  {stem}: {len(df):,} rows, {len(df.columns)} cols")


def _load_parquet(stem: str) -> pd.DataFrame:
    p = CLEAN_DIR / f"{stem}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _merge_into(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge new data into existing; new rows override on overlap."""
    if existing.empty: return new
    if new.empty:      return existing
    for col in new.columns:
        if col not in existing.columns:
            existing[col] = np.nan
    combined = pd.concat([existing, new])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def _load_if(path: Path, parser):
    return parser(path) if path.exists() else pd.DataFrame()


# ── SSN parsers ───────────────────────────────────────────────────────────────

def parse_ssn_txt(path: Path) -> pd.DataFrame:
    """SILSO space-separated daily file (SN_d_tot_V2.0.txt)."""
    log.info(f"[SSN] Parsing SILSO txt: {path.name}")
    df = pd.read_csv(path, sep=r"\s+", header=None, na_values=[-1, -1.0],
                     names=["year","month","day","dec_yr","ssn","ssn_std","n","def"])
    return finalize(df, ["ssn", "ssn_std"])


def parse_ssn_csv(path: Path) -> pd.DataFrame:
    """SILSO semicolon-separated daily CSV (SN_d_tot_V2.0.csv / silso_sunspots_daily.csv)."""
    log.info(f"[SSN] Parsing SILSO csv: {path.name}")
    try:
        df = pd.read_csv(path, sep=";", header=None,
                         names=["year","month","day","frac_year","sn","sn_err","n_obs","definitive"])
        df["ssn"]     = pd.to_numeric(df["sn"],     errors="coerce").replace(-1, np.nan)
        df["ssn_std"] = pd.to_numeric(df["sn_err"], errors="coerce").replace(-1, np.nan)
        return finalize(df, ["ssn", "ssn_std"])
    except Exception as e:
        log.warning(f"[SSN/csv] {e}")
        return pd.DataFrame()


def parse_ssn_recent(path: Path) -> pd.DataFrame:
    """SILSO EISN recent file (last few days, highest priority for latest dates)."""
    log.info(f"[SSN] Parsing SILSO EISN recent: {path.name}")
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split()
                if len(parts) < 4: continue
                try:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    ssn = float(parts[4]) if len(parts) > 4 and parts[4] not in ("-1","999") else np.nan
                    ssn_std = float(parts[5]) if len(parts) > 5 and parts[5] not in ("-1","999") else np.nan
                    rows.append({"year": year, "month": month, "day": day,
                                 "ssn": ssn, "ssn_std": ssn_std})
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        log.warning(f"[SSN/recent] {e}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return finalize(df, ["ssn", "ssn_std"]) if not df.empty else pd.DataFrame()


def build_ssn(raw_dir: Path) -> pd.DataFrame:
    """Merge all SSN sources: txt > csv > EISN recent (newest dates win)."""
    sources = []

    # Primary: space-separated txt
    for fname in ["ssn_daily.txt"]:
        p = raw_dir / fname
        if p.exists():
            df = parse_ssn_txt(p)
            if not df.empty:
                sources.append(df)
                break

    # Secondary: semicolon CSV (same data, different format)
    for fname in ["silso_sunspots_daily.csv"]:
        p = raw_dir / fname
        if p.exists() and not sources:
            df = parse_ssn_csv(p)
            if not df.empty:
                sources.append(df)
                break

    # Recent supplement: EISN (highest fidelity for last few days)
    for fname in ["silso_sunspots_recent.txt"]:
        p = raw_dir / fname
        if p.exists():
            df = parse_ssn_recent(p)
            if not df.empty:
                sources.append(df)

    if not sources:
        log.warning("[SSN] No source found")
        return pd.DataFrame()

    result = sources[0]
    for extra in sources[1:]:
        result = _merge_into(result, extra)
    return result.sort_index()


# ── Kp parsers & pipeline ─────────────────────────────────────────────────────

def parse_kp_flatfile(path: Path) -> pd.DataFrame:
    """GFZ Kp flat file: Kp_ap_Ap_SN_F107_since_1932.txt
    Format: YYYY MM DD  Bartels NK  Kp1..Kp8  ap1..ap8  Ap  SN  F10.7obs  F10.7adj  D
    Kp values are stored as Kp*10 integers (e.g., Kp=1.3 → 13).
    """
    log.info(f"[Kp/GFZ] Parsing flat file: {path.name}")
    records = []
    offsets = [0, 3, 6, 9, 12, 15, 18, 21]
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 13:
                    continue
                try:
                    year  = int(parts[0])
                    month = int(parts[1])
                    day   = int(parts[2])
                    # Try column offsets: current format starts Kp at col 5
                    kp_raw = None
                    for start in [5, 7, 6]:
                        if len(parts) >= start + 8:
                            vals, ok = [], True
                            for p in parts[start:start + 8]:
                                try:
                                    v = float(p)
                                    if v < -1 or v > 100:
                                        ok = False; break
                                    vals.append(v)
                                except ValueError:
                                    ok = False; break
                            if ok and len(vals) == 8:
                                kp_raw = vals; break
                    if kp_raw is None:
                        continue

                    # Kp values are Kp*10 integers; divide by 10
                    # Detect: if any value > 9 it's definitely *10 scale
                    # If all <= 9 AND all are integers, still divide (quiet period)
                    max_v = max(v for v in kp_raw if not np.isnan(v))
                    all_int = all(float(v) == int(float(v)) for v in kp_raw)
                    kp_vals = [v / 10.0 for v in kp_raw] if (max_v > 9 or all_int) else kp_raw

                    base = pd.Timestamp(year, month, day)
                    for i, kp in enumerate(kp_vals):
                        records.append({
                            "datetime": base + pd.Timedelta(hours=offsets[i]),
                            "kp_3h": max(0.0, min(9.0, float(kp))) if not np.isnan(kp) else np.nan
                        })
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        log.warning(f"[Kp/GFZ] Parse error: {e}")
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = df["datetime"].dt.normalize()
    daily = (df.groupby("date", as_index=False)
               .agg(kp_daily_mean=("kp_3h", "mean"),
                    kp_daily_max =("kp_3h", "max")))
    return finalize(daily, ["kp_daily_mean", "kp_daily_max"])


def parse_kp_noaa_json(path: Path) -> pd.DataFrame:
    """NOAA SWPC planetary K-index JSON (list of lists, first row = headers)."""
    log.info(f"[Kp/NOAA] Parsing NOAA SWPC JSON: {path.name}")
    try:
        with open(path) as f:
            data = json.load(f)
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        headers = [str(h).lower() for h in data[0]]
        df = pd.DataFrame(data[1:], columns=headers)
        time_col = next((c for c in df.columns if "time" in c), None)
        kp_col   = next((c for c in df.columns if c == "kp"), None)
        if not time_col or not kp_col:
            return pd.DataFrame()
        df["datetime"] = pd.to_datetime(df[time_col], errors="coerce")
        df["kp_3h"]    = pd.to_numeric(df[kp_col], errors="coerce").clip(0, 9)
        df["date"]     = df["datetime"].dt.normalize()
        df = df.dropna(subset=["date", "kp_3h"])
        daily = (df.groupby("date", as_index=False)
                   .agg(kp_daily_mean=("kp_3h", "mean"),
                        kp_daily_max =("kp_3h", "max")))
        return finalize(daily, ["kp_daily_mean", "kp_daily_max"])
    except Exception as e:
        log.warning(f"[Kp/NOAA] {e}")
        return pd.DataFrame()


def parse_kp_legacy_csv(path: Path) -> pd.DataFrame:
    """kp_merged_clean.csv produced by old download_data.py (3-hourly Kp series)."""
    log.info(f"[Kp/legacy] Parsing legacy CSV: {path.name}")
    try:
        df = pd.read_csv(path, parse_dates=["time_tag"])
        df = df.rename(columns={"time_tag": "datetime"})
        if "Kp" not in df.columns:
            return pd.DataFrame()
        df["kp_3h"] = pd.to_numeric(df["Kp"], errors="coerce").clip(0, 9)
        df["date"]  = pd.to_datetime(df["datetime"]).dt.normalize()
        df = df.dropna(subset=["date", "kp_3h"])
        daily = (df.groupby("date", as_index=False)
                   .agg(kp_daily_mean=("kp_3h", "mean"),
                        kp_daily_max =("kp_3h", "max")))
        return finalize(daily, ["kp_daily_mean", "kp_daily_max"])
    except Exception as e:
        log.warning(f"[Kp/legacy] {e}")
        return pd.DataFrame()


def parse_dgd(path: Path) -> pd.DataFrame:
    """NCEI Daily Geomagnetic Data — Kp cross-check / fallback."""
    log.info(f"[DGD] Parsing: {path.name}")
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                     on_bad_lines="skip", engine="python")
    rename_map = {0: "year", 1: "month", 2: "day"}
    for idx, name in {3: "a_fredericksburg", 12: "a_college", 21: "ap_planetary"}.items():
        if idx in df.columns:
            rename_map[idx] = name
    df = df.rename(columns=rename_map)
    kp_cols = []
    for i in range(8):
        if (22 + i) in df.columns:
            df[f"kp_dgd_{i+1}"] = pd.to_numeric(df[22 + i], errors="coerce")
            kp_cols.append(f"kp_dgd_{i+1}")
    num = [c for c in ["a_fredericksburg","a_college","ap_planetary"] if c in df.columns]
    df = to_num(df, num, [-1, -999])
    df["kp_dgd_mean"] = df[kp_cols].mean(axis=1) if kp_cols else np.nan
    df["kp_dgd_max"]  = df[kp_cols].max(axis=1)  if kp_cols else np.nan
    return finalize(df, num + ["kp_dgd_mean", "kp_dgd_max"] + kp_cols)


def build_kp(raw_dir: Path) -> pd.DataFrame:
    """Build daily Kp from all available sources — full pipeline.

    Priority  Source                              Coverage
    ────────  ──────────────────────────────────  ────────────────
    1         GFZ flat file (kp_historic_gfz.txt) 1932-present
    2         NOAA SWPC JSON (noaa_kp_index.json)  last 7 days
    3         DGD cross-check (always supplement)  1986-present
    4         Legacy kp_merged_clean.csv           whatever exists

    All sources are merged; newer / more recent data wins on overlap.
    Result saved to data/clean/kp_daily_clean.parquet.
    """
    sources = []

    # 1. GFZ flat file — primary historical source
    for fname in ["kp_historic_gfz.txt"]:
        p = raw_dir / fname
        if p.exists():
            df = parse_kp_flatfile(p)
            if not df.empty:
                sources.append(("GFZ-flat", df))
                log.info(f"[Kp] GFZ flat: {len(df):,} days "
                         f"({df.index.min().date()} – {df.index.max().date()})")
            break

    # 2. NOAA SWPC recent JSON — fills the very latest days
    for fname in ["noaa_kp_index.json"]:
        p = raw_dir / fname
        if p.exists():
            df = parse_kp_noaa_json(p)
            if not df.empty:
                sources.append(("NOAA-recent", df))
                log.info(f"[Kp] NOAA recent: {len(df):,} days")

    # 3. DGD — always use as supplement / gap-filler
    dgd_path = raw_dir / "dgd_1996_present.txt"
    if dgd_path.exists():
        dgd = parse_dgd(dgd_path)
        if not dgd.empty and "kp_dgd_mean" in dgd.columns:
            df = dgd[["kp_dgd_mean","kp_dgd_max"]].rename(
                columns={"kp_dgd_mean":"kp_daily_mean","kp_dgd_max":"kp_daily_max"})
            sources.append(("DGD", df))
            log.info(f"[Kp] DGD supplement: {len(df):,} days")

    # 4. Legacy kp_merged_clean.csv from old download_data.py
    for legacy_dir in [Path("data"), raw_dir]:
        p = legacy_dir / "kp_merged_clean.csv"
        if p.exists():
            df = parse_kp_legacy_csv(p)
            if not df.empty:
                sources.append(("legacy-csv", df))
                log.info(f"[Kp] legacy CSV: {len(df):,} days")
            break

    if not sources:
        log.warning("[Kp] No Kp source found — kp_daily_clean.parquet will be empty")
        return pd.DataFrame()

    # Merge: first source is base; each subsequent fills gaps
    result = sources[0][1]
    for name, extra in sources[1:]:
        merged = _merge_into(result, extra)
        # Only update NaN slots — don't let lower-priority sources overwrite good data
        for col in ["kp_daily_mean", "kp_daily_max"]:
            if col in result.columns and col in extra.columns:
                merged[col] = result[col].combine_first(extra[col].reindex(result.index.union(extra.index)))
        result = merged.sort_index()

    log.info(f"[Kp] Final dataset: {len(result):,} days "
             f"({result.index.min().date()} – {result.index.max().date()})")
    return result


# ── DSD parser (F10.7 + flares) ───────────────────────────────────────────────

def parse_dsd(path: Path) -> pd.DataFrame:
    log.info(f"[DSD] Parsing: {path.name}")
    cols = ["year","month","day","f107_dsd","ssn_sesc","sunspot_area","new_regions",
            "solar_mean_field","xray_bkgd","flare_C","flare_M","flare_X","flare_S",
            "opt_1","opt_2","opt_3"]
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                     on_bad_lines="skip", names=cols)
    num = [c for c in cols if c not in ("year","month","day","xray_bkgd")]
    df = to_num(df, num, [-999, -999.0])
    if "xray_bkgd" in df.columns:
        df["xray_bkgd"] = df["xray_bkgd"].replace("*", np.nan)
    flares = [c for c in ["flare_C","flare_M","flare_X"] if c in df.columns]
    opts   = [c for c in ["opt_1","opt_2","opt_3"]       if c in df.columns]
    df["flare_xray_total"]    = df[flares].sum(axis=1, min_count=1) if flares else np.nan
    df["flare_optical_total"] = df[opts].sum(axis=1, min_count=1)   if opts   else np.nan
    return finalize(df, ["f107_dsd","ssn_sesc","sunspot_area","new_regions","solar_mean_field",
                          "xray_bkgd","flare_C","flare_M","flare_X","flare_S",
                          "flare_xray_total","flare_optical_total"])


def parse_xray_noaa_json(path: Path) -> pd.DataFrame:
    """NOAA SWPC X-ray 7-day JSON — supplement for recent flare data."""
    log.info(f"[Xray/NOAA] Parsing: {path.name}")
    try:
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if df.empty or "time_tag" not in df.columns:
            return pd.DataFrame()
        df["datetime"] = pd.to_datetime(df["time_tag"], utc=True).dt.tz_localize(None)
        df["date"]     = df["datetime"].dt.normalize()
        if "flux" in df.columns:
            df["xray_flux"] = pd.to_numeric(df["flux"], errors="coerce")
            daily = (df[df.get("energy","") == "0.1-0.8nm"]
                       .groupby("date", as_index=False)
                       .agg(xray_flux_max=("xray_flux","max")))
            return finalize(daily, ["xray_flux_max"])
        return pd.DataFrame()
    except Exception as e:
        log.warning(f"[Xray/NOAA] {e}")
        return pd.DataFrame()


# ── Dst parsers ───────────────────────────────────────────────────────────────

def parse_dst_kyoto(path: Path) -> pd.DataFrame:
    log.info(f"[Dst/Kyoto] Parsing: {path.name}")
    try:
        colspecs = [(3,5),(5,7),(8,10)] + [(20+i*4,24+i*4) for i in range(24)] + [(116,120)]
        df = pd.read_fwf(path, colspecs=colspecs, header=None)
        df = df.rename(columns={0:"yy",1:"month",2:"day",27:"dst_wdc_mean"})
        for c in ["yy","month","day"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["yy","month","day"])
        df["year"]  = np.where(df["yy"]<57, 2000+df["yy"], 1900+df["yy"]).astype(int)
        df["month"] = df["month"].astype(int)
        df["day"]   = df["day"].astype(int)
        h = [c for c in range(3, 27) if c in df.columns]
        df[h] = df[h].apply(pd.to_numeric, errors="coerce").replace([9999,-9999], np.nan)
        df["dst_daily_mean"]  = df[h].mean(axis=1)
        df["dst_daily_min"]   = df[h].min(axis=1)
        df["dst_daily_max"]   = df[h].max(axis=1)
        df["dst_daily_std"]   = df[h].std(axis=1)
        df["dst_hours_valid"] = df[h].notna().sum(axis=1)
        keep = ["dst_daily_mean","dst_daily_min","dst_daily_max","dst_daily_std","dst_hours_valid"]
        if "dst_wdc_mean" in df.columns:
            keep.append("dst_wdc_mean")
        return finalize(df, keep)
    except Exception as e:
        log.warning(f"[Dst/Kyoto] {e}")
        return pd.DataFrame()


def parse_dst_omni2(raw_dir: Path) -> pd.DataFrame:
    log.info("[Dst/OMNI2] Parsing NASA SPDF OMNI2")
    dfs = []
    for fp in sorted(raw_dir.glob("omni2_*.dat")):
        try:
            tmp = pd.read_csv(fp, sep=r"\s+", header=None, on_bad_lines="skip")
            if tmp.shape[1] < 41: continue
            tmp = tmp[[0,1,40]].copy()
            tmp.columns = ["year","doy","dst"]
            tmp["dst"]  = pd.to_numeric(tmp["dst"], errors="coerce").replace([99999,-99999], np.nan)
            tmp["date"] = pd.to_datetime(
                tmp["year"].astype(str)+tmp["doy"].astype(str).str.zfill(3),
                format="%Y%j", errors="coerce")
            dfs.append(tmp.dropna(subset=["date","dst"]))
        except Exception as e:
            log.warning(f"[OMNI2] {fp.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    combined = (pd.concat(dfs, ignore_index=True)
                  .groupby("date", as_index=False)
                  .agg(dst_daily_mean=("dst","mean"), dst_daily_min=("dst","min"),
                       dst_daily_max=("dst","max"),  dst_daily_std=("dst","std"),
                       dst_hours_valid=("dst","count")))
    return finalize(combined, ["dst_daily_mean","dst_daily_min","dst_daily_max",
                                "dst_daily_std","dst_hours_valid"])


def parse_dst_noaa_json(path: Path) -> pd.DataFrame:
    """NOAA SWPC geospace DST 7-day JSON — recent supplement."""
    log.info(f"[Dst/NOAA] Parsing: {path.name}")
    try:
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if df.empty or "time_tag" not in df.columns or "dst" not in df.columns:
            return pd.DataFrame()
        df["datetime"]    = pd.to_datetime(df["time_tag"], utc=True).dt.tz_localize(None)
        df["date"]        = df["datetime"].dt.normalize()
        df["dst_hourly"]  = pd.to_numeric(df["dst"], errors="coerce")
        df = df.dropna(subset=["date","dst_hourly"])
        daily = (df.groupby("date", as_index=False)
                   .agg(dst_daily_mean=("dst_hourly","mean"),
                        dst_daily_min =("dst_hourly","min"),
                        dst_daily_max =("dst_hourly","max")))
        return finalize(daily, ["dst_daily_mean","dst_daily_min","dst_daily_max"])
    except Exception as e:
        log.warning(f"[Dst/NOAA] {e}")
        return pd.DataFrame()


def build_dst(raw_dir: Path) -> pd.DataFrame:
    """Merge all Dst sources: OMNI2 (hist) + Kyoto (2005+) + NOAA recent (7-day)."""
    parts = []
    omni  = parse_dst_omni2(raw_dir)
    if not omni.empty:  parts.append(omni)

    kyoto = _load_if(raw_dir / "dst_kyoto_hourly.txt", parse_dst_kyoto)
    if not kyoto.empty: parts.append(kyoto)

    noaa = _load_if(raw_dir / "noaa_dst_7day.json", parse_dst_noaa_json)
    if not noaa.empty:  parts.append(noaa)

    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    for col in combined.select_dtypes(include="object").columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined


# ── F10.7 & flares ────────────────────────────────────────────────────────────

def parse_f107_json(path: Path) -> pd.DataFrame:
    log.info(f"[F10.7] Parsing: {path.name}")
    try:
        df = pd.read_json(path)
        if df.empty or "reporting_schedule" not in df.columns:
            return pd.DataFrame()
        df = df[df["reporting_schedule"]=="Noon"].copy()
        df["date"] = pd.to_datetime(df["time_tag"], errors="coerce").dt.normalize()
        df = df.rename(columns={"flux":"f107_swpc","ninety_day_mean":"f107_90d_mean"})
        return finalize(df, [c for c in ["f107_swpc","f107_90d_mean"] if c in df.columns])
    except Exception as e:
        log.warning(f"[F10.7] {e}")
        return pd.DataFrame()


def build_f107(dsd: pd.DataFrame, swpc: pd.DataFrame) -> pd.DataFrame:
    """DSD F10.7 (historical) + SWPC JSON (recent gap-fill)."""
    idx = pd.date_range(START_DATE, pd.Timestamp.now().date(), freq="D", name="date")
    f = pd.DataFrame(index=idx)
    if not dsd.empty and "f107_dsd" in dsd.columns:
        f["f107"] = dsd["f107_dsd"].reindex(idx)
    else:
        f["f107"] = np.nan
    if not swpc.empty and "f107_swpc" in swpc.columns:
        f["f107"] = f["f107"].combine_first(swpc["f107_swpc"].reindex(idx))
    return f[f["f107"].notna()]


def parse_flare_events(path: Path) -> pd.DataFrame:
    """Legacy NGDC XRS flare event file (1996-2016)."""
    log.info(f"[Flares] Parsing: {path.name}")
    try:
        raw = pd.read_csv(path, sep="\n", header=None, dtype=str)[0]
        ext = raw.str.extract(r'^.{5}(\d{2})(\d{2})(\d{2}).*([BCMX])\s*(\d+\.?\d*)')
        ext = ext.dropna()
        ext.columns = ["yy","mm","dd","cls","mag"]
        ext["year"]  = ext["yy"].astype(int).apply(lambda x: 2000+x if x<50 else 1900+x)
        ext["month"] = ext["mm"].astype(int)
        ext["day"]   = ext["dd"].astype(int)
        ext["date"]  = pd.to_datetime(ext[["year","month","day"]], errors="coerce")
        ext = ext.dropna(subset=["date"])
        daily = (ext.groupby(["date","cls"]).size()
                    .unstack(fill_value=0).add_prefix("flare_event_"))
        daily["flare_events_total"] = daily.sum(axis=1)
        daily.index.name = "date"
        return daily[daily.index >= START_DATE]
    except Exception as e:
        log.warning(f"[Flares] {e}")
        return pd.DataFrame()


def build_flares(dsd: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    idx = pd.date_range(START_DATE, pd.Timestamp.now().date(), freq="D", name="date")
    f = pd.DataFrame(index=idx)
    for col in [c for c in ["flare_C","flare_M","flare_X","flare_S",
                              "flare_xray_total","flare_optical_total"]
                if not dsd.empty and c in dsd.columns]:
        f[col] = dsd[col].reindex(idx)
    if not events.empty:
        for col in events.columns:
            f[col] = events[col].reindex(idx).fillna(0)
    return f.dropna(how="all")


# ── Post-processing ───────────────────────────────────────────────────────────

def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    linear = ["ssn","ssn_std","f107","sunspot_area","dst_daily_mean","dst_daily_min",
              "dst_daily_max","dst_daily_std","ap_planetary","a_fredericksburg","a_college",
              "kp_daily_mean","kp_daily_max","solar_mean_field"] + [f"kp_{i}" for i in range(1,9)]
    ffill  = ["flare_C","flare_M","flare_X","flare_S","flare_xray_total","flare_optical_total",
              "new_regions","flare_events_total"] + \
             [c for c in df.columns if c.startswith("flare_event_")]
    for col in [c for c in linear if c in df.columns]:
        df[col] = df[col].interpolate(limit=7, limit_area="inside")
    for col in [c for c in ffill if c in df.columns]:
        df[col] = df[col].ffill(limit=3).fillna(0)
    if "dst_hours_valid" in df.columns:
        df["dst_hours_valid"] = df["dst_hours_valid"].fillna(0)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df["year"]        = df.index.year
    df["month"]       = df.index.month
    df["day_of_year"] = df.index.dayofyear
    if "dst_daily_min" in df.columns:
        df["storm_category"] = pd.cut(
            df["dst_daily_min"], bins=[-np.inf,-200,-100,-50,np.inf],
            labels=["severe","intense","moderate","quiet"], ordered=False)
    if "f107" in df.columns:
        df["solar_activity_level"] = pd.cut(
            df["f107"], bins=[-np.inf,100,150,200,np.inf],
            labels=["low","moderate","high","very_high"], ordered=False)
    if "kp_daily_max" in df.columns:
        df["kp_storm_level"] = pd.cut(
            df["kp_daily_max"], bins=[-np.inf,4.99,5.99,6.99,7.99,8.99,np.inf],
            labels=["quiet","G1_minor","G2_moderate","G3_strong","G4_severe","G5_extreme"],
            ordered=False)
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(update_mode: bool = False):
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"clean.py — {'UPDATE' if update_mode else 'FULL REBUILD'}")
    log.info("=" * 60)

    # Parse all raw sources
    ssn    = build_ssn(RAW_DIR)
    kp     = build_kp(RAW_DIR)
    dsd    = _load_if(RAW_DIR / "dsd_1996_present.txt",        parse_dsd)
    f107j  = _load_if(RAW_DIR / "f107_recent.json",            parse_f107_json)
    fla    = _load_if(RAW_DIR / "flares_events_1996_2016.txt", parse_flare_events)
    dst    = build_dst(RAW_DIR)
    f107   = build_f107(dsd, f107j)
    flares = build_flares(dsd, fla)

    if update_mode:
        log.info("Loading existing parquets for incremental merge…")
        # Load existing and merge new data in (new rows override old)
        old_ssn = _load_parquet("sunspots_daily_clean")
        if not old_ssn.empty and "sn" in old_ssn.columns:
            old_ssn = old_ssn.rename(columns={"sn":"ssn","sn_err":"ssn_std"})
        ssn    = _merge_into(old_ssn,                   ssn)
        kp     = _merge_into(_load_parquet("kp_daily_clean"),     kp)
        dst    = _merge_into(_load_parquet("dst_daily_clean"),     dst)
        f107   = _merge_into(_load_parquet("f107_daily_clean"),    f107)
        flares = _merge_into(_load_parquet("flares_daily_clean"),  flares)

    # Save themed parquets
    log.info("Saving individual parquets…")
    if not ssn.empty:
        _save(ssn.rename(columns={"ssn":"sn","ssn_std":"sn_err"}), "sunspots_daily_clean")
    _save(kp,     "kp_daily_clean")
    _save(dst,    "dst_daily_clean")
    _save(f107,   "f107_daily_clean")
    _save(flares, "flares_daily_clean")

    # Build master merged dataset
    log.info("Building master merged dataset…")
    if update_mode:
        merged = _load_parquet("solar_weather_daily")
        for c in ["year","month","day_of_year","storm_category",
                  "solar_activity_level","kp_storm_level"]:
            if c in merged.columns:
                merged = merged.drop(columns=[c])
    else:
        merged = pd.DataFrame(
            index=pd.date_range(START_DATE, pd.Timestamp.now().date(), freq="D"))
        merged.index.name = "date"

    for src_df, src_cols in [
        (ssn,    ["ssn","ssn_std"]),
        (dsd,    ["ssn_sesc","sunspot_area","new_regions","solar_mean_field","xray_bkgd"]),
        (kp,     None),
        (dst,    None),
        (f107,   None),
        (flares, None),
    ]:
        if src_df is not None and not src_df.empty:
            cols = [c for c in (src_cols or src_df.columns) if c in src_df.columns]
            new_data = src_df[cols]
            merged = _merge_into(merged, new_data) if update_mode else merged.join(new_data, how="left")

    merged = interpolate_missing(merged)
    merged = add_derived(merged)
    _save(merged, "solar_weather_daily")

    log.info("=" * 60)
    log.info("Pipeline complete")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline(update_mode="--update" in sys.argv)
