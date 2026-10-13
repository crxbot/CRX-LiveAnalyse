#!/usr/bin/env python3
"""HymecNG + RV-Hybrid im NATIVEN DWD-Raster (1100 x 1200, stereographisch).

Kein Warp nach Web Mercator. Ausgabe: rohe uint8-Bytes, keine Dateiendung.
Layout: 1200 Zeilen x 1100 Spalten, row-major, Zeile 0 = Norden (wie im HDF5).
Dateigroesse immer exakt xsize * ysize = 1.320.000 Bytes.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import requests
from pyproj import Transformer
from scipy import ndimage

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
SRC_DIR = Path("data/hymecng")
RV_SRC_DIR = Path("data/rv")
OUT_DIR = Path("output/hymecng")

FILENAME_RE = re.compile(r"composite_HymecNG_(\d{8})_(\d{4})_(\d{3})-hd5")

# Code-Bedeutung in der Ausgabedatei:
#   255 = kein Daten / ausserhalb      1 = Regen ohne RV-Wert (unsichtbar)
#   30 Niesel, 31 leicht, 32 maessig, 33 stark
#   4/5 gefrierender Regen             61/62 Schneeregen leicht / maessig+
#   71/72/73 Schnee leicht/maessig/stark
#   8 Graupel, 9/10 Hagel
#   200 Blitz, 201 Blitz in Hagelzone
HAIL_CLASSES = {9, 10}

RAIN_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.06, 0.1, 30),
    (0.1, 1.25, 31),
    (1.25, 10.0, 32),
    (10.0, float("inf"), 33),
]
SNOW_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.06, 1.0, 71),
    (1.0, 4.0, 72),
    (4.0, float("inf"), 73),
]
SLEET_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.06, 1.0, 61),
    (1.0, float("inf"), 62),
]

SNOW_TYPE_CODES = {7, 71, 72, 73}
SLEET_TYPE_CODES = {6, 61, 62}
FREEZING_RAIN_TYPE_CODES = {4, 5}
HAIL_TYPE_CODES = {9, 10}

MIN_PRECIP_RATE_MMH = 0.06

NODATA_CLASS = -1
NODATA_BYTE = 255
THUNDER_CODE = 200
STRONG_THUNDER_CODE = 201

LIGHTNING_BASE_URL = "https://radar.wetterstation-neustadt.de/blitze/archive/"
LIGHTNING_BACKUP_URL = "https://nowsky.vercel.app/api/lightning"
LIGHTNING_WINDOW_MINUTES = 5
# Native Pixel sind 1 km. Frueher 8 px im Web-Mercator-Bild (~0,65 km/px bei 50 N)
# -> ca. 5 km Radius -> hier 5 Pixel.
LIGHTNING_MARKER_RADIUS_PX = 5
LIGHTNING_PRECIP_MMH_THRESHOLD = 0.1

BERLIN = ZoneInfo("Europe/Berlin")

RV_DEFAULT_GAIN = 0.0009999999317806213
RV_DEFAULT_OFFSET = -0.0009999999317806213
RV_DEFAULT_NODATA = 4294967295.0
RV_DEFAULT_UNDETECT = 0.0
RV_STEP_MINUTES = 5.0   # ACRR pro 5 min -> mm/h = x * 60 / 5

PRECIP_SOURCE_CODES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
REFINED_PRECIP_CODES = [30, 31, 32, 33, 61, 62, 71, 72, 73]
ALL_PRECIP_CODES = PRECIP_SOURCE_CODES + REFINED_PRECIP_CODES


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def parse_timestamp(filename: str) -> datetime:
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Dateiname passt nicht zum Schema: {filename}")
    date_str, time_str, _ = m.groups()
    naive = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    return naive.replace(tzinfo=timezone.utc)


def rv_path_for(ts: datetime) -> Path:
    return RV_SRC_DIR / f"composite_rv_{ts:%Y%m%d}_{ts:%H%M}_000-hd5"


_DATA_PATH_RE = re.compile(r"(^|/)dataset\d+/data\d+/data$")


def _find_2d_dataset_by_quantity(h5file: h5py.File, keywords: tuple[str, ...], error_msg: str) -> h5py.Dataset:
    candidates: list[h5py.Dataset] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and _DATA_PATH_RE.search(name):
            candidates.append(obj)

    h5file.visititems(visitor)

    if not candidates:
        def loose_visitor(name, obj):
            if (
                isinstance(obj, h5py.Dataset)
                and name.endswith("/data")
                and obj.ndim == 2
                and "quality" not in name
            ):
                candidates.append(obj)

        h5file.visititems(loose_visitor)

    if not candidates:
        raise RuntimeError(error_msg)

    matches: list[h5py.Dataset] = []
    for ds in candidates:
        what = ds.parent.get("what")
        if what is not None and "quantity" in what.attrs:
            quantity = what.attrs["quantity"]
            if isinstance(quantity, bytes):
                quantity = quantity.decode(errors="ignore")
            if any(k in str(quantity).upper() for k in keywords):
                matches.append(ds)

    if matches:
        if len(matches) > 1:
            print(f"Warnung: {len(matches)} Datasets passen auf {keywords} — nehme {matches[0].name}.", file=sys.stderr)
        return matches[0]

    candidates.sort(key=lambda ds: ds.name)
    chosen = candidates[0]
    print(
        f"Warnung: keine Quantity passte auf {keywords} — nehme Fallback-Dataset {chosen.name}. "
        f"Bitte pruefen!",
        file=sys.stderr,
    )
    return chosen


def find_classification_dataset(h5file: h5py.File) -> h5py.Dataset:
    return _find_2d_dataset_by_quantity(
        h5file,
        keywords=("CLASS", "PRECIP", "HCLASS", "TYPE"),
        error_msg="Kein 2D-Datensatz in der HymecNG-Datei gefunden.",
    )


def find_rate_dataset(h5file: h5py.File) -> h5py.Dataset:
    return _find_2d_dataset_by_quantity(
        h5file,
        keywords=("RATE", "ACRR"),
        error_msg="Kein 2D-Datensatz in der RV-Datei gefunden.",
    )


def _attr_float(attrs, key: str, default: float | None = None) -> float | None:
    v = attrs.get(key)
    if v is None:
        return default
    return float(np.asarray(v).ravel()[0])


def _attr_str(attrs, key: str, default: str = "") -> str:
    v = attrs.get(key)
    if v is None:
        return default
    if isinstance(v, bytes):
        return v.decode(errors="ignore")
    return str(v)


def load_classification_array(dataset: h5py.Dataset, nodata_class: int) -> np.ndarray:
    """Liest die HymecNG-Klassifikation (nativ) und maskiert nodata/undetect."""
    raw = dataset[()]
    what = dataset.parent.get("what")
    attrs = what.attrs if what is not None else {}

    nodata = _attr_float(attrs, "nodata")
    undetect = _attr_float(attrs, "undetect")

    class_arr = raw.astype(np.int32)
    mask_invalid = np.zeros(class_arr.shape, dtype=bool)
    if nodata is not None:
        mask_invalid |= raw == nodata
    if undetect is not None:
        mask_invalid |= raw == undetect
    class_arr[mask_invalid] = nodata_class
    return class_arr


def find_where_group(h5file: h5py.File) -> h5py.Group | None:
    required = ("projdef", "xsize", "ysize", "xscale", "yscale", "LL_lon", "LL_lat")

    def complete(grp) -> bool:
        return all(k in grp.attrs for k in required)

    root_where = h5file.get("where")
    if root_where is not None and complete(root_where):
        return root_where

    found: list[h5py.Group] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.split("/")[-1] == "where" and complete(obj):
            found.append(obj)

    h5file.visititems(visitor)
    return found[0] if found else None


def extract_grid_info(where: h5py.Group) -> dict:
    def as_str(key: str) -> str:
        v = where.attrs[key]
        return v.decode() if isinstance(v, bytes) else str(v)

    def as_float(key: str) -> float:
        return float(where.attrs[key])

    return {
        "projdef": as_str("projdef"),
        "xsize": int(as_float("xsize")),
        "ysize": int(as_float("ysize")),
        "xscale": as_float("xscale"),
        "yscale": as_float("yscale"),
        "ll_lon": as_float("LL_lon"),
        "ll_lat": as_float("LL_lat"),
    }


def grids_identical(a: dict, b: dict, tol: float = 1e-6) -> bool:
    if a["projdef"] != b["projdef"] or a["xsize"] != b["xsize"] or a["ysize"] != b["ysize"]:
        return False
    return all(abs(a[k] - b[k]) <= tol for k in ("xscale", "yscale", "ll_lon", "ll_lat"))


# --------------------------------------------------------------------------- #
# Raster-Geometrie (nativ)
# --------------------------------------------------------------------------- #
def lonlat_to_native_rowcol(lon, lat, grid: dict, to_proj: Transformer):
    """lon/lat (Skalar oder Array) -> (row, col) im nativen Raster.
    Zeile 0 = Norden. Rueckgabe als float-Pixelindizes (floor fuer Pixel)."""
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    x, y = to_proj.transform(lon, lat)
    col = (np.asarray(x) - ll_x) / grid["xscale"]
    row = grid["ysize"] - 1 - (np.asarray(y) - ll_y) / grid["yscale"]
    return row, col


def resample_nearest_to_grid(
    data: np.ndarray,
    src_grid: dict,
    dst_grid: dict,
    fill_value: float,
) -> np.ndarray:
    """Falls RV-Raster nicht mit HymecNG-Raster identisch ist: nearest neighbor
    auf das HymecNG-Raster umsampeln."""
    to_wgs84_dst = Transformer.from_crs(dst_grid["projdef"], "EPSG:4326", always_xy=True)
    to_proj_dst = Transformer.from_crs("EPSG:4326", dst_grid["projdef"], always_xy=True)
    to_proj_src = Transformer.from_crs("EPSG:4326", src_grid["projdef"], always_xy=True)

    ll_x, ll_y = to_proj_dst.transform(dst_grid["ll_lon"], dst_grid["ll_lat"])
    cols = np.arange(dst_grid["xsize"])
    rows = np.arange(dst_grid["ysize"])
    xs = ll_x + (cols + 0.5) * dst_grid["xscale"]
    ys = ll_y + (dst_grid["ysize"] - 1 - rows + 0.5) * dst_grid["yscale"]
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = to_wgs84_dst.transform(xx.ravel(), yy.ravel())

    row_s, col_s = lonlat_to_native_rowcol(lon, lat, src_grid, to_proj_src)
    row_i = np.floor(row_s + 0.5).astype(np.int64).reshape(xx.shape)
    col_i = np.floor(col_s + 0.5).astype(np.int64).reshape(xx.shape)
    valid = (row_i >= 0) & (row_i < src_grid["ysize"]) & (col_i >= 0) & (col_i < src_grid["xsize"])

    out = np.full(xx.shape, fill_value, dtype=np.float64)
    out[valid] = data[row_i[valid], col_i[valid]]
    return out


# --------------------------------------------------------------------------- #
# RV laden (nativ)
# --------------------------------------------------------------------------- #
def load_rv_rate_native(rv_path: Path, class_grid: dict) -> np.ndarray:
    """RV-Rate in mm/h auf dem HymecNG-Raster. Bei identischem Raster 1:1,
    sonst nearest-neighbor-Resampling."""
    with h5py.File(rv_path, "r") as f:
        ds = find_rate_dataset(f)
        raw = ds[()]
        what = ds.parent.get("what")
        attrs = what.attrs if what is not None else {}
        quantity = _attr_str(attrs, "quantity").upper()
        gain = _attr_float(attrs, "gain", RV_DEFAULT_GAIN)
        offset = _attr_float(attrs, "offset", RV_DEFAULT_OFFSET)
        nodata = _attr_float(attrs, "nodata", RV_DEFAULT_NODATA)
        undetect = _attr_float(attrs, "undetect", RV_DEFAULT_UNDETECT)

        where = find_where_group(f)
        if where is None:
            raise RuntimeError("Keine 'where'-Projektionsinfo in der RV-Datei gefunden.")
        rv_grid = extract_grid_info(where)

    rate = raw.astype(np.float64) * gain + offset
    if undetect is not None:
        rate[raw == undetect] = 0.0
    if nodata is not None:
        rate[raw == nodata] = np.nan

    if "RATE" not in quantity:
        rate *= 60.0 / RV_STEP_MINUTES

    print(
        f"RV-Composite: {rv_path.name} (quantity={quantity or '?'}, "
        f"Raster {rv_grid['xsize']}x{rv_grid['ysize']})"
    )

    if grids_identical(class_grid, rv_grid):
        print("RV-Raster identisch mit HymecNG-Raster -> keine Umrechnung noetig.")
        return rate

    print(
        "Warnung: RV- und HymecNG-Raster unterscheiden sich -> nearest-neighbor-Resampling.",
        file=sys.stderr,
    )
    return resample_nearest_to_grid(rate, rv_grid, class_grid, fill_value=np.nan)


# --------------------------------------------------------------------------- #
# HYBRID-STRATEGIE: RV fuer Detektion + Intensitaet, HymecNG fuer Art
# --------------------------------------------------------------------------- #
def refine_with_hybrid_strategy(
    class_arr: np.ndarray,
    rate: np.ndarray | None,
) -> np.ndarray:
    refined = class_arr.copy()

    if rate is None:
        mask_rain = class_arr == 3
        mask_snow = class_arr == 7
        mask_sleet = class_arr == 6
        refined[mask_rain] = 31
        refined[mask_snow] = SNOW_MMH_THRESHOLDS[0][2]
        refined[mask_sleet] = SLEET_MMH_THRESHOLDS[0][2]
        if mask_rain.any() or mask_snow.any() or mask_sleet.any():
            print("Warnung: RV nicht verfuegbar, verwende niedrigste Intensitaetsstufe als Fallback.", file=sys.stderr)
        return refined

    has_rain = ~np.isnan(rate) & (rate >= MIN_PRECIP_RATE_MMH)

    # Regen (3)
    mask_refine_rain = (class_arr == 3) & has_rain
    for lower, upper, new_code in RAIN_MMH_THRESHOLDS:
        m = mask_refine_rain & (rate >= lower) & (rate < upper)
        refined[m] = new_code
    m = (class_arr == 3) & has_rain & (refined == 3)
    refined[m] = RAIN_MMH_THRESHOLDS[0][2]

    # Schnee (7)
    mask_refine_snow = (class_arr == 7) & has_rain
    for lower, upper, new_code in SNOW_MMH_THRESHOLDS:
        m = mask_refine_snow & (rate >= lower) & (rate < upper)
        refined[m] = new_code
    m = (class_arr == 7) & has_rain & (refined == 7)
    refined[m] = SNOW_MMH_THRESHOLDS[0][2]

    # Schneeregen (6)
    mask_refine_sleet = (class_arr == 6) & has_rain
    for lower, upper, new_code in SLEET_MMH_THRESHOLDS:
        m = mask_refine_sleet & (rate >= lower) & (rate < upper)
        refined[m] = new_code
    m = (class_arr == 6) & has_rain & (refined == 6)
    refined[m] = SLEET_MMH_THRESHOLDS[0][2]

    # Naechstgelegenen bekannten Niederschlagstyp bestimmen
    known_type = np.isin(class_arr, PRECIP_SOURCE_CODES)
    if known_type.any():
        _, (ir, ic) = ndimage.distance_transform_edt(~known_type, return_indices=True)
        nearest_type = class_arr[ir, ic]
    else:
        nearest_type = np.full(class_arr.shape, 3, dtype=class_arr.dtype)

    is_snow_type = np.isin(nearest_type, tuple(SNOW_TYPE_CODES))
    is_sleet_type = np.isin(nearest_type, tuple(SLEET_TYPE_CODES))
    is_preserved_type = np.isin(nearest_type, tuple(FREEZING_RAIN_TYPE_CODES | {8} | HAIL_TYPE_CODES))
    is_rain_type = ~is_snow_type & ~is_sleet_type & ~is_preserved_type

    # Luecken fuellen (RV hat Niederschlag, HymecNG nichts Konkretes)
    gap = has_rain & ~np.isin(class_arr, PRECIP_SOURCE_CODES)

    for lower, upper, new_code in RAIN_MMH_THRESHOLDS:
        m = gap & is_rain_type & (rate >= lower) & (rate < upper)
        refined[m] = new_code
    for lower, upper, new_code in SNOW_MMH_THRESHOLDS:
        m = gap & is_snow_type & (rate >= lower) & (rate < upper)
        refined[m] = new_code
    for lower, upper, new_code in SLEET_MMH_THRESHOLDS:
        m = gap & is_sleet_type & (rate >= lower) & (rate < upper)
        refined[m] = new_code

    m = gap & is_preserved_type
    refined[m] = nearest_type[m]

    return refined


# --------------------------------------------------------------------------- #
# Blitze (nativ)
# --------------------------------------------------------------------------- #
def _window_ms(ts: datetime, minutes: int) -> tuple[int, int]:
    end = int(ts.timestamp() * 1000)
    return end - minutes * 60_000, end


def _fetch_strikes_primary(ts: datetime, minutes: int) -> list[tuple[float, float]]:
    ts_local = ts.astimezone(BERLIN)
    url = f"{LIGHTNING_BASE_URL}{ts_local:%Y-%m-%d-%H%M}.json"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        print(f"Warnung: Primaerer Blitz-Feed liefert 404 ({url}) - nutze Backup-API.", file=sys.stderr)
        raise FileNotFoundError(url)
    resp.raise_for_status()

    start_ms, end_ms = _window_ms(ts, minutes)
    return [
        (s["lat"], s["lon"])
        for s in resp.json().get("strikes", [])
        if start_ms <= s.get("t", 0) <= end_ms
    ]


def _fetch_strikes_backup(ts: datetime, minutes: int) -> list[tuple[float, float]]:
    resp = requests.get(LIGHTNING_BACKUP_URL, timeout=30)
    resp.raise_for_status()

    start_ms, end_ms = _window_ms(ts, minutes)
    strikes = []
    for s in resp.json().get("strikes", []):
        t_ms = int(datetime.fromisoformat(s["time"].replace("Z", "+00:00")).timestamp() * 1000)
        if start_ms <= t_ms <= end_ms:
            strikes.append((s["lat"], s["lon"]))
    return strikes


def fetch_recent_strikes(ts: datetime, minutes: int = LIGHTNING_WINDOW_MINUTES) -> list[tuple[float, float]]:
    try:
        return _fetch_strikes_primary(ts, minutes)
    except FileNotFoundError:
        return _fetch_strikes_backup(ts, minutes)


def apply_lightning_overlay(
    out_codes: np.ndarray,        # uint8, nativ, wird in-place veraendert
    class_native: np.ndarray,     # verfeinerte Klassen (int32), nativ
    rate_native: np.ndarray | None,
    ts: datetime,
    grid: dict,
    to_proj: Transformer,
) -> int:
    strikes = fetch_recent_strikes(ts)
    print(f"{len(strikes)} Blitze in den letzten {LIGHTNING_WINDOW_MINUTES} Minuten geladen.")

    h, w = class_native.shape
    radius = LIGHTNING_MARKER_RADIUS_PX

    precip_mask = np.isin(class_native, ALL_PRECIP_CODES)
    if rate_native is not None:
        rate_ok = np.isnan(rate_native) | (rate_native >= LIGHTNING_PRECIP_MMH_THRESHOLD)
        precip_mask &= rate_ok

    offsets = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    circle = dr * dr + dc * dc <= radius * radius

    hits = 0
    for lat, lon in strikes:
        row_f, col_f = lonlat_to_native_rowcol(lon, lat, grid, to_proj)
        row = int(np.floor(row_f)) + 1
        col = int(np.floor(col_f))
        if not (0 <= col < w and 0 <= row < h):
            continue
        if not precip_mask[row, col]:
            continue

        r0, r1 = max(0, row - radius), min(h, row + radius + 1)
        c0, c1 = max(0, col - radius), min(w, col + radius + 1)
        circ = circle[r0 - (row - radius): r1 - (row - radius),
                      c0 - (col - radius): c1 - (col - radius)]

        area = precip_mask[r0:r1, c0:c1] & circ
        hail = np.isin(class_native[r0:r1, c0:c1], tuple(HAIL_CLASSES))
        out_codes[r0:r1, c0:c1][area & ~hail] = THUNDER_CODE
        out_codes[r0:r1, c0:c1][area & hail] = STRONG_THUNDER_CODE
        hits += 1
    return hits


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    candidates = sorted(p for p in SRC_DIR.glob("composite_HymecNG_*-hd5") if FILENAME_RE.match(p.name))
    if not candidates:
        sys.exit(f"Keine HymecNG-Datei in {SRC_DIR} gefunden.")
    src_path = candidates[-1]
    ts = parse_timestamp(src_path.name)

    with h5py.File(src_path, "r") as f:
        ds = find_classification_dataset(f)
        class_arr = load_classification_array(ds, NODATA_CLASS)
        class_arr[class_arr == 2] = 3   # Code 2 wie Code 3 behandeln
        where = find_where_group(f)
        if where is None:
            sys.exit("Keine 'where'-Projektionsinfo in der HD5-Datei gefunden.")
        grid = extract_grid_info(where)

    h, w = class_arr.shape
    if (w, h) != (grid["xsize"], grid["ysize"]):
        sys.exit(f"Rastergroesse passt nicht: Array {w}x{h}, where {grid['xsize']}x{grid['ysize']}")
    print(f"Natives Raster: {w} x {h} px, {grid['xscale']} m / px")

    to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)

    # RV laden (nativ)
    rate = None
    rv_path = rv_path_for(ts)
    try:
        if not rv_path.exists():
            raise FileNotFoundError(f"{rv_path} nicht gefunden")
        rate = load_rv_rate_native(rv_path, grid)
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        print(f"Warnung: RV-Composite nicht verfuegbar ({e}). Nutze HymecNG-Fallback.", file=sys.stderr)

    # Hybrid-Verfeinerung
    print("Wende Hybrid-Strategie an (RV + HymecNG)...")
    class_arr = refine_with_hybrid_strategy(class_arr, rate)

    # uint8
    out_codes = class_arr.copy()
    out_codes[out_codes < 0] = NODATA_BYTE
    out_codes = out_codes.astype(np.uint8)

    # Blitze
    try:
        hits = apply_lightning_overlay(out_codes, class_arr, rate, ts, grid, to_proj)
        print(f"{hits} Blitz-Treffer eingetragen.")
    except requests.RequestException as e:
        print(f"Warnung: Blitzdaten konnten nicht geladen werden ({e}). Ueberspringe Overlay.", file=sys.stderr)

    # Speichern: rohe Bytes, keine Dateiendung, Zeile 0 = Norden
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"liveanalyse_{ts.astimezone(BERLIN):%Y%m%d_%H%M}"
    np.ascontiguousarray(out_codes).tofile(out_path)
    print(f"Gespeichert: {out_path} ({w}x{h}, {out_codes.nbytes} Bytes)")

    # Metadaten fuer den Client (einmalig kopieren, aendern sich nicht)
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    meta = {
        "width": w,
        "height": h,
        "projdef": grid["projdef"],
        "xscale": grid["xscale"],
        "yscale": grid["yscale"],
        "ll_x": ll_x,
        "ll_y": ll_y,
        "rowOrder": "north-to-south",
    }
    print("Client-Metadaten:", json.dumps(meta))


if __name__ == "__main__":
    main()