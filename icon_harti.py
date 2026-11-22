#!/usr/bin/env python3
"""
Hărți meteo pentru România din modele globale/europene deschise (prin Open-Meteo),
rafinate pe relieful comun al tuturor hărților: grila și orografia ALADIN 2,3 km
(ICON-EU, DWD ~7 km și ECMWF IFS ~9 km).

Rafinarea NU este o rulare nouă a modelului. Câmpurile modelului sunt interpolate pe grila fină și
corectate statistic după relieful comun (orografia ALADIN 2,3 km):
  - temperatura: gradientul vertical al modelului (ICON: între 2 m și 850/700 hPa; ECMWF: -6 °C/km);
  - punctul de rouă și umezeala: umezeală absolută păstrată, rouă -1,7 °C/km;
  - precipitațiile: redistribuite spre versanții expuși vântului și spre altitudine, păstrând
    cantitatea modelului pe fiecare zonă de mărimea unei celule a modelului; ninsoarea după temperatura rafinată;
  - vântul: întărit pe creste, slăbit în văi;
  - norii joși (și satelitul simulat) și vizibilitatea: după umezeala rafinată.

Rulează din oră în oră (Task Scheduler); descarcă doar rulările de 00 și 12 UTC, doar zona României.
"""
import base64
import gzip
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

try:
    import requests
except ImportError:
    requests = None
import omfiles

MODEL = "icon"          # "icon" sau "ecmwf" (setat la instalare, fiecare folder are copia lui)

CFG = {
    "icon": {
        "title": "ICON-EU", "who": "DWD", "src": "dwd_icon_eu", "native": "7 km", "fine": "2,3 km",
        "grid": {"type": "regular", "lon0": -23.5, "lat0": 29.5, "d": 0.0625, "nx": 1377, "ny": 657},
        "out": (19.0, 42.5, 0.03, 0.021, 367, 334),
        "sigma_m": 5500.0, "ua": True, "ua_step": 3, "readahead": 4096, "workers": 16,
        "surf": ["temperature_2m", "relative_humidity_2m", "wind_u_component_10m", "wind_v_component_10m",
                 "wind_gusts_10m", "pressure_msl", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
                 "cloud_cover_high", "cape", "visibility", "freezing_level_height"],
        "log": "icon_harti.log", "task": "Harti ICON",
        "max_h": 180,
        # după +120 h ICON-EU se oprește: continuăm cu ICON global (aceeași rulare), pe aceeași grilă fină
        "ext": {"src": "dwd_icon", "from_h": 121, "title": "ICON global", "native": "13 km", "sigma_m": 9000.0,
                "grid": {"type": "global"}},
    },
    "ecmwf": {
        "title": "ECMWF IFS", "who": "ECMWF", "src": "ecmwf_ifs", "native": "9 km", "fine": "2,3 km",
        "grid": {"type": "o1280"},
        "out": (19.0, 42.5, 0.03, 0.021, 367, 334),
        "sigma_m": 7000.0, "ua": False, "ua_step": 4, "readahead": 32768, "workers": 16,
        "surf": ["temperature_2m", "dew_point_2m", "wind_u_component_10m", "wind_v_component_10m",
                 "wind_gusts_10m", "pressure_msl", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
                 "cloud_cover_high", "cape", "visibility", "wind_u_component_200m", "wind_v_component_200m"],
        "log": "ecmwf_harti.log", "task": "Harti ECMWF",
        "max_h": 240,
    },
}
C = CFG[MODEL]
S3 = "https://openmeteo.s3.amazonaws.com/data_spatial/%s/" % C["src"]
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FORMAT = "aladin1"                 # același format de date ca la ALADIN (pagina e comună)
WORKERS = C["workers"]

LON0, LAT0, DLON, DLAT, CW, CH = C["out"]
STEPS = list(range(1, 13)) + list(range(15, 49, 3)) + list(range(54, C["max_h"] + 1, 6))
EXT = C.get("ext")
S3_EXT = "https://openmeteo.s3.amazonaws.com/data_spatial/%s/" % EXT["src"] if EXT else None
# variabile care pot lipsi la modelul de continuare: valoare de rezervă
OPTIONAL = {"visibility": 24000.0, "cape": 0.0, "freezing_level_height": None}
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 200] if C["ua"] else []
UA_STEP = C["ua_step"]
MG_STEP, MG_TILE = 3, 16
ACC_SRC = ["precipitation", "snowfall_water_equivalent"]
UA_SRC = {"t": "temperature", "z": "geopotential_height", "u": "wind_u_component",
          "v": "wind_v_component", "rh": "relative_humidity"} if C["ua"] else {}

STEP = {"t2m": 0.1, "rh2m": 1, "u10": 0.25, "v10": 0.25, "gust": 0.5, "mslp": 0.1,
        "prectot": 0.02, "lssnow": 0.02, "tcc": 5, "lcc": 5, "mcc": 5, "hcc": 5,
        "cape": 10, "vis": 100}
if C["ua"]:
    STEP.update({"frz": 50, "thick": 0.5})
UA_STEP_BY = {"t": 0.2, "z": 5, "u": 0.5, "v": 0.5, "rh": 2}
MG_VARS = ["t2m", "rh2m", "u10", "v10", "gust", "mslp", "prectot", "lssnow", "tcc", "lcc", "mcc", "hcc"]
if C["ua"]:
    MG_VARS.append("frz")

log = logging.getLogger(MODEL)

# ---------------- grila de ieșire ----------------
OLON = LON0 + np.arange(CW) * DLON
OLAT = LAT0 + np.arange(CH) * DLAT
GLON, GLAT = np.meshgrid(OLON, OLAT)
DX_M = DLON * 111320.0 * np.cos(np.radians(GLAT))          # m pe pixel, pe x (variază cu latitudinea)
DY_M = DLAT * 110570.0
MARGIN = 0.35                                               # grade în plus în jurul zonei (pentru interpolare)


# ---------------- acces HTTP pe bucăți ----------------
_tls = threading.local()
STATS = {"req": 0, "bytes": 0}
_stats_lock = threading.Lock()


def http(url, rng=None, tries=6, want_total=False):
    """rng = (start, end_inclusiv) sau ("tail", n) pentru ultimii n octeți"""
    for k in range(tries):
        try:
            headers = {"User-Agent": "HartiMeteo/1.0"}
            total = None
            if rng and rng[0] == "tail":
                headers["Range"] = "bytes=-%d" % rng[1]
            elif rng:
                headers["Range"] = "bytes=%d-%d" % rng
            if requests is not None:
                s = getattr(_tls, "s", None)
                if s is None:
                    s = _tls.s = requests.Session()
                r = s.get(url, headers=headers, timeout=120)
                if r.status_code not in (200, 206):
                    raise IOError("HTTP %d pentru %s" % (r.status_code, url))
                data = r.content
                total = r.headers.get("Content-Range", "/").split("/")[-1]
            else:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                    total = (r.headers.get("Content-Range") or "/").split("/")[-1]
            with _stats_lock:
                STATS["req"] += 1; STATS["bytes"] += len(data)
            if want_total:
                return data, int(total)
            return data
        except Exception as e:
            if k == tries - 1:
                raise
            if k >= 1:
                log.warning("eroare %s (%s), reîncerc", url.rsplit("/", 1)[-1], e)
            _tls.s = None
            time.sleep(2 * (k + 1) + (hash(url) % 7) / 7)


class RangeFS:
    """sistem de fișiere minimal pentru omfiles: citește doar octeții ceruți, cu o rezervă
    (bucățile vecine sunt lipite în fișier) și păstrează ultimele câteva blocuri citite"""

    def __init__(self, size, tail=None, readahead=4096, keep=24):
        self._size = size
        self.blocks = []
        self.tail = tail
        self.readahead = readahead
        self.keep = keep

    def size(self, path):
        return self._size

    def cat_file(self, path, start=None, end=None, **kw):
        start = start or 0
        end = self._size if end is None else end
        for blk in ([self.tail] if self.tail else []) + self.blocks:
            if blk[0] <= start and end <= blk[0] + len(blk[1]):
                return blk[1][start - blk[0]:end - blk[0]]
        e2 = min(self._size, max(end, start + self.readahead))
        data = http(path, (start, e2 - 1))
        self.blocks.insert(0, (start, data))
        del self.blocks[self.keep:]
        return data[:end - start]


_tails = {}
_tails_lock = threading.Lock()
TAIL = 65536


def file_tail(url):
    """mărimea fișierului și ultimii 64 KB (acolo e cuprinsul: variabile și tabele de poziții)"""
    with _tails_lock:
        if url in _tails:
            return _tails[url]
    data, total = http(url, ("tail", TAIL), want_total=True)
    res = (total, (total - len(data), data))
    with _tails_lock:
        _tails[url] = res
        if len(_tails) > 300:
            _tails.pop(next(iter(_tails)))
    return res


# ---------------- citirea grilei modelului -> grila fină (biliniar) ----------------
class RegularGrid:
    def __init__(self, g):
        self.g = g
        d = g["d"]
        fi = (GLON - g["lon0"]) / d
        fj = (GLAT - g["lat0"]) / d
        self.I0 = int(np.floor(fi.min())) - 3; self.I1 = int(np.ceil(fi.max())) + 4
        self.J0 = int(np.floor(fj.min())) - 3; self.J1 = int(np.ceil(fj.max())) + 4
        ii = fi - self.I0; jj = fj - self.J0
        self.ia = np.floor(ii).astype(int); self.ja = np.floor(jj).astype(int)
        self.wi = (ii - self.ia).astype(np.float32); self.wj = (jj - self.ja).astype(np.float32)

    def read(self, v):
        shp = tuple(v.shape)
        if shp != (self.g["ny"], self.g["nx"]):
            raise RuntimeError("grila %s are forma %s, așteptam %s" % (C["src"], shp, (self.g["ny"], self.g["nx"])))
        s = np.nan_to_num(np.asarray(v[self.J0:self.J1, self.I0:self.I1], dtype=np.float32))
        a = s[self.ja, self.ia]; b = s[self.ja, self.ia + 1]; c = s[self.ja + 1, self.ia]; d = s[self.ja + 1, self.ia + 1]
        return ((a * (1 - self.wi) + b * self.wi) * (1 - self.wj) + (c * (1 - self.wi) + d * self.wi) * self.wj).astype(np.float32)


class O1280Grid:
    """grila gaussiană redusă octaedrică O1280 a ECMWF (rânduri de la nord la sud, longitudini de la 0°)"""
    N = 1280

    def __init__(self):
        cache = os.path.join(HERE, "o1280_lat.npy")
        if os.path.exists(cache):
            lat = np.load(cache)
        else:
            x, _ = np.polynomial.legendre.leggauss(2 * self.N)
            lat = np.degrees(np.arcsin(x))[::-1].copy()        # de la nord la sud
            np.save(cache, lat)
        i = np.arange(2 * self.N)
        ring = np.where(i < self.N, i + 1, 2 * self.N - i)
        nlon = 4 * ring + 16
        off = np.concatenate([[0], np.cumsum(nlon)[:-1]])
        self.total = int(nlon.sum())
        la_min, la_max = OLAT.min() - MARGIN, OLAT.max() + MARGIN
        lo_min, lo_max = OLON.min() - MARGIN, OLON.max() + MARGIN
        rows = np.where((lat >= la_min) & (lat <= la_max))[0]
        rows = np.arange(max(rows[0] - 1, 0), min(rows[-1] + 2, 2 * self.N))
        self.segs = []                       # (indice_start, indice_stop, lon_start, dlon) pe fiecare rând
        for r in rows:
            dl = 360.0 / nlon[r]
            a = int(np.floor(lo_min / dl)) - 1; b = int(np.ceil(lo_max / dl)) + 2
            self.segs.append((int(off[r] + a), int(off[r] + b), a * dl, dl))
        self.rlat = lat[rows]
        # pentru fiecare punct fin: rândul de la nord (k) și cel de la sud (k+1)
        k = np.searchsorted(-self.rlat, -GLAT, side="right") - 1
        self.k = np.clip(k, 0, len(rows) - 2)
        la_n = self.rlat[self.k]; la_s = self.rlat[self.k + 1]
        self.wlat = ((la_n - GLAT) / (la_n - la_s)).astype(np.float32)

    def _row_interp(self, vals, q):
        out = np.empty(GLAT.shape, dtype=np.float32)
        for kk in np.unique(q):
            m = q == kk
            a0, _b0, l0, dl = self.segs[kk]
            x = (GLON[m] - l0) / dl
            i = np.clip(np.floor(x).astype(int), 0, len(vals[kk]) - 2)
            w = x - i
            out[m] = vals[kk][i] * (1 - w) + vals[kk][i + 1] * w
        return out

    def read(self, v):
        shp = tuple(v.shape)
        if shp not in ((1, self.total), (self.total,)):
            raise RuntimeError("grila ECMWF are forma %s, așteptam (1, %d)" % (shp, self.total))
        vals = []
        for a, b, _l0, _dl in self.segs:
            seg = v[0, a:b] if len(shp) == 2 else v[a:b]
            vals.append(np.nan_to_num(np.asarray(seg, dtype=np.float32).ravel()))
        n = self._row_interp(vals, self.k)
        s = self._row_interp(vals, self.k + 1)
        return (n * (1 - self.wlat) + s * self.wlat).astype(np.float32)


GRIDS = {}
_GRID_LOCK = threading.Lock()


def global_grid(shape):
    """grilă regulată globală Open-Meteo (de la -90°/-180°), pasul dedus din dimensiuni"""
    ny, nx = shape
    d = 180.0 / (ny - 1)
    if abs(nx * d - 360.0) > 1.01 * d:
        raise RuntimeError("grila globală are forma neașteptată %s" % (shape,))
    log.info("grila %s: %d x %d, pas %.4f°", EXT["src"], nx, ny, d)
    return RegularGrid({"type": "regular", "lon0": -180.0, "lat0": -90.0, "d": d, "nx": nx, "ny": ny})


def grid(src="main", shape=None):
    with _GRID_LOCK:
        if src not in GRIDS:
            if src == "ext":
                GRIDS[src] = global_grid(shape)
            else:
                GRIDS[src] = RegularGrid(C["grid"]) if C["grid"]["type"] == "regular" else O1280Grid()
        return GRIDS[src]


def read_vars(url, names, src="main"):
    """citește variabilele cerute dintr-un fișier .om, doar zona României, direct pe grila fină"""
    size, tail = file_tail(url)
    fs = RangeFS(size, tail, readahead=EXT["readahead"] if src == "ext" and "readahead" in EXT else C["readahead"])
    rd = omfiles.OmFileReader.from_fsspec(fs, url)
    out = {}
    for nm in names:
        try:
            v = rd.get_child_by_name(nm)
        except Exception:
            v = None
        if v is None:
            if src == "ext" and nm in OPTIONAL:
                out[nm] = None
                continue
            raise RuntimeError("variabila %s lipsește din %s" % (nm, url.rsplit("/", 1)[-1]))
        out[nm] = grid(src, tuple(v.shape)).read(v)
    return out


RUN_HOURS = (0, 12)


def latest_run():
    """ultima rulare de 00 sau 12 UTC disponibilă"""
    j = json.loads(http(S3 + "latest.json"))
    ref = datetime.strptime(j["reference_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if ref.hour not in RUN_HOURS:
        ref = ref - timedelta(hours=ref.hour % 12)
        j = json.loads(http(S3 + ref.strftime("%Y/%m/%d/%H%MZ/") + "meta.json"))
    valid = [datetime.strptime(v, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc) for v in j["valid_times"]]
    return ref, valid, j.get("completed", True)


def om_url(ref, t, src="main"):
    return (S3_EXT if src == "ext" else S3) + ref.strftime("%Y/%m/%d/%H%MZ/") + t.strftime("%Y-%m-%dT%H%M") + ".om"


def ext_run(ref):
    """orele modelului de continuare (ICON global) pentru aceeași rulare; {} dacă nu e gata"""
    try:
        j = json.loads(http(S3_EXT + ref.strftime("%Y/%m/%d/%H%MZ/") + "meta.json"))
    except Exception as e:
        log.info("%s: rularea nu e încă disponibilă (%s)", EXT["title"], e)
        return {}, False
    valid = [datetime.strptime(v, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc) for v in j["valid_times"]]
    return {int(round((t - ref).total_seconds() / 3600)): t for t in valid}, j.get("completed", True)


# ---------------- relieful real (Mapterhorn), mediat pe celulele grilei fine ----------------
DEM_Z = 7


def _tile_xy(lon, lat, z):
    n = 2 ** z * 512
    x = (lon + 180.0) / 360.0 * n
    la = np.radians(lat)
    y = (1 - np.log(np.tan(la) + 1 / np.cos(la)) / np.pi) / 2 * n
    return x, y


def build_dem():
    from PIL import Image
    sub = 5
    oy, ox = (np.arange(sub) + 0.5) / sub - 0.5, (np.arange(sub) + 0.5) / sub - 0.5
    lons = (GLON[:, :, None, None] + ox[None, None, None, :] * DLON) + 0 * oy[None, None, :, None]
    lats = (GLAT[:, :, None, None] + oy[None, None, :, None] * DLAT) + 0 * ox[None, None, None, :]
    px, py = _tile_xy(lons, lats, DEM_Z)
    tx0, tx1 = int(px.min() // 512), int(px.max() // 512)
    ty0, ty1 = int(py.min() // 512), int(py.max() // 512)
    mos = np.zeros(((ty1 - ty0 + 1) * 512, (tx1 - tx0 + 1) * 512), dtype=np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            data = http("https://tiles.mapterhorn.com/%d/%d/%d.webp" % (DEM_Z, tx, ty))
            im = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32)
            el = im[:, :, 0] * 256 + im[:, :, 1] + im[:, :, 2] / 256 - 32768
            mos[(ty - ty0) * 512:(ty - ty0 + 1) * 512, (tx - tx0) * 512:(tx - tx0 + 1) * 512] = el
    ix = np.clip((px - tx0 * 512).astype(int), 0, mos.shape[1] - 1)
    iy = np.clip((py - ty0 * 512).astype(int), 0, mos.shape[0] - 1)
    z = np.clip(mos[iy, ix], 0, None).mean(axis=(2, 3))
    return z.astype(np.float32)


def load_dem():
    """relieful comun al tuturor hărților: orografia ALADIN 2,3 km (aceeași grilă)"""
    for q in (os.path.join(HERE, "relief_aladin.npy"), os.path.join(HERE, "..", "HartiAladin4", "orog_aladin.npy")):
        if os.path.exists(q):
            a = np.load(q)
            if a.shape == (CH, CW):
                return a.astype(np.float32)
    p = os.path.join(HERE, "relief_%s.npy" % MODEL)
    if os.path.exists(p):
        a = np.load(p)
        if a.shape == (CH, CW):
            return a
    a = build_dem()
    np.save(p, a)
    log.info("relief descărcat (Mapterhorn): %d x %d, max %.0f m", CW, CH, a.max())
    return a


# ---------------- rafinarea pe relief ----------------
def _blur1(a, sig, axis):
    if sig <= 0.3:
        return a
    r = int(math.ceil(3 * sig))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()
    pad = [(0, 0), (0, 0)]; pad[axis] = (r, r)
    ap = np.pad(a, pad, mode="reflect")
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis, ap)


def gblur(a, sigma_m):
    sx = sigma_m / float(np.mean(DX_M)); sy = sigma_m / DY_M
    return _blur1(_blur1(a.astype(np.float64), sx, 1), sy, 0).astype(np.float32)


class Terrain:
    def __init__(self, zf, sigma_m=None):
        sigma_m = sigma_m or C["sigma_m"]
        self.zf = zf
        self.zm = gblur(zf, sigma_m)                       # relieful „văzut” de model (netezit)
        self.dz = (zf - self.zm).astype(np.float32)             # cât e punctul mai sus/jos decât în model
        gy, gx = np.gradient(zf.astype(np.float64))
        self.dzdx = (gx / DX_M).astype(np.float32)
        self.dzdy = (gy / DY_M).astype(np.float32)
        self.sm = lambda a: gblur(a, sigma_m)


def es(T):
    return 6.112 * np.exp(17.62 * T / (243.12 + T))


def td_from_rh(T, rh):
    rh = np.clip(rh, 1, 100)
    g = np.log(rh / 100.0) + 17.62 * T / (243.12 + T)
    return 243.12 * g / (17.62 - g)


def refine_precip(tr, P, us, vs):
    """redistribuie cantitatea pe versanții expuși și la altitudine, păstrând media pe zona unei celule de model"""
    # viteza verticală forțată de relief, netezită pe ~3 km (picăturile nu cad exact pe pantă), normalizată
    s = gblur(us * tr.dzdx + vs * tr.dzdy, 3000.0) / 0.08
    w = (1 + 0.45 * np.clip(s, 0, 4)) * np.exp(-0.3 * np.clip(-s, 0, 4)) * (1 + 0.2 * np.clip(tr.dz / 500.0, -1, 2))
    w = np.clip(w, 0.25, 3.5).astype(np.float32)
    out = P * w / np.maximum(tr.sm(w), 1e-3)
    return np.where(P > 0, np.clip(out, 0, None), 0).astype(np.float32)


def refine(tr, c, gamma, us, vs):
    """c: câmpurile modelului interpolate pe grila fină (T în °C, Td în °C, vânt în m/s, nori în %)"""
    f = {}
    dz = tr.dz
    T = c["t"] + gamma * dz
    Td = np.minimum(c["td"] - 0.0017 * dz, T)
    rh_c = np.clip(100 * es(c["td"]) / es(c["t"]), 0, 100)
    rh = np.clip(100 * es(Td) / es(T), 0, 100)
    f["t2m"] = T.astype(np.float32); f["rh2m"] = rh.astype(np.float32)
    k = np.clip(1 + 0.35 * dz / 500.0, 0.55, 1.6)
    kg = np.clip(1 + 0.25 * dz / 500.0, 0.65, 1.5)
    f["u10"] = (c["u"] * k).astype(np.float32); f["v10"] = (c["v"] * k).astype(np.float32)
    f["gust"] = np.maximum(c["gust"] * kg, np.hypot(f["u10"], f["v10"])).astype(np.float32)
    d = rh - rh_c
    lcc = c["lcc"]
    up = lcc + (100 - lcc) * np.clip(d / 20.0, 0, 1) * np.clip((rh - 80) / 15.0, 0, 1)
    dn = lcc * np.clip(1 + d / 30.0, 0, 1)
    lf = np.clip(np.where(d > 0, up, dn), 0, 100)
    other = np.maximum(c["mcc"], c["hcc"]) / 100.0
    tcc = np.clip(c["tcc"] + (lf - lcc) * (1 - other), 0, 100)
    f["lcc"] = lf.astype(np.float32); f["mcc"] = c["mcc"]; f["hcc"] = c["hcc"]
    f["tcc"] = np.maximum(tcc, np.maximum(lf, np.maximum(c["mcc"], c["hcc"]))).astype(np.float32)
    fog = (rh >= 96) & (lf > lcc)
    f["vis"] = np.where(fog, c["vis"] * np.clip(1 - (rh - 95) / 5 * 0.9, 0.1, 1), c["vis"]).clip(0, None).astype(np.float32)
    return f


# ---------------- codare (comună cu ALARO/ALADIN) ----------------
def med_residual(x):
    a = np.zeros_like(x); b = np.zeros_like(x); c = np.zeros_like(x)
    a[:, 1:] = x[:, :-1]; b[1:, :] = x[:-1, :]; c[1:, 1:] = x[:-1, :-1]
    mx = np.maximum(a, b); mn = np.minimum(a, b)
    p = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    p[0, 1:] = x[0, :-1]; p[1:, 0] = x[:-1, 0]; p[0, 0] = 0
    return x - p


def pack_residual(r):
    z = (r << 1) ^ (r >> 63)
    if z.max() >= 65536:
        raise ValueError("valori în afara intervalului")
    z = z.astype(np.uint16).ravel()
    return np.concatenate([(z & 255).astype(np.uint8), (z >> 8).astype(np.uint8)]).tobytes()


def enc_field(v, st):
    v = np.nan_to_num(np.asarray(v, dtype=np.float64))
    k = np.round(v / st).astype(np.int64)
    k0 = int(k.min())
    return k0, st, base64.b64encode(gzip.compress(pack_residual(med_residual(k - k0)), 9)).decode()


def write_js(path, var, key, obj):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("window.%s=window.%s||{};window.%s[%s]=" % (var, var, var, json.dumps(key))
                 + json.dumps(obj, separators=(",", ":")) + ";\n")


def write_hour(tmp, h, f):
    out = {"h": h, "f": {}}
    for name in STEP:
        k0, st, d = enc_field(f[name], STEP[name])
        out["f"][name] = {"k0": k0, "st": st, "p1": f["p1"], "p2": h, "d": d}
    write_js(os.path.join(tmp, "h%02d.js" % h), "ALARO_DATA", h, out)
    if C["ua"]:
        ua = {"h": h, "f": {}}
        for short in UA_SRC:
            for L in LEVELS:
                v = f["%s%d" % (short, L)][::UA_STEP, ::UA_STEP]
                k0, st, d = enc_field(v, UA_STEP_BY[short])
                ua["f"]["%s%d" % (short, L)] = {"k0": k0, "st": st, "d": d}
        write_js(os.path.join(tmp, "h%02d_p.js" % h), "ALARO_UA", h, ua)


def write_orog(tmp, hgt):
    k0, st, d = enc_field(hgt, 1)
    rec = {"w": CW, "h": CH, "k0": k0, "st": st, "d": d}
    with open(os.path.join(tmp, "orog.js"), "w", encoding="utf-8") as fh:
        fh.write("window.ALARO_OROG=" + json.dumps(rec, separators=(",", ":")) + ";\n")


def write_meteograms(tmp, mg):
    os.makedirs(os.path.join(tmp, "mg"))
    ny, nx = mg[MG_VARS[0]].shape[1:]
    for ty in range(0, ny, MG_TILE):
        for tx in range(0, nx, MG_TILE):
            th, tw = min(MG_TILE, ny - ty), min(MG_TILE, nx - tx)
            obj = {"tw": tw, "th": th, "v": {}}
            for name in MG_VARS:
                a = mg[name][:, ty:ty + th, tx:tx + tw].astype(np.float64)
                st = STEP[name]
                k = np.round(a / st).astype(np.int64)
                k0 = int(k.min())
                r = np.stack([med_residual(k[n] - k0) for n in range(k.shape[0])])
                obj["v"][name] = {"k0": k0, "st": st, "d": base64.b64encode(gzip.compress(pack_residual(r), 9)).decode()}
            key = "%d_%d" % (tx // MG_TILE, ty // MG_TILE)
            write_js(os.path.join(tmp, "mg", "m_%s.js" % key), "ALARO_MG", key, obj)


# ---------------- istoric: ultimele rulări, fiecare în data/<rulare>/ ----------------
KEEP_RUNS = 4
_RUN_RE = re.compile(r"^\d{10}$")


def _read_meta_file(path):
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.read().split("=", 1)[1].rstrip().rstrip(";"))


def hist_runs():
    """rulările păstrate, cea mai nouă prima"""
    if not os.path.isdir(DATA):
        return []
    return sorted([d for d in os.listdir(DATA) if _RUN_RE.match(d) and os.path.exists(os.path.join(DATA, d, "meta.js"))],
                  reverse=True)


def hist_index():
    runs = []
    for r in hist_runs():
        try:
            m = _read_meta_file(os.path.join(DATA, r, "meta.js"))
            runs.append({"run": r, "updated": m.get("updated"), "h0": m["hours"][0], "h1": m["hours"][-1]})
        except Exception:
            log.warning("rularea %s din istoric e incompletă, o ignor", r)
    with open(os.path.join(DATA, "runs.js"), "w", encoding="utf-8") as fh:
        fh.write("window.ALARO_RUNS=" + json.dumps(runs) + ";\n")


def hist_migrate():
    """mută datele din vechiul format (direct în data/) în data/<rulare>/"""
    flat = os.path.join(DATA, "meta.js")
    if not os.path.exists(flat):
        return
    try:
        run = str(_read_meta_file(flat).get("run"))
    except Exception:
        run = ""
    if not _RUN_RE.match(run):
        return
    dst = os.path.join(DATA, run)
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for name in os.listdir(DATA):
        if name == run or _RUN_RE.match(name) or name == "runs.js":
            continue
        os.rename(os.path.join(DATA, name), os.path.join(dst, name))
    log.info("datele existente au fost mutate în istoric (data/%s)", run)
    hist_index()


def hist_install(tmp, run):
    """pune rularea nouă în data/<rulare>/, păstrează doar ultimele KEEP_RUNS"""
    os.makedirs(DATA, exist_ok=True)
    hist_migrate()
    dst = os.path.join(DATA, run)
    old_dir = dst + "_vechi"
    shutil.rmtree(old_dir, ignore_errors=True)
    if os.path.isdir(dst):
        os.rename(dst, old_dir)
    os.rename(tmp, dst)
    shutil.rmtree(old_dir, ignore_errors=True)
    for r in hist_runs()[KEEP_RUNS:]:
        shutil.rmtree(os.path.join(DATA, r), ignore_errors=True)
        log.info("rularea %s a ieșit din istoric", r)
    for name in os.listdir(DATA):          # resturi de la întreruperi
        if name.endswith("_vechi"):
            shutil.rmtree(os.path.join(DATA, name), ignore_errors=True)
    hist_index()


def current_meta():
    try:
        hist_migrate()
    except Exception:
        log.exception("mutarea în istoric a eșuat")
    runs = hist_runs()
    try:
        return _read_meta_file(os.path.join(DATA, runs[0], "meta.js")) if runs else None
    except Exception:
        return None


def _git_sha(data):
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def gh_existing(conf):
    """fișierele deja publicate (cale -> sha), ca să trimitem doar ce s-a schimbat"""
    try:
        ref = gh(conf, "GET", "/git/refs/heads/gh-pages", ok404=True)
        if ref.get("_status") or "object" not in ref:
            return {}
        com = gh(conf, "GET", "/git/commits/" + ref["object"]["sha"])
        t = gh(conf, "GET", "/git/trees/%s?recursive=1" % com["tree"]["sha"])
        if t.get("truncated"):
            return {}
        return {e["path"]: e["sha"] for e in t.get("tree", []) if e.get("type") == "blob"}
    except Exception as e:
        log.warning("nu pot citi ce e deja publicat (%s), trimit tot", e)
        return {}


def gh_blob_sha(conf, existing, local):
    with open(local, "rb") as fh:
        data = fh.read()
    sha = _git_sha(data)
    if existing.get("__any__", {}).get(sha):
        return sha, False
    b = gh(conf, "POST", "/git/blobs", {"content": base64.b64encode(data).decode(), "encoding": "base64"})
    return b["sha"], True


# ---------------- publicare pe GitHub Pages ----------------
GH_API = os.environ.get("ALARO_GH_API", "https://api.github.com")
GH_CONF = os.path.join(HERE, "github.txt")
GH_MARK = os.path.join(HERE, "github_publicat.txt")


def gh_config():
    if not os.path.exists(GH_CONF):
        return None
    conf = {}
    with open(GH_CONF, encoding="utf-8-sig") as fh:
        for line in fh:
            if "=" in line:
                k, v = line.split("=", 1)
                conf[k.strip().lower()] = v.strip()
    if conf.get("repo") and conf.get("token"):
        return conf
    log.warning("github.txt incomplet (trebuie repo= și token=)")
    return None


def _gh_req(conf, method, url, data=None, ok404=False):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": "Bearer " + conf["token"], "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "HartiMeteo/1.0",
        "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")[:300]
        if ok404 and e.code in (403, 404, 409, 422):
            return {"_status": e.code, "_msg": msg}
        raise RuntimeError("GitHub %s %s -> %s %s" % (method, url.split("github.com", 1)[-1], e.code, msg))


def gh(conf, method, path, data=None, ok404=False):
    url = "%s/repos/%s%s" % (GH_API, conf["repo"], path)
    for i in range(5):
        try:
            return _gh_req(conf, method, url, data, ok404)
        except RuntimeError:
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            if i == 4:
                raise
            log.warning("GitHub: conexiune întreruptă (%s), reîncerc...", e)
            time.sleep(5 * (i + 1))


def ensure_repo(conf):
    r = gh(conf, "GET", "", ok404=True)
    if r.get("_status") != 404:
        return
    name = conf["repo"].split("/", 1)[1]
    log.info("depozitul %s nu există, îl creez...", conf["repo"])
    r = _gh_req(conf, "POST", GH_API + "/user/repos", {"name": name, "description":
        "Hărți meteo %s rafinate pe relief pentru România" % C["title"], "auto_init": True}, ok404=True)
    if r.get("_status"):
        raise RuntimeError("nu pot crea depozitul %s (%s). Creează-l pe github.com (Public, gol) și dă-i "
                           "tokenului acces la el." % (conf["repo"], r.get("_msg", "")[:120]))
    time.sleep(3)


def publish(force=False):
    conf = gh_config()
    if not conf:
        return
    meta = current_meta()
    if not meta:
        return
    import hashlib
    pages = [n for n in ("index.html", "comparare.html") if os.path.exists(os.path.join(HERE, n))]
    ph = hashlib.md5(b"".join(open(os.path.join(HERE, n), "rb").read() for n in pages)).hexdigest()[:10]
    stamp = "%s %s %s" % (meta.get("run"), meta.get("updated"), ph)
    if not force and os.path.exists(GH_MARK):
        with open(GH_MARK, encoding="utf-8") as fh:
            if fh.read().strip() == stamp:
                return
    ensure_repo(conf)
    log.info("public pe GitHub (%s)...", conf["repo"])
    files = [(n, os.path.join(HERE, n)) for n in pages]
    for root, _dirs, names in os.walk(DATA):
        for name in sorted(names):
            if name.endswith(".js"):
                full = os.path.join(root, name)
                files.append(("data/" + os.path.relpath(full, DATA).replace(os.sep, "/"), full))
    first = gh(conf, "POST", "/git/blobs", {"content": "", "encoding": "utf-8"}, ok404=True)
    if first.get("_status") == 409:
        gh(conf, "PUT", "/contents/README.md", {"message": "init", "content": base64.b64encode(
            ("Hărți meteo %s rafinate pe relief (date %s prin Open-Meteo)\n" % (C["title"], C["who"])).encode()).decode()})
    tree = [{"path": ".nojekyll", "mode": "100644", "type": "blob",
             "sha": gh(conf, "POST", "/git/blobs", {"content": "", "encoding": "utf-8"})["sha"]}]
    existing = gh_existing(conf)
    existing["__any__"] = {s: True for p, s in existing.items() if p != "__any__"}
    sent = 0
    for path, local in files:
        sha, up = gh_blob_sha(conf, existing, local)
        sent += up
        tree.append({"path": path, "mode": "100644", "type": "blob", "sha": sha})
    log.info("fișiere: %d în total, %d trimise acum", len(files), sent)
    tsha = gh(conf, "POST", "/git/trees", {"tree": tree})["sha"]
    csha = gh(conf, "POST", "/git/commits", {"message": "date " + stamp, "tree": tsha, "parents": []})["sha"]
    r = gh(conf, "PATCH", "/git/refs/heads/gh-pages", {"sha": csha, "force": True}, ok404=True)
    if r.get("_status"):
        gh(conf, "POST", "/git/refs", {"ref": "refs/heads/gh-pages", "sha": csha})
    pg = gh(conf, "GET", "/pages", ok404=True)
    if pg.get("_status") in (403, 404):
        r = gh(conf, "POST", "/pages", {"source": {"branch": "gh-pages", "path": "/"}}, ok404=True)
        if r.get("_status"):
            log.warning("activează GitHub Pages manual: Settings > Pages > Branch: gh-pages / (root)")
    owner, name = conf["repo"].split("/", 1)
    with open(GH_MARK, "w", encoding="utf-8") as fh:
        fh.write(stamp)
    log.info("publicat: https://%s.github.io/%s/", owner.lower(), name)


# ---------------- actualizare ----------------
def step_names():
    names = list(C["surf"])
    for short, src in UA_SRC.items():
        for L in LEVELS:
            names.append("%s_%dhPa" % (src, L))
    return names


def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


CHECK = {"București": (44.43, 26.10), "Cluj-Napoca": (46.77, 23.60), "Vf. Omu": (45.445, 25.457), "Constanța": (44.18, 28.65)}


def at(a, lat, lon):
    j = int(round((lat - LAT0) / DLAT)); i = int(round((lon - LON0) / DLON))
    return float(a[j, i])


def build_fields(raw, acc, p1, h, tr, first):
    if raw.get("visibility") is None:
        raw["visibility"] = np.full_like(raw["temperature_2m"], OPTIONAL["visibility"])
    if raw.get("cape") is None:
        raw["cape"] = np.zeros_like(raw["temperature_2m"])
    if "freezing_level_height" in raw and raw["freezing_level_height"] is None:
        raw["freezing_level_height"] = np.clip(tr.zm + raw["temperature_2m"] / 0.0065, 0, None).astype(np.float32)
    c = {"t": raw["temperature_2m"]}
    if "relative_humidity_2m" in raw:
        c["td"] = td_from_rh(c["t"], raw["relative_humidity_2m"])
    else:
        c["td"] = np.minimum(raw["dew_point_2m"], c["t"])
    c["u"] = raw["wind_u_component_10m"]; c["v"] = raw["wind_v_component_10m"]
    c["gust"] = raw["wind_gusts_10m"]
    c["tcc"], c["lcc"], c["mcc"], c["hcc"] = (np.clip(raw[k], 0, 100) for k in
                                              ("cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"))
    c["vis"] = np.clip(raw["visibility"], 0, None)
    if C["ua"]:
        zm = tr.zm + 2
        g850 = (raw["temperature_850hPa"] - c["t"]) / np.maximum(raw["geopotential_height_850hPa"] - zm, 1)
        g700 = (raw["temperature_700hPa"] - c["t"]) / np.maximum(raw["geopotential_height_700hPa"] - zm, 1)
        gamma = np.where(raw["geopotential_height_850hPa"] - zm >= 400, g850, g700)
        gamma = np.clip(gamma, -0.0098, 0.010).astype(np.float32)
        low = raw["geopotential_height_850hPa"] - zm < 600
        us = np.where(low, raw["wind_u_component_700hPa"], raw["wind_u_component_850hPa"])
        vs = np.where(low, raw["wind_v_component_700hPa"], raw["wind_v_component_850hPa"])
    else:
        gamma = np.float32(-0.0060)
        us, vs = raw["wind_u_component_200m"], raw["wind_v_component_200m"]
    f = refine(tr, c, gamma, us, vs)
    f["p1"] = p1
    p = raw["pressure_msl"]
    f["mslp"] = (p / 100.0 if float(np.median(p)) > 5000 else p).astype(np.float32)   # ECMWF dă Pa, ICON hPa
    f["cape"] = np.clip(raw["cape"], 0, None)
    ks = [k for k in acc if p1 < k <= h]
    P = np.clip(sum(acc[k]["precipitation"] for k in ks), 0, None)
    S = np.clip(sum(acc[k]["snowfall_water_equivalent"] for k in ks), 0, None)
    Pf = refine_precip(tr, P, us, vs)
    frac = np.where(P > 0.01, np.clip(S / np.maximum(P, 1e-6), 0, 1), 0)
    frac_f = np.clip(frac + (c["t"] - f["t2m"]) / 2.5 * (P > 0.01), 0, 1)
    frac_f = np.where(f["t2m"] > 4, 0, frac_f)
    f["prectot"] = Pf; f["lssnow"] = (Pf * frac_f).astype(np.float32)
    if C["ua"]:
        f["frz"] = np.clip(raw["freezing_level_height"], 0, None)
        f["thick"] = ((raw["geopotential_height_500hPa"] - raw["geopotential_height_1000hPa"]) / 10).astype(np.float32)
        for short, src in UA_SRC.items():
            for L in LEVELS:
                f["%s%d" % (short, L)] = raw["%s_%dhPa" % (src, L)]
    if first:
        for n, (la, lo) in CHECK.items():
            log.info("verificare +%02dh %s: relief %4.0f m (model ~%4.0f m) · T model %.1f -> %.1f °C · "
                     "precip %.2f -> %.2f mm · vânt %.1f -> %.1f m/s", h, n, at(tr.zf, la, lo), at(tr.zm, la, lo),
                     at(c["t"], la, lo), at(f["t2m"], la, lo), at(P, la, lo), at(Pf, la, lo),
                     math.hypot(at(c["u"], la, lo), at(c["v"], la, lo)), math.hypot(at(f["u10"], la, lo), at(f["v10"], la, lo)))
        log.info("verificare masă precipitații: model %.3f mm, rafinat %.3f mm (medie pe zonă)", float(P.mean()), float(Pf.mean()))
    return f


def ext_ok(raw, tr):
    """verificare de bun-simț pentru primul pas al modelului de continuare (grilă/orientare corecte)"""
    t = raw["temperature_2m"]; p = raw["pressure_msl"]
    p = p / 100.0 if float(np.median(p)) > 5000 else p
    corr = float(np.corrcoef(t.ravel(), tr.zf.ravel())[0, 1])
    ok = -45 < float(t.min()) and float(t.max()) < 50 and float(t.std()) > 0.3 and 930 < float(np.median(p)) < 1070 and corr < 0.2
    log.info("%s: verificare T %.1f…%.1f °C, presiune mediană %.0f hPa, corelație T–relief %.2f -> %s",
             EXT["title"], float(t.min()), float(t.max()), float(np.median(p)), corr, "bun" if ok else "RESPINS")
    return ok


def update(force):
    ref, valid, completed = latest_run()
    run = ref.strftime("%Y%m%d%H")
    avail = {int(round((t - ref).total_seconds() / 3600)): (t, "main") for t in valid}
    if EXT:
        eav, ecomp = ext_run(ref)
        if eav and ecomp:
            for k, t in eav.items():
                if k >= EXT["from_h"]:
                    avail[k] = (t, "ext")
        elif eav:
            log.info("%s: rularea %s e încă în curs, deocamdată doar până la +%dh", EXT["title"], run, EXT["from_h"] - 1)
    steps = [s for s in STEPS if s in avail]
    if not steps:
        raise RuntimeError("rularea %s nu are încă ore disponibile" % run)
    old = current_meta()
    same_grid = old and old.get("grid", {}).get("w") == CW and old.get("grid", {}).get("h") == CH
    if not completed and old and not force and same_grid:
        log.info("rularea %s e încă în curs la sursă, aștept să fie completă", run)
        return
    if not force and old and same_grid and old.get("format") == FORMAT and old.get("run") == run and old.get("hours") == steps:
        log.info("deja la zi: rularea %s, %d momente", run, len(steps))
        return
    tr = Terrain(load_dem())
    tr_ext = Terrain(tr.zf, EXT["sigma_m"]) if EXT else None
    grid()
    names = step_names()
    ext_steps = [h for h in steps if avail[h][1] == "ext"]
    if ext_steps:
        try:
            probe = {}
            for c in chunks(names, 6):
                probe.update(read_vars(om_url(ref, avail[ext_steps[0]][0], "ext"), c, "ext"))
            good = ext_ok(probe, tr_ext)
        except Exception as e:
            log.warning("%s nu poate fi citit (%s)", EXT["title"], e)
            good = False
        if not good:
            log.warning("continuarea cu %s e oprită pentru această rulare; prognoza merge până la +%dh",
                        EXT["title"], EXT["from_h"] - 1)
            for k in [k for k, v in avail.items() if v[1] == "ext"]:
                del avail[k]
            steps = [h for h in steps if h in avail]
            ext_steps = []
            if not force and old and same_grid and old.get("run") == run and old.get("hours") == steps:
                log.info("deja la zi: rularea %s, %d momente", run, len(steps))
                return
    log.info("%s: rularea %s (%s), momente: %s", C["title"], run, "completă" if completed else "în curs", steps)
    tmp = DATA + "_nou"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        hrs = sorted(k for k in avail if 0 < k <= steps[-1])
        accl = list(ex.map(lambda k: read_vars(om_url(ref, avail[k][0], avail[k][1]), ACC_SRC, avail[k][1]), hrs))
        acc = dict(zip(hrs, accl))
        log.info("precipitații citite (%d momente, %.1f MB)", len(hrs), STATS["bytes"] / 1e6)
        mg = {}
        pint = []
        p1 = 0

        def submit(h):
            t, src = avail[h]
            url = om_url(ref, t, src)
            return [ex.submit(read_vars, url, c, src) for c in chunks(names, 3)]
        pending = {steps[0]: submit(steps[0])}
        for n, h in enumerate(steps):
            if n + 1 < len(steps):
                pending[steps[n + 1]] = submit(steps[n + 1])
            raw = {}
            for fut in pending.pop(h):
                raw.update(fut.result())
            src = avail[h][1]
            f = build_fields(raw, acc, p1, h, tr_ext if src == "ext" else tr, n == 0 or (src == "ext" and h == ext_steps[0]))
            write_hour(tmp, h, f)
            pint.append([p1, h])
            for name in MG_VARS:
                sub = f[name][::MG_STEP, ::MG_STEP]
                if name not in mg:
                    mg[name] = np.zeros((len(steps),) + sub.shape, dtype=np.float32)
                mg[name][n] = sub
            log.info("+%02dh gata%s (%d cereri, %.1f MB până acum)", h, " (%s)" % EXT["title"] if src == "ext" else "",
                     STATS["req"], STATS["bytes"] / 1e6)
            p1 = h
            del raw, f
    write_orog(tmp, tr.zf)
    write_meteograms(tmp, mg)
    meta = {"format": FORMAT, "model": MODEL, "run": run, "hours": steps, "pint": pint, "levels": LEVELS,
            "grid": {"w": CW, "h": CH, "lon0": LON0, "lat0": LAT0, "dlon": DLON, "dlat": DLAT},
            "ua_step": UA_STEP, "mg": {"step": MG_STEP, "tile": MG_TILE},
            "parts": ([{"to": EXT["from_h"] - 1, "name": "%s %s" % (C["title"], C["native"])},
                       {"to": steps[-1], "name": "%s %s" % (EXT["title"], EXT["native"])}] if ext_steps else []),
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(os.path.join(tmp, "meta.js"), "w", encoding="utf-8") as fh:
        fh.write("window.ALARO_META=" + json.dumps(meta) + ";\n")
    hist_install(tmp, run)
    size = sum(os.path.getsize(os.path.join(r, x)) for r, _d, fs in os.walk(os.path.join(DATA, run)) for x in fs)
    log.info("gata: %d momente în %.0f s, %d cereri, %.0f MB descărcați, %.1f MB pe disc",
             len(steps), time.time() - t0, STATS["req"], STATS["bytes"] / 1e6, size / 1e6)


def main():
    force = "--force" in sys.argv
    try:
        update(force)
    finally:
        try:
            publish("--publica" in sys.argv)
        except Exception:
            log.exception("publicarea pe GitHub a eșuat (se reîncearcă la următoarea rulare)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(HERE, C["log"]), encoding="utf-8")]
                        + ([logging.StreamHandler(sys.stdout)] if sys.stdout else []))
    try:
        main()
    except Exception:
        log.exception("actualizarea a eșuat")
        sys.exit(1)
