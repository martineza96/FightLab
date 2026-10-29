"""Where fights happen and where fighters are from: home-crowd features.

Event country comes from the ufcstats event location ("City, Region, Country").
Fighter country isn't in the ufcstats data, so it comes from ESPN's public athlete
profiles ("citizenship"), looked up by name once and cached per fighter. Citizenship is
a proxy for "the crowd is with you": it ignores hometowns within a country and dual
nationals.
"""
import json
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ufc.ingest import _name_key as _ascii_key

SEARCH = "https://site.api.espn.com/apis/search/v2"
ATHLETE = "https://site.web.api.espn.com/apis/common/v3/sports/mma/athletes/{id}"
HEADERS = {"User-Agent": "ufc-dataset-local-app/1.0 (personal project)"}

# no fans in the arena (COVID): Brasília behind closed doors on 2020-03-14 through the last
# empty event before UFC 261 (2021-04-24, full crowd in Jacksonville). A few Jan-2021 Abu Dhabi
# events admitted ~2,000 fans; they're still counted as empty here.
EMPTY_FROM, EMPTY_TO = pd.Timestamp("2020-03-14"), pd.Timestamp("2021-04-17")

COUNTRY_ALIASES = {
    "united states": "USA", "united states of america": "USA", "us": "USA", "usa": "USA",
    "england": "United Kingdom", "scotland": "United Kingdom", "wales": "United Kingdom",
    "northern ireland": "United Kingdom", "great britain": "United Kingdom", "uk": "United Kingdom",
    "united kingdom": "United Kingdom", "republic of ireland": "Ireland",
    "korea": "South Korea", "republic of korea": "South Korea", "south korea": "South Korea",
    "czechia": "Czech Republic", "uae": "United Arab Emirates", "holland": "Netherlands",
    "russian federation": "Russia", "the netherlands": "Netherlands", "default": None,
}

CONTINENT = {
    "North America": ["USA", "Canada", "Mexico", "Puerto Rico", "Cuba", "Jamaica", "Dominican Republic", "Haiti",
                      "Panama", "Costa Rica", "Guatemala", "Honduras", "El Salvador", "Nicaragua", "Trinidad and Tobago",
                      "Bahamas", "Barbados", "Suriname"],
    "South America": ["Brazil", "Argentina", "Chile", "Peru", "Colombia", "Venezuela", "Ecuador", "Uruguay", "Paraguay", "Bolivia"],
    "Europe": ["United Kingdom", "Ireland", "France", "Germany", "Netherlands", "Belgium", "Sweden", "Norway", "Denmark",
               "Finland", "Iceland", "Poland", "Czech Republic", "Slovakia", "Austria", "Switzerland", "Italy", "Spain",
               "Portugal", "Croatia", "Serbia", "Bosnia and Herzegovina", "Slovenia", "Montenegro", "Albania", "Kosovo",
               "Romania", "Bulgaria", "Greece", "Hungary", "Lithuania", "Latvia", "Estonia", "Ukraine", "Belarus",
               "Moldova", "Russia", "North Macedonia", "Luxembourg", "Malta", "Cyprus"],
    "Asia": ["China", "Japan", "South Korea", "Philippines", "Singapore", "Thailand", "Vietnam", "Indonesia", "Malaysia",
             "India", "Mongolia", "Kazakhstan", "Kyrgyzstan", "Uzbekistan", "Tajikistan", "Turkmenistan", "Afghanistan",
             "Taiwan", "Hong Kong", "Myanmar", "Pakistan"],
    "Middle East & Caucasus": ["United Arab Emirates", "Saudi Arabia", "Qatar", "Bahrain", "Kuwait", "Iran", "Iraq",
                               "Israel", "Jordan", "Lebanon", "Syria", "Turkey", "Azerbaijan", "Armenia", "Georgia",
                               "Palestine", "Egypt"],
    "Oceania": ["Australia", "New Zealand", "Fiji", "Samoa", "Tonga", "Papua New Guinea"],
    "Africa": ["Nigeria", "Cameroon", "South Africa", "Ghana", "Kenya", "Morocco", "Tunisia", "Algeria", "Senegal",
               "DR Congo", "Democratic Republic of the Congo", "Angola", "Zimbabwe", "Uganda", "Ethiopia", "Ivory Coast"],
}
CONTINENT_OF = {c: k for k, cs in CONTINENT.items() for c in cs}


def norm_country(c):
    if c is None or (isinstance(c, float) and np.isnan(c)):
        return None
    c = str(c).strip()
    return COUNTRY_ALIASES.get(c.lower(), c)


def event_country(location):
    if not isinstance(location, str) or not location.strip():
        return None
    return norm_country(location.split(",")[-1])


def continent(country):
    return CONTINENT_OF.get(country)


def _key(name):
    plain = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return _ascii_key(plain)


# ---------------------------------------------------------------------------
# ESPN lookup (cached, resumable)
# ---------------------------------------------------------------------------
def _get(url, params=None, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404,):
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def _country_of(a):
    """ESPN puts 'default' in citizenship when it's unknown; fall back to the flag, then to birthplace."""
    for c in [a.get("citizenship"), (a.get("flag") or {}).get("alt"), (a.get("citizenshipCountry") or {}).get("abbreviation"),
              ((a.get("birthPlace") or {}).get("country") if isinstance(a.get("birthPlace"), dict) else None)]:
        c = norm_country(c)
        if c and c.lower() not in ("default", "none", ""):
            return COUNTRY_ALIASES.get(c.lower(), {"BRA": "Brazil", "USA": "USA", "GBR": "United Kingdom"}.get(c, c))
    return None


def _athlete(espn_id):
    d = _get(ATHLETE.format(id=espn_id))
    a = (d or {}).get("athlete") or {}
    dob = pd.to_datetime(a.get("displayDOB"), errors="coerce", format="%d/%m/%Y")
    return {"espn_id": str(espn_id), "espn_name": a.get("displayName"),
            "country": _country_of(a),
            "gym": (a.get("association") or {}).get("name"),
            "espn_dob": None if pd.isna(dob) else dob.date().isoformat(),
            "weight_class": (a.get("weightClass") or {}).get("text")}


def lookup(name, dob=None, cache_dir="data/processed/cache/espn_athletes", pause=0.25):
    """Find one of our fighters on ESPN and return their country (cached per fighter)."""
    dob_tag = "" if dob is None or pd.isna(dob) else pd.Timestamp(dob).strftime("%Y%m%d")
    cache = Path(cache_dir) / f"{_key(name) or 'x'}_{dob_tag}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    base = name.split(" (")[0]
    res = _get(SEARCH, {"query": base, "limit": 8})
    cands = []
    for block in (res or {}).get("results", []):
        if block.get("type") != "player":
            continue
        for c in block.get("contents", []):
            if c.get("sport") == "mma" and "~a:" in (c.get("uid") or ""):
                cands.append({"id": c["uid"].split("~a:")[-1], "name": c.get("displayName")})
    exact = [c for c in cands if _key(c["name"]) == _key(base)]
    pool = exact or cands[:3]
    out = {"name": name, "found": False, "match": None}
    best = None
    for c in pool[:4]:
        time.sleep(pause)
        a = _athlete(c["id"])
        dob_ok = dob is not None and not pd.isna(dob) and a["espn_dob"] == pd.Timestamp(dob).date().isoformat()
        score = (2 if dob_ok else 0) + (1 if _key(a["espn_name"] or "") == _key(base) else 0)
        if best is None or score > best[0]:
            best = (score, a, dob_ok)
        if dob_ok:
            break
    if best and best[0] >= 1:
        score, a, dob_ok = best
        out.update(a, found=True, match="dob-confirmed" if dob_ok else ("exact-name" if score >= 1 else "fuzzy"),
                   candidates=len(cands))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    time.sleep(pause)
    return out


def fetch_origins(directory, cache_dir="data/processed/cache/espn_athletes", out="data/processed/fighter_origin.parquet",
                  since="2005-01-01", workers=4, progress=True):
    """Look up every fighter with a UFC fight since `since`; write a tidy parquet."""
    d = directory.reset_index() if "name" not in directory.columns else directory
    d = d[d.last_fight >= since].drop_duplicates("name")
    rows, t0 = [], time.time()
    def job(r):
        try:
            return lookup(r.name, r.dob, cache_dir)
        except Exception as e:                       # one bad record shouldn't stop the run
            return {"name": r.name, "found": False, "match": f"error: {e}"[:80]}
    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(job, d.itertuples(index=False))):
            rows.append(res)
            if progress and (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(d)} looked up · {sum(r['found'] for r in rows)} found · {time.time() - t0:.0f}s", flush=True)
    o = pd.DataFrame(rows)
    keep = ["name", "found", "match", "espn_id", "espn_name", "country", "gym", "espn_dob", "weight_class"]
    o = o.reindex(columns=keep)
    o.to_parquet(out, index=False)
    return o


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
HOME_FEATURES = ["home", "same_continent", "travel"]


def home_features(ff, fights, origin):
    """Per fighter-fight: home (same country as the event), same continent, travelled across continents,
    plus whether the arena had fans. Unknown origin -> NaN (not 'away')."""
    f = fights.set_index("fight_id")
    out = ff[["fight_id", "fighter", "date"]].copy()
    out["event_country"] = out.fight_id.map(f.location.map(event_country))
    country = origin.set_index("name").country
    out["country"] = out.fighter.map(country)
    known = out.country.notna() & out.event_country.notna()
    out["home"] = (out.country == out.event_country).astype(float).where(known)
    ec, fc = out.event_country.map(continent), out.country.map(continent)
    ok = known & ec.notna() & fc.notna()
    out["same_continent"] = (ec == fc).astype(float).where(ok)
    out["travel"] = (ec != fc).astype(float).where(ok)
    out["crowd"] = (~out.date.between(EMPTY_FROM, EMPTY_TO)).astype(float)
    return out.drop(columns=["date"])


# ---------------------------------------------------------------------------
# Wikidata fallback for fighters ESPN has no country for
# ---------------------------------------------------------------------------
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_QUERY = """SELECT ?p ?pLabel ?dob ?cLabel WHERE {
  ?p wdt:P106 wd:Q11607585 .                       # occupation: mixed martial artist
  OPTIONAL { ?p wdt:P569 ?dob }
  OPTIONAL { ?p wdt:P1532 ?c1 }                    # country for sport
  OPTIONAL { ?p wdt:P27 ?c2 }                      # citizenship
  BIND(COALESCE(?c1, ?c2) AS ?c)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }"""


def fetch_wikidata(path="data/raw/wikidata_mma_fighters.json"):
    r = requests.get(WIKIDATA_SPARQL, params={"query": WIKIDATA_QUERY},
                     headers={**HEADERS, "Accept": "application/sparql-results+json"}, timeout=120)
    r.raise_for_status()
    Path(path).write_text(r.text)
    return path


def fill_from_wikidata(origin, directory, path="data/raw/wikidata_mma_fighters.json"):
    """Fill missing countries: same name key AND same birth date (or a unique name if we lack a dob)."""
    rows = json.loads(Path(path).read_text())["results"]["bindings"]
    wd = pd.DataFrame({"key": [_key(b["pLabel"]["value"]) for b in rows],
                       "dob": [pd.to_datetime(b.get("dob", {}).get("value", "")[:10], errors="coerce") for b in rows],
                       "country": [norm_country(b.get("cLabel", {}).get("value")) for b in rows]})
    wd = wd.dropna(subset=["country"]).drop_duplicates(["key", "dob"])
    o = origin.copy()
    o["country"] = o.country.where(~o.country.isin(["default"]), None)
    d = directory.drop_duplicates("name").set_index("name").dob
    filled = 0
    for i in o.index[o.country.isna()]:
        name = o.at[i, "name"]
        k, dob = _key(name.split(" (")[0]), d.get(name)
        c = wd[wd.key == k]
        if pd.notna(dob):
            c = c[c.dob == dob]
        elif len(c) != 1:
            continue
        if len(c):
            o.at[i, "country"] = c.country.iloc[0]
            o.at[i, "match"] = "wikidata"
            o.at[i, "found"] = True
            filled += 1
    o.attrs["wikidata_filled"] = filled
    return o


def build_origins(processed="data/processed", refresh_wikidata=False):
    """Full pipeline: ESPN lookups (cached) + Wikidata fill -> data/processed/fighter_origin.parquet."""
    directory = pd.read_parquet(Path(processed) / "fighters.parquet")
    o = fetch_origins(directory, cache_dir=Path(processed) / "cache" / "espn_athletes",
                      out=Path(processed) / "fighter_origin.parquet", since="1993-01-01")
    wd = Path("data/raw/wikidata_mma_fighters.json")
    if refresh_wikidata or not wd.exists():
        fetch_wikidata(wd)
    o = fill_from_wikidata(o, directory, wd)
    o["found"] = o.country.notna()
    o.to_parquet(Path(processed) / "fighter_origin.parquet", index=False)
    return o
