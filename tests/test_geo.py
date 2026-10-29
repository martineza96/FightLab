"""Location / home-advantage features (offline: no ESPN calls)."""
import numpy as np
import pandas as pd

from ufc import geo


def test_event_country_parsing():
    assert geo.event_country("Edmonton, Alberta, Canada") == "Canada"
    assert geo.event_country("Las Vegas, Nevada, USA") == "USA"
    assert geo.event_country("London, England, United Kingdom") == "United Kingdom"
    assert geo.event_country(None) is None


def test_country_aliases_and_continents():
    assert geo.norm_country("England") == "United Kingdom"
    assert geo.norm_country("United States") == "USA"
    assert geo.continent("Brazil") == "South America" and geo.continent("Russia") == "Europe"


def test_every_event_location_parses():
    fights = pd.read_parquet("data/processed/fights.parquet")
    ev = fights.drop_duplicates("event")
    assert ev.location.map(geo.event_country).notna().mean() > 0.99


def test_home_features():
    fights = pd.DataFrame({"fight_id": [1, 2, 3], "location": ["Rio, RJ, Brazil", "Las Vegas, Nevada, USA", "Sydney, NSW, Australia"]})
    ff = pd.DataFrame({"fight_id": [1, 1, 2, 3], "fighter": ["A", "B", "A", "C"],
                       "date": pd.to_datetime(["2019-01-01", "2019-01-01", "2020-06-01", "2022-01-01"])})
    origin = pd.DataFrame({"name": ["A", "B"], "country": ["Brazil", "USA"]})
    h = geo.home_features(ff, fights, origin).set_index(["fight_id", "fighter"])
    assert h.loc[(1, "A"), "home"] == 1 and h.loc[(1, "B"), "home"] == 0
    assert h.loc[(1, "B"), "travel"] == 1 and h.loc[(2, "A"), "same_continent"] == 0
    assert np.isnan(h.loc[(3, "C"), "home"])                        # unknown origin stays unknown, not "away"
    assert h.loc[(2, "A"), "crowd"] == 0 and h.loc[(1, "A"), "crowd"] == 1   # 2020-06 = empty arena
