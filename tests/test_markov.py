"""Markov fight-simulator tests. Run from the project root:  PYTHONPATH=src pytest -q tests"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ufc import markov as MK

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sim():
    ff = pd.read_parquet(ROOT / "data/processed/fighter_fights.parquet")
    return MK.Simulator(ff, "2024-01-01", MK.Params(k_min=40, k_ko=400, up_mult=0.55), half_life=2)


def test_fighter_vs_themselves_is_even(sim):
    o = sim.matchup("Islam Makhachev", "Islam Makhachev", rounds=5, n=20000, seed=1)
    assert abs(o["p_a"][0] - 0.5) < 0.015


def test_swapping_fighters_mirrors_the_result(sim):
    ab = sim.matchup("Jon Jones", "Tom Aspinall", rounds=5, n=20000, seed=2)
    ba = sim.matchup("Tom Aspinall", "Jon Jones", rounds=5, n=20000, seed=3)
    assert abs(ab["p_a"][0] - (1 - ba["p_a"][0])) < 0.02
    assert abs(ab["A KO/TKO"][0] - ba["B KO/TKO"][0]) < 0.02


def test_outcomes_sum_to_one_and_seed_is_reproducible(sim):
    a = sim.matchup("Alexander Volkanovski", "Max Holloway", rounds=3, n=3000, seed=5)
    b = sim.matchup("Alexander Volkanovski", "Max Holloway", rounds=3, n=3000, seed=5)
    assert math.isclose(sum(a[k][0] for k in MK.OUT7), 1, abs_tol=1e-9)
    assert a["p_a"][0] == b["p_a"][0]
    assert math.isclose(a["finish_round"][0].sum(), 1 - a["A DEC"][0] - a["B DEC"][0], abs_tol=0.02)


def test_career_sums_use_only_earlier_fights(sim):
    ff, ex, cs = sim.ff, sim.ex, MK.career_sums(sim.ff, sim.ex)          # no decay: plain sums
    for name in ["Alexander Volkanovski", "Movsar Evloev", "Valentina Shevchenko"]:
        rows = ff[ff.fighter == name].sort_values(["date", "fight_id"])
        for i, idx in enumerate(rows.index):
            earlier = rows.index[:i]
            assert math.isclose(cs.at[idx, "o_dist_land"], ex.loc[earlier, "o_dist_land"].sum())
            assert cs.at[idx, "n_prev"] == i


def test_play_by_play_ends_with_the_result(sim):
    o = sim.matchup("Islam Makhachev", "Ilia Topuria", rounds=5, n=50, seed=11, log=3)
    for j in range(3):
        ev, strikes, result = MK.play_by_play(o, ("Makhachev", "Topuria"), 0, j)
        assert len(strikes) >= 1 and isinstance(result, str)
        if o["log_end"][0, j]:
            assert "wins by" in ev.event.iloc[-1]
        else:
            assert len(strikes) == 5 and result.startswith("Goes the distance")
