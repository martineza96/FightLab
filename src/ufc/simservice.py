"""Serve notebook 10's Markov-chain fight simulator to the app.

SimService builds a Simulator on every fight in the data with the settings chosen on
validation in notebook 10, calibrates the league multipliers once (cached per data
fingerprint in data/processed/models/), and turns one matchup's 10,000 simulations into
plain JSON: outcome odds, Monte Carlo diagnostics, distributions, and a few complete
fights (step-by-step logs) for the animated replay.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ufc import markov as MK
from ufc.predict import _fingerprint

CHOSEN = dict(k_min=40.0, k_ko=400.0, k_sub=4.0)   # notebook 10, section 4 (lowest validation log loss)
HALF_LIFE = 2.0
CAL_KEYS = ["up_mult", "clinch_mult", "ko_mult", "sub_mult"]
POS_NAMES = ["distance", "clinch", "A top", "B top"]

# Notebook 10's executed results, shown on the Model tab (test = fights from 2024 on).
REPORT = {
    "test": [
        {"model": "Always pick the red corner", "accuracy": 55.1, "log_loss": 0.690, "roc_auc": 0.5},
        {"model": "Simple formula on the simulator's stats", "accuracy": 60.0, "log_loss": 0.665, "roc_auc": 0.634},
        {"model": "Simulator (10,000 fights each)", "accuracy": 61.7, "log_loss": 0.656, "roc_auc": 0.659},
        {"model": "Predictor (the app's main model)", "accuracy": 63.8, "log_loss": 0.634, "roc_auc": 0.696},
    ],
    "judges": {"decisions": 2600, "accuracy": 85.3, "kd_strikes": 8.2, "top_min_strikes": 4.3, "td_strikes": 2.3},
    "checks": {"strikes_per_min": [6.73, 7.13], "fight_minutes": [10.95, 10.85], "td_per_15": [3.06, 2.82]},
    "settings": {"half_life_years": HALF_LIFE, "k_minutes": CHOSEN["k_min"], "k_ko_strikes": CHOSEN["k_ko"]},
    "precision_10k": 0.8,
}


class SimService:
    def __init__(self, processed="data/processed", cache=True):
        p = Path(processed)
        self.ff = pd.read_parquet(p / "fighter_fights.parquet")
        fights = pd.read_parquet(p / "fights.parquet")
        self.latest = self.ff.date.max()
        path = p / "models" / f"markov_{_fingerprint(processed)}.json"
        self.sim = MK.Simulator(self.ff, self.latest + pd.Timedelta(days=1), MK.Params(**CHOSEN), half_life=HALF_LIFE)
        if cache and path.exists():
            for k, v in json.loads(path.read_text()).items():
                setattr(self.sim.P, k, v)
        else:
            recent = fights[fights.corners_reliable & fights.winner.isin(["Red", "Blue"]) &
                            (fights.date >= self.latest - pd.DateOffset(years=5))]
            MK.calibrate(self.sim, recent.sample(min(800, len(recent)), random_state=0), n=300, iters=4, verbose=False)
            if cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({k: float(getattr(self.sim.P, k)) for k in CAL_KEYS}))
        self.last_wc = self.ff.sort_values("date").groupby("fighter").weight_class.last()

    # ------------------------------------------------------------------ one matchup
    def run(self, a, b, rounds=3, n=10000, seed=0, replays=6, spaghetti=60):
        S = self.sim
        wc = self.last_wc.get(a, "ALL")
        pa, pb, cin = S.params_for(S.career_now(a), S.career_now(b), [wc])
        logn = max(replays, spaghetti)
        o = MK.simulate(pa, pb, cin, [rounds], S.judge, S.P, n=n, seed=seed, log=logn, raw=True)
        raw = {k: v[0] for k, v in o["raw"].items()}
        end, er, mins = raw["end"], raw["end_round"], raw["minutes"]
        full = end == 0
        outc = {k: float(o[k][0]) for k in MK.OUT7}
        return {
            "a": a, "b": b, "rounds": rounds, "n": n, "seed": seed, "weight_class": wc,
            "p_a": float(o["p_a"][0]),
            "outcomes": outc,
            "finish_rate": float(1 - outc["A DEC"] - outc["B DEC"]),
            "heatmap": [[float(((end == code) & (er == r + 1)).mean()) for r in range(rounds)] for code in (1, 2, 3, 4)],
            "convergence": self._convergence(raw["a_wins"], seed),
            "length_hist": self._length_hist(end, mins, rounds),
            "strikes_hist": self._hist2(raw["a_sig"], raw["b_sig"], width=5),
            "td_hist": self._counts(raw["a_td"], raw["b_td"], top=8),
            "means": {k: float(raw[k].mean()) for k in ["a_sig", "b_sig", "a_td", "b_td", "a_kd", "b_kd", "a_sub", "b_sub",
                                                        "a_top", "b_top", "t_dist", "t_clinch", "minutes"]},
            "medians": {"a_sig": float(np.median(raw["a_sig"])), "b_sig": float(np.median(raw["b_sig"])),
                        "minutes": float(np.median(mins))},
            "dna": self._dna(pa, pb),
            "spaghetti": self._spaghetti(o, spaghetti),
            "replays": [self._replay(o, j) for j in range(replays)],
        }

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _convergence(w, seed, runs=4):
        """Running P(A wins) at log-spaced checkpoints: the real run plus re-orderings of the same simulations."""
        n = len(w)
        cps = np.unique(np.round(np.logspace(1, np.log10(n), 45)).astype(int))
        sd = w.std()
        rng = np.random.default_rng(seed + 99)
        orders = [np.arange(n)] + [rng.permutation(n) for _ in range(runs)]
        paths = [np.cumsum(w[ix])[cps - 1] / cps for ix in orders]
        return [{"n": int(c), "est": float(paths[0][i]), "lo": float(paths[0][i] - 1.96 * sd / np.sqrt(c)),
                 "hi": float(paths[0][i] + 1.96 * sd / np.sqrt(c)), **{f"r{k}": float(paths[k][i]) for k in range(1, runs + 1)}}
                for i, c in enumerate(cps)]

    @staticmethod
    def _length_hist(end, mins, rounds):
        edges = np.arange(0, rounds * 5 + 1, 1.0)
        idx = np.clip(np.digitize(mins, edges) - 1, 0, len(edges) - 2)
        out = []
        for i in range(len(edges) - 1):
            m = idx == i
            out.append({"minute": int(edges[i]), "a_finish": int((m & ((end == 1) | (end == 2))).sum()),
                        "b_finish": int((m & ((end == 3) | (end == 4))).sum()), "decision": int((m & (end == 0)).sum())})
        return out

    @staticmethod
    def _hist2(x, y, width=5):
        hi = float(np.quantile(np.r_[x, y], 0.995))
        edges = np.arange(0, hi + width, width)
        hx, _ = np.histogram(x, edges)
        hy, _ = np.histogram(y, edges)
        n = len(x)
        return [{"from": int(edges[i]), "a": float(hx[i] / n), "b": float(hy[i] / n)} for i in range(len(hx))]

    @staticmethod
    def _counts(x, y, top=8):
        x, y = np.minimum(x.astype(int), top), np.minimum(y.astype(int), top)
        return [{"k": f"{k}+" if k == top else str(k), "a": float((x == k).mean()), "b": float((y == k).mean())} for k in range(top + 1)]

    @staticmethod
    def _dna(pa, pb):
        """The matchup-specific rates the simulation runs on, in readable units."""
        rows = [("Sig. strikes / min at distance", "dist", 1, "{:.1f}"), ("Sig. strikes / min in the clinch", "clinch", 1, "{:.1f}"),
                ("Ground strikes / min on top", "gtop", 1, "{:.1f}"), ("Takedowns / 15 min standing", "td", 15, "{:.2f}"),
                ("Sub attempts / 15 min on the ground", "sub", 15, "{:.2f}"), ("Submission attempt success", "subconv", 1, "{:.0%}"),
                ("KO chance per 100 strikes landed", "ko", 100, "{:.2f}%"), ("Knockdowns per 100 standing strikes", "kd", 100, "{:.2f}"),
                ("Average time on top before stand-up", "up", None, "{:.0f} s")]
        out = []
        for label, k, mult, fmt in rows:
            va, vb = float(np.asarray(pa[k])[0]), float(np.asarray(pb[k])[0])
            if mult is None:      # stand-up hazard per minute -> mean seconds on top
                va, vb = 60 / max(va, 1e-6), 60 / max(vb, 1e-6)
            else:
                va, vb = va * mult, vb * mult
            out.append({"label": label, "a": va, "b": vb, "a_text": fmt.format(va), "b_text": fmt.format(vb),
                        "higher_better": True})   # every row is "this fighter against this opponent": more is better
        return out

    @staticmethod
    def _stop(o, j):
        ends = [L["end"][0, j] > 0 for L in o["logs"]]
        return (int(np.argmax(ends)) + 1) if any(ends) else len(o["logs"])

    def _spaghetti(self, o, k, every=6):
        """Running strike differential (A − B) every 30 s for k simulations, cut at the finish."""
        logs = o["logs"]
        sa = np.array([L["sa"][0, :k] for L in logs], float)
        sb = np.array([L["sb"][0, :k] for L in logs], float)
        diff = np.cumsum(sa - sb, axis=0)
        out = []
        for j in range(k):
            stop = self._stop(o, j)
            idx = list(range(every - 1, stop, every))
            if not idx or idx[-1] != stop - 1:
                idx.append(stop - 1)
            e = int(o["log_end"][0, j])
            a_won = e in (1, 2) or (e == 0 and o["log_p_dec_a"][0, j] >= 0.5)
            out.append({"t": [round((i + 1) * MK.DT / 60, 3) for i in idx], "d": [int(diff[i, j]) for i in idx],
                        "winner": "a" if a_won else "b", "finished": bool(e)})
        return out

    def _replay(self, o, j):
        logs = o["logs"]
        stop = self._stop(o, j)
        ev = []
        prev = 0
        for s in range(stop):
            L = logs[s]
            pos = int(L["pos"][0, j])
            if s % MK.STEPS_PER_ROUND == 0:
                prev = 0
            kind = {(0, 1): ("clinch", None), (1, 0): ("break", None), (2, 0): ("up", "b"), (3, 0): ("up", "a"),
                    (2, 3): ("reversal", "b"), (3, 2): ("reversal", "a")}.get((prev, pos))
            if kind:
                ev.append({"s": s, "type": kind[0], "who": kind[1]})
            for key, typ, who in [("kda", "kd", "a"), ("kdb", "kd", "b"), ("tda", "td", "a"), ("tdb", "td", "b"),
                                  ("suba", "sub", "a"), ("subb", "sub", "b")]:
                if L[key][0, j]:
                    ev.append({"s": s, "type": typ, "who": who})
            prev = pos
        e = int(o["log_end"][0, j])
        p_dec = float(o["log_p_dec_a"][0, j])
        return {"steps": stop, "pos": [int(logs[s]["pos"][0, j]) for s in range(stop)],
                "sa": [int(logs[s]["sa"][0, j]) for s in range(stop)], "sb": [int(logs[s]["sb"][0, j]) for s in range(stop)],
                "events": ev, "end": {0: "decision", 1: "a_ko", 2: "a_sub", 3: "b_ko", 4: "b_sub"}[e],
                "end_round": int(o["log_end_round"][0, j]) if e else None, "p_dec_a": p_dec}
