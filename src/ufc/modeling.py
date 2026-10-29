"""Shared training / evaluation helpers for the prediction notebooks (04 recipe).

- corner-swap augmentation: every training fight appears twice with corners swapped,
  so the model must explain outcomes symmetrically; `red_corner` (+1 / −1) isolates
  the red-corner effect
- Clipper → median imputation (+ missing flags) → scaling → logistic regression
- time split: train before 2024-01-01, test from then on
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ufc.features import Clipper, swap_corners

SPLIT = pd.Timestamp("2024-01-01")
FLAGS = ["r_debut", "b_debut", "reach_missing"]


def augment(df):
    a = df.assign(red_corner=1)
    b = swap_corners(df).assign(red_corner=-1)
    return pd.concat([a, b], ignore_index=True)


def logistic(C=0.5):
    return make_pipeline(Clipper(0.01, 0.99), SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def boosted(seed=0):
    return HistGradientBoostingClassifier(learning_rate=0.04, max_iter=400, max_leaf_nodes=15,
                                          min_samples_leaf=60, l2_regularization=1.0, random_state=seed)


def diffs(names):
    return [f + "_diff" for f in names]


def fit(model, train, cols, target="red_win"):
    a = augment(train)
    return model.fit(a[cols], a[target])


def predict(model, test, cols):
    return model.predict_proba(test.assign(red_corner=1)[cols])[:, 1]


def scores(y, p):
    y = np.asarray(y)
    return {"accuracy": accuracy_score(y, p > 0.5), "log_loss": log_loss(y, p),
            "roc_auc": roc_auc_score(y, p) if np.std(p) > 0 else 0.5}


def bootstrap_delta(y, p_new, p_old, n=2000, seed=0):
    """95% interval for log_loss(new) − log_loss(old), resampling test fights."""
    y = np.asarray(y)
    eps = 1e-15
    def ll(p):
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = ll(np.asarray(p_new)) - ll(np.asarray(p_old))
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return d.mean(), np.quantile(boots, 0.025), np.quantile(boots, 0.975)


RATE_FEATURES = ["slpm", "sapm", "sig_acc", "sig_def", "td_per15", "td_acc", "td_def",
                 "sub_per15", "kd_per15", "ctrl_share"]
WEIGHT_LBS = {"Strawweight": 115, "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145,
              "Lightweight": 155, "Welterweight": 170, "Middleweight": 185, "Light Heavyweight": 205,
              "Heavyweight": 265, "Catch Weight": 170, "Open Weight": 230}
OUTCOMES = ["Red KO/TKO", "Red SUB", "Red DEC", "Blue KO/TKO", "Blue SUB", "Blue DEC", "Other"]


def load_model_table(processed="data/processed", shrink_k=None, split=SPLIT):
    """Fighter-fight history + every form.py feature -> one row per fight (red/blue/diff).

    shrink_k: if set, career rate stats are partially pooled toward the division mean
    (computed on training-period rows only) before the table is built.
    Returns (table, long fighter-fight table).
    """
    from ufc import form as F
    from ufc.features import add_history, build_model_table
    fights = pd.read_parquet(f"{processed}/fights.parquet")
    ff0 = pd.read_parquet(f"{processed}/fighter_fights.parquet")
    hist = pd.read_parquet(f"{processed}/elo_history.parquet")
    ff = add_history(ff0).merge(F.build_all(ff0, hist), on=F.KEY, how="left")
    if shrink_k:
        ff = shrink(ff, RATE_FEATURES, "n_fights", "weight_class", shrink_k, ref=ff[ff.date < split])
    t = build_model_table(fights, ff, extra=F.ALL_NEW)
    f = fights.set_index("fight_id")
    t["method_group"] = t.fight_id.map(f.method_group)
    t["finish_round"] = t.fight_id.map(f.finish_round)
    t["outcome"] = np.where(t.method_group.isin(["KO/TKO", "SUB", "DEC"]),
                            t.winner + " " + t.method_group, "Other")
    t["weight_lbs"] = t.weight_class.map(WEIGHT_LBS).fillna(170)
    t["women"] = (t.gender == "Women").astype(int)
    t["five_rounds"] = (t.total_rounds == 5).astype(int)
    exp = np.minimum(t.r_n_fights, t.b_n_fights)
    t["experience"] = pd.Categorical(np.select([exp == 0, exp < 5], ["debut involved", "one has 1–4"], "both 5+"),
                                     categories=["debut involved", "one has 1–4", "both 5+"])
    return t.copy(), ff


METHODS = ["KO/TKO", "SUB", "DEC"]


def method_probs(df, p_red, prior, k=3.0, loser_weight=1.0, other=0.003):
    """P(outcome) for the 7 outcomes = P(winner) × P(method | that winner wins).

    P(method | A wins) ∝ A's shrunk win-method mix × (B's shrunk loss-method mix)^loser_weight.
    Shrinking toward `prior` (dict method -> share) keeps a 2-fight fighter from looking
    like a 100% KO artist. Mixes come from career wins (ko/sub/dec_wins) and the last 5
    losses (l5_ko_l / l5_sub_l / l5_dec_l).
    """
    def mix(pre, kind):
        if kind == "win":
            cnt = {"KO/TKO": df[pre + "ko_wins"], "SUB": df[pre + "sub_wins"], "DEC": df[pre + "dec_wins"]}
        else:
            cnt = {"KO/TKO": df[pre + "l5_ko_l"], "SUB": df[pre + "l5_sub_l"], "DEC": df[pre + "l5_dec_l"]}
        m = pd.DataFrame({x: cnt[x].fillna(0).values + k * prior[x] for x in METHODS})
        return m.div(m.sum(axis=1), axis=0)
    out = {}
    for win_pre, lose_pre, p, side in [("r_", "b_", p_red, "Red"), ("b_", "r_", 1 - np.asarray(p_red), "Blue")]:
        m = mix(win_pre, "win") * mix(lose_pre, "loss") ** loser_weight
        m = m.div(m.sum(axis=1), axis=0)
        for x in METHODS:
            out[f"{side} {x}"] = np.asarray(p) * m[x].values * (1 - other)
    out = pd.DataFrame(out, index=df.index)
    out["Other"] = other
    return out[OUTCOMES]


def contributions(model, X):
    """Exact per-feature log-odds contributions for a Clipper→Imputer→Scaler→Logistic pipeline.

    Returns a DataFrame (rows = X rows, columns = features and missing-flags) whose row sums
    plus the intercept equal the model's log-odds for that row.
    """
    cols = list(X.columns)
    Z = model[:-1].transform(X)
    imp = model.named_steps["simpleimputer"]
    names = cols + [f"missing: {cols[i]}" for i in imp.indicator_.features_] if imp.add_indicator else cols
    c = pd.DataFrame(np.asarray(Z) * model[-1].coef_[0], columns=names, index=X.index)
    return c, model[-1].intercept_[0]


def swap_outcome(o):
    return o.str.replace("Red", "§").str.replace("Blue", "Red").str.replace("§", "Blue")


def shrink(df, cols, n_col, group_col, k, ref=None):
    """Partial pooling: (n·x + k·μ) / (n + k), μ = mean of the group in `ref` (training rows).

    A fighter with 1 fight gets mostly the division average; one with 20 keeps their own.
    Missing x (no history) becomes μ.
    """
    ref = df if ref is None else ref
    out = df.copy()
    for c in cols:
        mu = ref.groupby(group_col)[c].mean()
        m = df[group_col].map(mu).fillna(ref[c].mean())
        n = df[n_col].fillna(0)
        x = df[c].where(df[c].notna(), m)
        out[c] = (n * x + k * m) / (n + k)
    return out
