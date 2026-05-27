import numpy as np
import pandas as pd
import random
import string
import json
import joblib
from pathlib import Path

# ─── Load artifacts ───────────────────────────────────────────────────────────
BASE = Path(__file__).parent

with open(BASE / "config.json") as f:
    CONFIG = json.load(f)

W_STAR    = CONFIG["optimal_weight"]          # 0.3
THR_BLOCK = CONFIG["threshold_layer3"]["block"]      # 0.8
THR_ALERT = CONFIG["threshold_layer3"]["alert_otp"]  # 0.5
FEAT_A    = CONFIG["feature_list"]["model_a"]
FEAT_B    = CONFIG["feature_list"]["model_b"]

MODEL_A = joblib.load(BASE / "model_a_pipeline.joblib")
MODEL_B = joblib.load(BASE / "model_b_pipeline.joblib")

PAYEE_PROFILE = pd.read_csv(BASE / "payee_profile_store.csv")
# Normalise column name — support both 'mule_account_id' and 'nameDest'
_id_col = "mule_account_id" if "mule_account_id" in PAYEE_PROFILE.columns else PAYEE_PROFILE.columns[0]
PAYEE_PROFILE = PAYEE_PROFILE.set_index(_id_col)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _rand_id(prefix: str, n: int = 4) -> str:
    return prefix + "".join(random.choices(string.digits, k=n))


def generate_accounts(n: int = 5, seed: int = None) -> list[dict]:
    """
    Create n random sender accounts with a risk profile each.
    Called once per simulation session.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    profiles = []
    for _ in range(n):
        risk = random.choices(
            ["low", "medium", "high"], weights=[0.60, 0.28, 0.12]
        )[0]
        profiles.append(
            {
                "account_id": _rand_id("ACC-"),
                "risk_level": risk,
                "balance": round(random.uniform(5_000, 500_000), 2),
                "avg_tx_amount": round(random.uniform(1_000, 80_000), 2),
                "tx_history_steps": random.randint(50, 400),
                "accessibility_prob": {"low": 0.04, "medium": 0.15, "high": 0.55}[risk],
                "remote_app_prob":    {"low": 0.02, "medium": 0.08, "high": 0.40}[risk],
                "status": "normal",
            }
        )
    return profiles


def _lookup_payee(dest_id: str) -> dict:
    """Return payee profile row or cold-start defaults."""
    if dest_id in PAYEE_PROFILE.index:
        row = PAYEE_PROFILE.loc[dest_id]
        return {
            "dest_cashout_ratio":      float(row.get("dest_cashout_ratio", 0.0)),
            "dest_avg_step_to_cashout": float(row.get("dest_avg_step_to_cashout", 0.0)),
            "dest_tx_count_received":  float(row.get("dest_tx_count_received", 0.0)),
            "dest_is_new_account":     int(row.get("dest_is_new_account", 1)),
        }
    return {
        "dest_cashout_ratio": 0.0,
        "dest_avg_step_to_cashout": 0.0,
        "dest_tx_count_received": 0.0,
        "dest_is_new_account": 1,
    }


def generate_transaction(account: dict, step: int, tx_counter: int) -> dict:
    """
    Simulate one TRANSFER transaction for the given account.
    Returns a feature dict ready for model scoring.
    """
    risk = account["risk_level"]
    avg  = account["avg_tx_amount"]

    # Device signals
    is_acc  = int(random.random() < account["accessibility_prob"])
    is_rem  = int(random.random() < account["remote_app_prob"])
    hesit   = round(np.clip(np.random.lognormal(3.2 if risk == "low" else 4.0, 0.9), 5, 300), 1)

    # Velocity features
    tx_count_per_step   = max(1, int(np.random.poisson(2 if risk == "low" else 4)))
    dest_tx_count_24step = max(0, int(np.random.poisson(1 if risk == "low" else 3)))

    # Amount anomaly
    multiplier = np.random.lognormal(0, 0.4) if risk == "low" else np.random.lognormal(0.8, 0.9)
    amount     = round(avg * multiplier, 2)
    amount_vs_avg_ratio = round(amount / avg, 4)

    # Temporal
    hour = random.randint(0, 23)
    is_odd_hours = int(hour < 6 or hour >= 22)

    # Recipient
    is_first_time = int(random.random() < (0.10 if risk == "low" else 0.45))

    # Pick a payee — sometimes from profile store, sometimes new
    if random.random() < 0.3 and len(PAYEE_PROFILE) > 0:
        dest_id = random.choice(PAYEE_PROFILE.index.tolist())
    else:
        dest_id = _rand_id("C", 6)

    payee = _lookup_payee(dest_id)

    return {
        "tx_id":     f"TX-{tx_counter:05d}",
        "step":      step,
        "account_id": account["account_id"],
        "dest_id":   dest_id,
        "amount":    amount,
        "hour":      hour,
        # Model A features
        "is_accessibility_enabled":  is_acc,
        "is_remote_app_running":     is_rem,
        "hesitation_time_sec":       hesit,
        "tx_count_per_step":         tx_count_per_step,
        "dest_tx_count_24step":      dest_tx_count_24step,
        "amount_vs_avg_ratio":       amount_vs_avg_ratio,
        "is_odd_hours":              is_odd_hours,
        "is_first_time_recipient":   is_first_time,
        # Model B features (payee profile)
        **payee,
    }


def score_transaction(tx: dict) -> dict:
    """
    Run 3-layer scoring pipeline on a single transaction dict.
    Returns tx enriched with scores and decision.
    """
    row_a = pd.DataFrame([{f: tx[f] for f in FEAT_A}])
    row_b = pd.DataFrame([{f: tx[f] for f in FEAT_B}])

    prob_a = float(MODEL_A.predict_proba(row_a)[0, 1])
    prob_b = float(MODEL_B.predict_proba(row_b)[0, 1])

    # Layer 1 — Hard Rules
    rule1 = tx["is_accessibility_enabled"] == 1 and tx["is_remote_app_running"] == 1
    rule2 = tx["hesitation_time_sec"] > 45 and tx["is_first_time_recipient"] == 1

    # Ensemble score
    final_score = W_STAR * prob_a + (1 - W_STAR) * prob_b

    # Rule 2 bias
    if rule2:
        final_score = max(final_score, 0.6)

    # Layer 3 — Threshold decision
    if rule1:
        decision = "Block (Rule 1)"
        layer    = "Layer 1"
    elif final_score >= THR_BLOCK:
        decision = "Block"
        layer    = "Layer 3"
    elif final_score >= THR_ALERT:
        decision = "Alert + OTP"
        layer    = "Layer 3"
    else:
        decision = "Pass"
        layer    = "Layer 3"

    return {
        **tx,
        "prob_a":      round(prob_a, 4),
        "prob_b":      round(prob_b, 4),
        "final_score": round(final_score, 4),
        "rule1":       rule1,
        "rule2":       rule2,
        "decision":    decision,
        "layer":       layer,
    }
