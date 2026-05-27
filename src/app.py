import time
import random
import streamlit as st
from simulator import generate_accounts, generate_transaction, score_transaction

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection — Multi-Model",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Feed card base */
.tx-card {
    padding: 10px 14px;
    border-radius: 8px;
    border: 0.5px solid #e2e8f0;
    margin-bottom: 6px;
    background: #ffffff;
    font-size: 13px;
}
.tx-card.fraud  { border-left: 3px solid #ef4444; background: #fff5f5; }
.tx-card.alert  { border-left: 3px solid #f59e0b; background: #fffbeb; }
.tx-card.rule1  { border-left: 3px solid #8b5cf6; background: #f5f3ff; }
.tx-card.pass   { border-left: 3px solid #22c55e; background: #f0fdf4; }

.badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 100px;
    margin-right: 4px;
}
.badge-block { background:#ef4444; color:#fff; }
.badge-alert { background:#f59e0b; color:#fff; }
.badge-rule1 { background:#8b5cf6; color:#fff; }
.badge-pass  { background:#22c55e; color:#fff; }

.score-row { font-size: 11px; color: #64748b; margin-top: 4px; }
.tx-meta   { font-size: 11px; color: #94a3b8; margin-top: 2px; }

.alert-card {
    padding: 10px 12px;
    border-radius: 8px;
    border: 0.5px solid;
    margin-bottom: 8px;
    font-size: 12px;
}
.alert-card.block { border-color:#ef4444; background:#fff5f5; }
.alert-card.otp   { border-color:#f59e0b; background:#fffbeb; }
.alert-card.rule  { border-color:#8b5cf6; background:#f5f3ff; }

/* KPI strip */
.kpi-val { font-size: 28px; font-weight: 600; }
.kpi-lbl { font-size: 12px; color: #64748b; }
</style>
""", unsafe_allow_html=True)

# ─── Session state init ───────────────────────────────────────────────────────
def _init():
    st.session_state.setdefault("running",      False)
    st.session_state.setdefault("accounts",     [])
    st.session_state.setdefault("feed",         [])       # list of scored tx dicts
    st.session_state.setdefault("alerts",       [])       # fraud/alert only
    st.session_state.setdefault("step",         1)
    st.session_state.setdefault("tx_counter",   1)
    st.session_state.setdefault("total_tx",     0)
    st.session_state.setdefault("total_block",  0)
    st.session_state.setdefault("total_alert",  0)
    st.session_state.setdefault("total_pass",   0)
    st.session_state.setdefault("amount_risk",  0.0)

_init()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Fraud Detection\n**Multi-Model System**")
    st.caption("Sender × Payee ensemble · 3-layer pipeline")
    st.divider()

    speed     = st.slider("Speed (seconds / transaction)", 0.3, 3.0, 1.0, 0.1)
    fraud_pct = st.slider("High-risk account ratio (%)", 5, 40, 12, 1)
    n_accs    = st.slider("Accounts to monitor", 3, 8, 5, 1)

    st.divider()

    col1, col2 = st.columns(2)
    start_btn  = col1.button("▶ Start",  use_container_width=True, type="primary")
    stop_btn   = col2.button("⏹ Stop",   use_container_width=True)
    reset_btn  = st.button("↺ Reset session", use_container_width=True)

    if start_btn:
        seed = random.randint(0, 99999)
        st.session_state.accounts   = generate_accounts(n=n_accs, seed=seed)
        st.session_state.running    = True
        st.session_state.feed       = []
        st.session_state.alerts     = []
        st.session_state.step       = 1
        st.session_state.tx_counter = 1
        st.session_state.total_tx   = 0
        st.session_state.total_block  = 0
        st.session_state.total_alert  = 0
        st.session_state.total_pass   = 0
        st.session_state.amount_risk  = 0.0

    if stop_btn:
        st.session_state.running = False

    if reset_btn:
        for k in ["running","accounts","feed","alerts","step","tx_counter",
                  "total_tx","total_block","total_alert","total_pass","amount_risk"]:
            st.session_state.pop(k, None)
        _init()
        st.rerun()

    st.divider()
    st.markdown("**Thresholds**")
    st.markdown("🔴 Block `≥ 0.80`  \n🟡 Alert+OTP `≥ 0.50`  \n🟢 Pass `< 0.50`")

    st.divider()
    st.markdown("**Accounts**")
    for acc in st.session_state.accounts:
        icon = {"normal": "🟢", "alert": "🟡", "blocked": "🔴"}.get(acc["status"], "⚪")
        risk_tag = {"low": "", "medium": " ⚠️", "high": " 🔥"}.get(acc["risk_level"], "")
        st.caption(f"{icon} `{acc['account_id']}`  {acc['risk_level']}{risk_tag}")

    st.divider()
    st.link_button(
        "GitHub — Fraud-Detection-Multi-Model",
        "https://github.com/PlamGG/Fraud-Detection-Multi-Model-",
        use_container_width=True,
    )

# ─── Main header ─────────────────────────────────────────────────────────────
st.markdown("## Fraud Detection — Multi-Model · Live Monitor")
status_txt = "🟢 Running" if st.session_state.running else "⏸ Paused"
st.caption(f"{status_txt}  ·  Model version 2026-05-24  ·  w\\* = 0.30  ·  Ensemble: Model A × Model B")

# ─── KPI strip ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total transactions", st.session_state.total_tx)
k2.metric("Blocked",  st.session_state.total_block,  delta=None)
k3.metric("Alert+OTP", st.session_state.total_alert, delta=None)
k4.metric("Passed",   st.session_state.total_pass,   delta=None)
k5.metric("Amount at risk (฿)",
          f"{st.session_state.amount_risk:,.0f}")

st.divider()

# ─── Two-column layout ────────────────────────────────────────────────────────
feed_col, alert_col = st.columns([3, 2], gap="medium")

feed_placeholder  = feed_col.empty()
alert_placeholder = alert_col.empty()

# ─── Render helpers ───────────────────────────────────────────────────────────
def _decision_class(tx: dict) -> str:
    d = tx["decision"]
    if "Rule" in d:  return "rule1"
    if "Block" in d: return "fraud"
    if "Alert" in d: return "alert"
    return "pass"

def _badge(tx: dict) -> str:
    d = tx["decision"]
    if "Rule" in d:  return '<span class="badge badge-rule1">Rule 1</span>'
    if "Block" in d: return '<span class="badge badge-block">Block</span>'
    if "Alert" in d: return '<span class="badge badge-alert">Alert+OTP</span>'
    return '<span class="badge badge-pass">Pass</span>'


def render_feed(feed: list):
    html = ""
    for tx in feed[:40]:
        cls   = _decision_class(tx)
        badge = _badge(tx)
        score_color = (
            "#ef4444" if tx["final_score"] >= 0.8 else
            "#f59e0b" if tx["final_score"] >= 0.5 else
            "#22c55e"
        )
        flags = []
        if tx["rule1"]: flags.append("Accessibility+Remote")
        if tx["rule2"]: flags.append("Hesitation+NewRecipient")
        flag_txt = " · ".join(flags) if flags else "No hard rule"

        html += f"""
<div class="tx-card {cls}">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>{badge} <strong>{tx['tx_id']}</strong></div>
    <div style="font-weight:600">฿ {tx['amount']:,.0f}</div>
  </div>
  <div class="tx-meta">{tx['account_id']} → {tx['dest_id']} · step {tx['step']}</div>
  <div class="score-row">
    Model A: <b>{tx['prob_a']:.3f}</b> &nbsp;
    Model B: <b>{tx['prob_b']:.3f}</b> &nbsp;
    Score: <b style="color:{score_color}">{tx['final_score']:.3f}</b> &nbsp;·&nbsp;
    {flag_txt}
  </div>
</div>"""
    return html


def render_alerts(alerts: list):
    html = ""
    for tx in alerts[:25]:
        d = tx["decision"]
        if "Rule" in d:
            cls, label = "rule", "Rule 1 — Hard Block"
            detail = "Accessibility Service + Remote Access Application both active. Blocked before ML scoring."
        elif "Block" in d:
            cls, label = "block", "Block"
            detail = f"Ensemble score {tx['final_score']:.3f} ≥ 0.80. Model A={tx['prob_a']:.3f} · Model B={tx['prob_b']:.3f}."
        else:
            cls, label = "otp", "Alert + OTP"
            detail = f"Score {tx['final_score']:.3f} ≥ 0.50. OTP challenge triggered."

        html += f"""
<div class="alert-card {cls}">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
    <span class="badge badge-{'rule1' if cls=='rule' else cls}">{label}</span>
    <strong style="font-size:12px">{tx['tx_id']} · {tx['account_id']}</strong>
  </div>
  <div style="font-size:11px;color:#374151">{detail}</div>
  <div style="font-size:10px;color:#94a3b8;margin-top:3px">
    ฿ {tx['amount']:,.0f} · step {tx['step']} · {tx['layer']}
  </div>
</div>"""
    return html


# ─── Simulation loop ──────────────────────────────────────────────────────────
if st.session_state.running and st.session_state.accounts:
    while st.session_state.running:
        # Pick a random account
        account = random.choice(st.session_state.accounts)

        # Generate + score
        raw_tx = generate_transaction(
            account,
            step=st.session_state.step,
            tx_counter=st.session_state.tx_counter,
        )
        tx = score_transaction(raw_tx)

        # Update account status
        d = tx["decision"]
        if "Block" in d:
            account["status"] = "blocked"
        elif "Alert" in d:
            account["status"] = "alert"

        # Prepend to feed (newest first)
        st.session_state.feed    = [tx] + st.session_state.feed
        st.session_state.total_tx += 1
        st.session_state.tx_counter += 1

        if "Block" in d:
            st.session_state.total_block += 1
            st.session_state.alerts = [tx] + st.session_state.alerts
            st.session_state.amount_risk += tx["amount"]
        elif "Alert" in d:
            st.session_state.total_alert += 1
            st.session_state.alerts = [tx] + st.session_state.alerts
            st.session_state.amount_risk += tx["amount"] * 0.5
        else:
            st.session_state.total_pass += 1

        # Advance step every 6 transactions
        if st.session_state.tx_counter % 6 == 0:
            st.session_state.step += 1

        # Render
        with feed_placeholder.container():
            st.markdown("#### Transaction feed")
            st.caption("Newest first · showing last 40")
            st.markdown(render_feed(st.session_state.feed), unsafe_allow_html=True)

        with alert_placeholder.container():
            n_alerts = len(st.session_state.alerts)
            st.markdown(f"#### Alerts ({n_alerts})")
            st.caption("Ordered by detection time")
            if st.session_state.alerts:
                st.markdown(render_alerts(st.session_state.alerts), unsafe_allow_html=True)
            else:
                st.info("No alerts yet.")

        time.sleep(speed)

else:
    # Static render when not running
    with feed_placeholder.container():
        st.markdown("#### Transaction feed")
        if st.session_state.feed:
            st.markdown(render_feed(st.session_state.feed), unsafe_allow_html=True)
        else:
            st.info("Press **▶ Start** in the sidebar to begin the simulation.")

    with alert_placeholder.container():
        st.markdown(f"#### Alerts ({len(st.session_state.alerts)})")
        if st.session_state.alerts:
            st.markdown(render_alerts(st.session_state.alerts), unsafe_allow_html=True)
        else:
            st.info("No alerts yet.")
