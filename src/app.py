import time
import random
from datetime import datetime
import streamlit as st
from simulator import generate_accounts, generate_transaction, score_transaction

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Analytics — Enterprise Monitor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* App Background & Base Typography */
.stApp { background-color: #f8fafc; }

/* Custom Table Design for Bank Grid */
.bank-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: white;
}
.bank-table th {
    background-color: #f1f5f9;
    color: #475569;
    text-align: left;
    padding: 8px 10px;
    border-bottom: 2px solid #cbd5e1;
    font-weight: 600;
    position: sticky;
    top: 0;
    z-index: 10;
}
.bank-table td {
    padding: 7px 10px;
    border-bottom: 1px solid #e2e8f0;
    color: #334155;
    vertical-align: middle;
}
.bank-table tr:hover { background-color: #f1f5f9; }

/* Decision Styles */
.text-pass { color: #16a34a; font-weight: 600; }
.text-alert { color: #d97706; font-weight: 600; }
.text-block { color: #dc2626; font-weight: 600; }

.status-badge {
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
}

/* GitHub Card Sticked to Sidebar Bottom */
.github-card {
    background: #0f172a;
    color: #ffffff;
    padding: 14px;
    border-radius: 8px;
    margin-top: 50px;
    border: 1px solid #1e293b;
}
.github-card a {
    color: #38bdf8 !important;
    text-decoration: none;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
}
.github-card a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ─── Session state init ───────────────────────────────────────────────────────
def _init():
    st.session_state.setdefault("running",      False)
    st.session_state.setdefault("accounts",     [])
    st.session_state.setdefault("feed",         [])       # สะสมจากบนลงล่าง
    st.session_state.setdefault("alerts",       [])       # สะสมจากบนลงล่าง
    st.session_state.setdefault("step",         1)
    st.session_state.setdefault("tx_counter",   1)
    st.session_state.setdefault("total_tx",     0)
    st.session_state.setdefault("total_block",  0)
    st.session_state.setdefault("total_alert",  0)
    st.session_state.setdefault("total_pass",   0)
    st.session_state.setdefault("amount_risk",  0.0)

_init()

# ─── Sidebar (Configuration & Hidden Controls) ────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    st.caption("Adjust variables for the transaction generator.")
    st.divider()

    speed     = st.slider("Generator Delay (seconds)", 0.3, 3.0, 1.0, 0.1)
    fraud_pct = st.slider("Mule Account Ratio (%)", 5, 40, 12, 1)
    n_accs    = st.slider("Accounts Network Size", 3, 8, 5, 1)

    st.divider()
    st.markdown("**System Actions**")
    reset_btn = st.button("↺ Reset Simulation Data", use_container_width=True)

    if reset_btn:
        for k in ["running","accounts","feed","alerts","step","tx_counter",
                  "total_tx","total_block","total_alert","total_pass","amount_risk"]:
            st.session_state.pop(k, None)
        _init()
        st.rerun()

    # GitHub Link ตรึงไว้ล่างสุดของ Sidebar
    st.markdown("""
    <div class="github-card">
        <p style="margin:0 0 6px 0; font-size:11px; color:#94a3b8;">Want to learn more?</p>
        <a href="https://github.com/PlamGG/Fraud-Detection-Multi-Model-" target="_blank">
            <svg height="18" width="18" viewBox="0 0 16 16" fill="currentColor" style="color:white;"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
            Read Technical Docs
        </a>
    </div>
    """, unsafe_allow_html=True)

# ─── Main Header & Dynamic Bank Controls ──────────────────────────────────────
title_col, ctrl_col = st.columns([2, 1])

with title_col:
    st.markdown("## 🏛️ Fraud Operations — Live Matrix Dashboard")
    status_txt = "🟢 MONITORING ACTIVE" if st.session_state.running else "⏸️ MONITORING PAUSED"
    st.caption(f"{status_txt} · Model Ensemble Framework A × B · Threshold Rules Enabled")

with ctrl_col:
    st.markdown("<div style='padding-top: 12px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    start_btn = c1.button("▶ Start / Resume", use_container_width=True, type="primary", disabled=st.session_state.running)
    pause_btn = c2.button("⏸ Pause", use_container_width=True, disabled=not st.session_state.running)

    if start_btn:
        if not st.session_state.accounts:  
            seed = random.randint(0, 99999)
            st.session_state.accounts = generate_accounts(n=n_accs, seed=seed)
        st.session_state.running = True
        st.rerun()

    if pause_btn:
        st.session_state.running = False
        st.rerun()

st.divider()

# ─── Top Split Grid: Financial Metrics & Accounts Grid ────────────────────────
metric_col, watchlist_col = st.columns([6, 4], gap="medium")

with metric_col:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Processed Transactions", f"{st.session_state.total_tx} tx")
    m2.metric("Critical Blocked", st.session_state.total_block)
    m3.metric("OTP Challenges", st.session_state.total_alert)
    m4.metric("Funds at Risk Exposure", f"฿{st.session_state.amount_risk:,.0f}")

with watchlist_col:
    st.markdown("<p style='font-size:12px; font-weight:600; color:#475569; margin:0 0 6px 0;'>🎯 Account Audit Matrix</p>", unsafe_allow_html=True)
    if st.session_state.accounts:
        acc_table = """<table class="bank-table">
            <tr><th>Account ID</th><th>Baseline Risk</th><th>Operational Status</th></tr>"""
        for acc in st.session_state.accounts:
            icon, status_color = {
                "normal": ("🟢 Normal", "background:#dcfce7;color:#16a34a"),
                "alert": ("🟡 Warning", "background:#fef3c7;color:#d97706"),
                "blocked": ("🔴 Blocked", "background:#fee2e2;color:#dc2626")
            }.get(acc["status"], ("⚪ Unknown", ""))
            
            risk_badge = f"🔥 {acc['risk_level'].upper()}" if acc['risk_level'] == 'high' else acc['risk_level'].upper()
            acc_table += f"""<tr>
                <td><code style='font-size:11px;'>{acc['account_id']}</code></td>
                <td>{risk_badge}</td>
                <td><span class="status-badge" style="{status_color}">{icon}</span></td>
            </tr>"""
        acc_table += "</table>"
        st.markdown(acc_table, unsafe_allow_html=True)
    else:
        st.caption("No accounts mapped. Click 'Start / Resume' to link core systems.")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Bottom Split Grid: Fixed-Height Containers ───────────────────────────────
feed_col, alert_col = st.columns([55, 45], gap="medium")

with feed_col:
    st.markdown("#### ⏱️ Real-Time Transaction Ledger")
    st.caption("Global financial traffic logging (Ordered: $001 \\rightarrow 002 \\rightarrow 003$)")
    feed_box = st.container(height=480) 

with alert_col:
    st.markdown("#### 🚨 Security Incident Alerts Queue")
    st.caption("Decisions requiring compliance review or automated intercept")
    alert_box = st.container(height=480) 

# ─── Function to build table HTML (Append-friendly) ──────────────────────────
def build_feed_table(feed_list) -> str:
    html_table = """<table class="bank-table">
        <tr><th>Time</th><th>Tx ID</th><th>Routing Sequence</th><th>Amount</th><th>Score</th><th>Action</th></tr>"""
    for f_tx in feed_list[-40:]: 
        style_cls = "text-block" if "Block" in f_tx["decision"] else "text-alert" if "Alert" in f_tx["decision"] else "text-pass"
        html_table += f"""<tr>
            <td style='color:#64748b; font-family:monospace;'>{f_tx.get('tx_time', '00:00:00')}</td>
            <td><strong>{f_tx['tx_id']}</strong></td>
            <td><code>{f_tx['account_id']}</code> ➔ <code>{f_tx['dest_id']}</code></td>
            <td><b>฿{f_tx['amount']:,.0f}</b></td>
            <td><code>{f_tx['final_score']:.2f}</code></td>
            <td class="{style_cls}">{f_tx['decision'].upper()}</td>
        </tr>"""
    html_table += "</table>"
    return html_table

# ─── Simulation Core Loop & Rendering Logic ──────────────────────────────────
if st.session_state.running and st.session_state.accounts:
    
    # 📌 สร้าง Placeholder ไว้ "นอกลูป" แค่ครั้งเดียว เพื่อล็อคหัวตารางให้ไม่วาดซ้ำซาก
    with feed_box:
        feed_placeholder = st.empty()
        
    with alert_box:
        alert_placeholder = st.empty()

    while st.session_state.running:
        # Pick account & process transaction
        account = random.choice(st.session_state.accounts)
        raw_tx = generate_transaction(account, step=st.session_state.step, tx_counter=st.session_state.tx_counter)
        tx = score_transaction(raw_tx)
        
        # ฝังเวลาปัจจุบันของระบบธนาคารระดับวินาที
        tx['tx_time'] = datetime.now().strftime("%H:%M:%S")

        # Handle status state mutations
        d = tx["decision"]
        if "Block" in d:   account["status"] = "blocked"
        elif "Alert" in d: account["status"] = "alert"

        # สะสมข้อมูลธุรกรรมต่อท้ายลิสต์เดิมลงไปข้างล่าง
        st.session_state.feed = st.session_state.feed + [tx]
        st.session_state.total_tx += 1
        st.session_state.tx_counter += 1

        if "Block" in d:
            st.session_state.total_block += 1
            st.session_state.alerts = st.session_state.alerts + [tx]
            st.session_state.amount_risk += tx["amount"]
        elif "Alert" in d:
            st.session_state.total_alert += 1
            st.session_state.alerts = st.session_state.alerts + [tx]
            st.session_state.amount_risk += tx["amount"] * 0.5
        else:
            st.session_state.total_pass += 1

        if st.session_state.tx_counter % 6 == 0:
            st.session_state.step += 1

        # 📌 เขียนทับกล่องเดิมเพื่อเพิ่มแถวใหม่ (+1 แถวต่อท้ายในตารางเดี่ยว) หัวตารางจะไม่ซ้ำแล้ว
        feed_placeholder.markdown(build_feed_table(st.session_state.feed), unsafe_allow_html=True)

        # อัปเดตฝั่ง Alerts Queue ในกล่องเดิมเช่นกันแบบเรียบเนียน
        with alert_placeholder.container():
            if st.session_state.alerts:
                for a_tx in st.session_state.alerts[-20:]:
                    title_text = f"🚨 [{a_tx['tx_time']}] {a_tx['tx_id']} | ฿{a_tx['amount']:,.0f} ➔ {a_tx['decision'].upper()}"
                    with st.expander(title_text):
                        st.markdown(f"""
                        **Incident Deep-Dive Metrics:**
                        * **Timestamp:** `{a_tx['tx_time']}`
                        * **Source Account:** `{a_tx['account_id']}`
                        * **Destination Target:** `{a_tx['dest_id']}`
                        * **Model A Probability (Behavior Engine):** `{a_tx['prob_a']:.3f}`
                        * **Model B Probability (Counterparty Network):** `{a_tx['prob_b']:.3f}`
                        * **Weighted Ensemble Score:** `{a_tx['final_score']:.3f}`
                        * **Pipeline Detection Layer:** `{a_tx['layer']}`
                        * **Hard Rules Active:** `Rule 1 (Remote Access): {a_tx['rule1']}` | `Rule 2 (Hesitation Pattern): {a_tx['rule2']}`
                        """)
            else:
                st.success("🟢 OPERATIONAL STATUS: SECURED — No anomalies flagged by models currently.")

        time.sleep(speed)
else:
    # Static render loop เมื่อกด PAUSE (ตารางคงสภาพล่าสุดนิ่งๆ ไว้ให้ตรวจสอบ)
    with feed_box:
        if st.session_state.feed:
            st.markdown(build_feed_table(st.session_state.feed), unsafe_allow_html=True)
        else:
            st.info("System Ready. Click **▶ Start / Resume** to map ledger feeds.")

    with alert_box:
        if st.session_state.alerts:
            for a_tx in st.session_state.alerts[-20:]:
                title_text = f"🚨 [{a_tx['tx_time']}] {a_tx['tx_id']} | ฿{a_tx['amount']:,.0f} ➔ {a_tx['decision'].upper()}"
                with st.expander(title_text):
                    st.markdown(f"""
                    **Incident Deep-Dive Metrics:**
                    * **Timestamp:** `{a_tx['tx_time']}`
                    * **Source Account:** `{a_tx['account_id']}`
                    * **Destination Target:** `{a_tx['dest_id']}`
                    * **Model A Probability:** `{a_tx['prob_a']:.3f}`
                    * **Model B Probability:** `{a_tx['prob_b']:.3f}`
                    * **Weighted Ensemble Score:** `{a_tx['final_score']:.3f}`
                    """)
        else:
            st.success("🟢 OPERATIONAL STATUS: SECURED — No anomalies flagged by models currently.")
