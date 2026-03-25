import warnings; warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.datasets import load_rossi
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioStat Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme state ─────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"
is_dark = st.session_state["theme"] == "dark"

# ── Design tokens ──────────────────────────────────────────────────────────────
P = dict(
    blue="#2563EB", blue_lt="rgba(37,99,235,0.12)",
    red="#DC2626",  red_lt="rgba(220,38,38,0.12)",
    teal="#0D9488", teal_lt="rgba(13,148,136,0.12)",
    amber="#D97706",amber_lt="rgba(217,119,6,0.12)",
    slate="#475569", bg="#F8FAFC", dark="#0F172A",
)

LEGEND_BASE = dict(
    bgcolor="rgba(30,41,59,0.9)" if is_dark else "rgba(255,255,255,0.8)",
    bordercolor="#334155"        if is_dark else "#E2E8F0",
    borderwidth=1, font_size=11,
)
CT = dict(  # chart theme — no 'legend' key; pass LEGEND_BASE per chart
    template="plotly_dark" if is_dark else "plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=12,
              color="#E2E8F0" if is_dark else P["dark"]),
    title_font=dict(size=14, family="Inter, system-ui, sans-serif",
                    color="#F1F5F9" if is_dark else P["dark"]),
    paper_bgcolor="#1E293B" if is_dark else "white",
    plot_bgcolor="#0F172A"  if is_dark else "#FAFBFC",
    margin=dict(l=55, r=25, t=55, b=55),
)

# ── CSS ────────────────────────────────────────────────────────────────────────
_dark_css = """
  /* ─ Dark mode overrides ─ */
  .stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
  .main,.block-container{background-color:#0f172a !important}
  .stMarkdown p,.stMarkdown li{color:#cbd5e1 !important}
  h1,h2,h3,h4{color:#f1f5f9 !important}
  .ds-card{background:#1e293b !important;border-color:#334155 !important;
    border-left-color:#3b82f6 !important}
  .ds-card h3{color:#93c5fd !important}
  .ds-card p,.ds-card .meta{color:#cbd5e1 !important}
  .ds-card .meta b{color:#60a5fa !important}
  .tag{background:#1e3a5f !important;color:#93c5fd !important}
  .kpi{background:#1e293b !important;border-color:#334155 !important}
  .kpi-val{color:#f1f5f9 !important}
  .kpi-lbl{color:#94a3b8 !important}
  .ibox{background:#052e16 !important;border-color:#166534 !important;
    color:#d1fae5 !important}
  .ibox b{color:#6ee7b7 !important}
  .wbox{background:#451a03 !important;border-color:#92400e !important;
    color:#fef3c7 !important}
  .sh{color:#e2e8f0 !important;border-bottom-color:#334155 !important}
  .stTabs [data-baseweb="tab-list"]{background:#1e293b !important}
  .stTabs [data-baseweb="tab"]{color:#94a3b8 !important}
  .stTabs [aria-selected="true"]{background:#0f172a !important;
    color:#60a5fa !important}
  .stDataFrame thead tr th{background:#1e293b !important;color:#e2e8f0 !important}
  .stDataFrame tbody tr td{color:#cbd5e1 !important;background:#0f172a !important}
  code{background:#1e293b !important;color:#93c5fd !important}
""" if is_dark else ""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Inter',system-ui,sans-serif}}

/* ─ Remove top empty space ─ */
header[data-testid="stHeader"]{{height:0 !important;min-height:0 !important;
  overflow:hidden;padding:0 !important}}
[data-testid="stToolbar"]{{display:none}}
div[data-testid="stDecoration"]{{display:none}}
.block-container{{padding-top:0.75rem !important}}

/* ─ App header ─ */
.app-hdr{{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 55%,#2563eb 100%);
  padding:1.2rem 2rem;border-radius:16px;margin-bottom:1rem;
  display:flex;align-items:center;gap:1rem}}
.app-hdr-text h1{{color:#fff;font-size:1.8rem;margin:0;font-weight:700;letter-spacing:-0.02em}}
.app-hdr-text p{{color:#93c5fd;margin:.2rem 0 0;font-size:.88rem}}

/* ─ Dataset card ─ */
.ds-card{{background:#f0f9ff;border:1px solid #bae6fd;border-left:5px solid #0284c7;
  border-radius:12px;padding:1.1rem 1.4rem;margin-bottom:1rem}}
.ds-card h3{{color:#0369a1;font-size:1rem;margin:0 0 .55rem;font-weight:700}}
.ds-card p{{color:#374151;font-size:.87rem;line-height:1.75;margin:0 0 .7rem}}
.ds-card .meta{{display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:.8rem;
  font-size:.8rem;color:#475569}}
.ds-card .meta b{{color:#0369a1}}
.tag{{background:#dbeafe;color:#1d4ed8;padding:.12rem .55rem;border-radius:999px;
  font-size:.72rem;font-weight:600;display:inline-block;margin:.1rem .15rem}}

/* ─ KPI strip ─ */
.kpi-row{{display:flex;gap:10px;flex-wrap:wrap;margin:1rem 0}}
.kpi{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;
  padding:14px 16px;flex:1;min-width:130px;
  box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.kpi-lbl{{font-size:.68rem;color:#6b7280;font-weight:700;text-transform:uppercase;
  letter-spacing:.07em}}
.kpi-val{{font-size:1.5rem;font-weight:700;color:#111827;margin-top:3px}}
.kpi-sub{{font-size:.76rem;margin-top:2px}}
.g{{color:#059669}}.r{{color:#dc2626}}.s{{color:#6b7280}}

/* ─ Interpretation / info boxes ─ */
.ibox{{background:#f0fdf4;border:1px solid #86efac;border-left:5px solid #16a34a;
  border-radius:10px;padding:1rem 1.4rem;margin:1rem 0;
  font-size:.87rem;color:#14532d;line-height:1.75}}
.ibox b{{color:#166534}}
.wbox{{background:#fffbeb;border:1px solid #fcd34d;border-left:5px solid #d97706;
  border-radius:10px;padding:.9rem 1.3rem;margin:.8rem 0;
  font-size:.85rem;color:#78350f;line-height:1.7}}

/* ─ Section headings ─ */
.sh{{font-size:.97rem;font-weight:700;color:#1f2937;
  border-bottom:2px solid #e5e7eb;padding-bottom:.45rem;margin:1.8rem 0 1rem}}

/* ─ Sidebar ─ */
section[data-testid="stSidebar"]>div:first-child{{background:#0f172a}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li{{color:#cbd5e1 !important;font-size:.83rem}}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{{color:#f1f5f9 !important}}
section[data-testid="stSidebar"] hr{{border-color:#1e293b}}
section[data-testid="stSidebar"] .stButton button{{
  background:#1e293b;color:#e2e8f0;border:1px solid #334155;
  border-radius:8px;font-size:.82rem;width:100%;padding:.45rem}}
section[data-testid="stSidebar"] .stButton button:hover{{
  background:#334155;border-color:#475569}}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"]{{background:#f1f5f9;border-radius:12px;padding:4px;gap:4px}}
.stTabs [data-baseweb="tab"]{{border-radius:8px;font-weight:600;font-size:.88rem;
  padding:9px 16px;color:#475569}}
.stTabs [aria-selected="true"]{{background:#fff !important;color:#1e40af !important;
  box-shadow:0 1px 4px rgba(0,0,0,.1)}}

/* ─ Tables ─ */
.stDataFrame thead tr th{{background:#f8fafc !important;font-weight:700 !important;
  color:#374151 !important;font-size:.82rem}}
.stDataFrame tbody tr td{{font-size:.82rem}}

{_dark_css}
</style>
""", unsafe_allow_html=True)

# ── App header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-hdr">
  <span style="font-size:2.4rem;line-height:1">🔬</span>
  <div class="app-hdr-text">
    <h1>BioStat Lab</h1>
    <p>Interactive biostatistical modeling suite &nbsp;·&nbsp; Real-world datasets &nbsp;·&nbsp; All computation server-side</p>
  </div>
</div>""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="padding:.6rem 0 .4rem">
      <h2 style="color:#f1f5f9;margin:0;font-size:1.1rem;font-weight:700">🔬 BioStat Lab</h2>
      <p style="color:#64748b;font-size:.73rem;margin:.15rem 0 0">v2.0 — Real Datasets</p>
    </div>""", unsafe_allow_html=True)

    # Theme toggle
    _theme_lbl = "☀️ Light Mode" if is_dark else "🌙 Dark Mode"
    if st.button(_theme_lbl, key="theme_btn", use_container_width=True):
        st.session_state["theme"] = "light" if is_dark else "dark"
        st.rerun()

    st.markdown("---")
    st.markdown("### 📁 Datasets")
    st.markdown("""
- **⚕️ Survival** — Rossi Recidivism (criminology)
- **📊 Mixed LM** — Sleep Deprivation (neuroscience)
- **🎯 Causal** — UCI Diabetes · Beer Tax (econ.)
- **🎲 Bayesian** — UCI Diabetes (biostatistics)
""")
    st.markdown("### 🧮 Methods")
    st.markdown("""
- Kaplan-Meier estimator
- Cox Proportional Hazards
- Linear Mixed Effects (MixedLM)
- Propensity Score Matching (1:1 NN)
- Difference-in-Differences (DiD)
- Conjugate Bayesian Regression
""")
    st.markdown("---")
    st.markdown("""<div style="color:#475569;font-size:.73rem;line-height:1.7">
Built with&nbsp; <b style="color:#94a3b8">Streamlit</b> ·
<b style="color:#94a3b8">lifelines</b> · <b style="color:#94a3b8">statsmodels</b><br>
<b style="color:#94a3b8">scikit-learn</b> · <b style="color:#94a3b8">Plotly</b>
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚕️  Survival Analysis",
    "📊  Linear Mixed Effects",
    "🎯  Causal Inference",
    "🎲  Bayesian Inference",
])

# ─────────────────────────────────────────────────────────────────────────────
# SHARED DATA — loaded once, cached
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets…")
def load_all_data():
    # Rossi
    rossi = load_rossi()

    # Sleep deprivation study (Belenky et al., 2003) via lme4
    try:
        sleep = sm.datasets.get_rdataset("sleepstudy", "lme4").data
        sleep_ok = True
    except Exception:
        # Synthetic fallback that preserves structure
        rng2 = np.random.default_rng(0)
        n_s, n_d = 18, 10
        subj_id = np.repeat(np.arange(308, 308 + n_s), n_d)
        days = np.tile(np.arange(n_d), n_s)
        ri = rng2.normal(0, 25, n_s)
        rs = rng2.normal(0, 4, n_s)
        react = 250 + 10.5 * days + ri[subj_id - 308] + rs[subj_id - 308] * days + rng2.normal(0, 25, n_s * n_d)
        sleep = pd.DataFrame({"Reaction": react, "Days": days, "Subject": subj_id.astype(str)})
        sleep_ok = False

    # Diabetes
    d = load_diabetes(as_frame=True)
    diab = d.frame.copy()

    # Fatalities (Card & Krueger DiD)
    try:
        fat = sm.datasets.get_rdataset("Fatalities", "AER").data
        fat["fatal_rate"] = fat["fatal"] / fat["pop"] * 10_000
        st_tax = fat.groupby("state")["beertax"].mean()
        fat["high_tax"] = fat["state"].map(lambda s: int(st_tax[s] > st_tax.median()))
        fat["post"] = (fat["year"].astype(int) >= 1986).astype(int)
        fat["interaction"] = fat["high_tax"] * fat["post"]
        fat_ok = True
    except Exception:
        fat, fat_ok = None, False

    return rossi, sleep, sleep_ok, diab, fat, fat_ok

rossi, sleep_df, sleep_ok, diab_df, fat_df, fat_ok = load_all_data()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SURVIVAL ANALYSIS (Rossi Recidivism)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""<div class="ds-card">
<h3>📂 Rossi Recidivism Dataset &nbsp;<span class="tag">Criminology</span><span class="tag">Right-censored</span><span class="tag">Observational</span></h3>
<p>A landmark study following <b>432 male prisoners</b> after release from Maryland correctional
facilities (Rossi, Berk &amp; Lenihan, 1980). Each subject was observed weekly for up to
<b>52 weeks</b>; the event of interest is <em>re-arrest</em>. The central question is whether
<b>financial aid</b> at release reduces the hazard of re-offending — a classic intent-to-treat
survival analysis.</p>
<div class="meta">
  <span><b>N</b> = 432 subjects</span>
  <span><b>Follow-up</b> = up to 52 weeks</span>
  <span><b>Events</b> = 114 arrests (26.4 %)</span>
  <span><b>Source</b> = <em>lifelines</em> Python package</span>
</div>
<b>Variables:</b>&nbsp; <code>week</code> (time to arrest / censoring) · <code>arrest</code> (event) ·
<code>fin</code> (financial aid) · <code>age</code> · <code>race</code> · <code>wexp</code>
(work experience) · <code>mar</code> (married) · <code>paro</code> (parole) ·
<code>prio</code> (prior convictions)
</div>""", unsafe_allow_html=True)

    ctrl1, main1 = st.columns([1, 3])

    with ctrl1:
        st.markdown("#### ⚙️ Controls")
        strat_var = st.selectbox(
            "Stratify KM by:",
            ["fin", "race", "mar", "paro", "wexp"],
            format_func={"fin": "Financial Aid", "race": "Race",
                         "mar": "Marital Status", "paro": "Parole Status",
                         "wexp": "Work Experience"}.__getitem__,
            key="t1_strat",
        )
        strat_map = {
            "fin":  {0: "No Aid",        1: "Financial Aid"},
            "race": {0: "White",          1: "Black"},
            "mar":  {0: "Unmarried",      1: "Married"},
            "paro": {0: "No Parole",      1: "On Parole"},
            "wexp": {0: "No Work Exp.",   1: "Work Experience"},
        }
        cox_vars = st.multiselect(
            "Cox PH covariates:",
            ["fin", "age", "race", "wexp", "mar", "paro", "prio"],
            default=["fin", "age", "race", "wexp", "mar", "paro", "prio"],
            key="t1_cox",
        )
        with st.expander("Variable definitions"):
            st.markdown("""
| Variable | Meaning |
|---|---|
| `fin` | Financial aid at release |
| `age` | Age (years) |
| `race` | 1 = Black |
| `wexp` | Full-time work pre-incarceration |
| `mar` | Married at release |
| `paro` | Released on parole |
| `prio` | # prior convictions |
""")

    with main1:
        # ── KM figure ────────────────────────────────────────────────────────
        fig_km = go.Figure()
        LINE_C = [P["blue"], P["red"]]
        FILL_C = [P["blue_lt"], P["red_lt"]]

        for i, (val, label) in enumerate(sorted(strat_map[strat_var].items())):
            sub = rossi[rossi[strat_var] == val]
            kmf = KaplanMeierFitter()
            kmf.fit(sub["week"], sub["arrest"], label=label)
            tl = kmf.timeline
            sf = kmf.survival_function_[label].values
            cl = kmf.confidence_interval_.iloc[:, 0].values
            cu = kmf.confidence_interval_.iloc[:, 1].values

            fig_km.add_trace(go.Scatter(
                x=np.concatenate([tl, tl[::-1]]),
                y=np.concatenate([cu, cl[::-1]]),
                fill="toself", fillcolor=FILL_C[i],
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=False,
            ))
            fig_km.add_trace(go.Scatter(
                x=tl, y=sf, mode="lines",
                line=dict(color=LINE_C[i], width=2.5, shape="hv"),
                name=f"{label}  (n = {len(sub)})",
            ))

        fig_km.update_layout(
            title=f"Kaplan-Meier Survival Curves — Stratified by <b>{strat_var.upper()}</b>",
            xaxis_title="Weeks since prison release",
            yaxis_title="P(not re-arrested)",
            yaxis=dict(range=[0, 1.04], tickformat=".0%"),
            legend=dict(**LEGEND_BASE, orientation="h", yanchor="top", y=-0.2),
            **CT, height=390,
        )
        st.plotly_chart(fig_km, use_container_width=True)

        # ── Log-rank + KPIs ───────────────────────────────────────────────
        g0_df = rossi[rossi[strat_var] == 0]
        g1_df = rossi[rossi[strat_var] == 1]
        lr = logrank_test(g0_df["week"], g1_df["week"],
                          event_observed_A=g0_df["arrest"],
                          event_observed_B=g1_df["arrest"])
        km0 = KaplanMeierFitter().fit(g0_df["week"], g0_df["arrest"])
        km1 = KaplanMeierFitter().fit(g1_df["week"], g1_df["arrest"])

        labels = strat_map[strat_var]
        p_sig  = lr.p_value < 0.05
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Log-rank p-value",
                  f"{lr.p_value:.4f}",
                  delta="Significant ★" if p_sig else "Not significant")
        c2.metric(f"Median survival — {labels[0]}",
                  f"{km0.median_survival_time_:.0f} wks",
                  delta=f"Arrest rate: {g0_df['arrest'].mean():.1%}")
        c3.metric(f"Median survival — {labels[1]}",
                  f"{km1.median_survival_time_:.0f} wks",
                  delta=f"Arrest rate: {g1_df['arrest'].mean():.1%}")
        c4.metric("Total events",
                  f"{rossi['arrest'].sum()}",
                  delta=f"of {len(rossi)} subjects ({rossi['arrest'].mean():.1%})")

        # ── Cox PH ────────────────────────────────────────────────────────
        st.markdown('<p class="sh">Cox Proportional Hazards Model</p>', unsafe_allow_html=True)
        if cox_vars:
            cph = CoxPHFitter()
            cph.fit(rossi[["week", "arrest"] + cox_vars],
                    duration_col="week", event_col="arrest")
            s = cph.summary

            def sig_stars(p):
                if p < 0.001: return " ★★★"
                if p < 0.01:  return " ★★"
                if p < 0.05:  return " ★"
                return ""

            cox_tbl = pd.DataFrame({
                "Hazard Ratio":    s["exp(coef)"].round(3),
                "95 % CI":         s.apply(
                    lambda r: f"[{r['exp(coef) lower 95%']:.3f}, {r['exp(coef) upper 95%']:.3f}]",
                    axis=1),
                "log HR":          s["coef"].round(3),
                "z-stat":          s["z"].round(3),
                "p-value":         s["p"].apply(lambda v: f"{v:.4f}{sig_stars(v)}"),
            })
            st.dataframe(cox_tbl, use_container_width=True)
            st.caption(f"C-statistic (concordance): **{cph.concordance_index_:.3f}** "
                       "· ★★★ p < 0.001  ★★ p < 0.01  ★ p < 0.05")

            if "fin" in cox_vars:
                fin_hr = s.loc["fin", "exp(coef)"]
                fin_p  = s.loc["fin", "p"]
                prio_hr = s.loc["prio", "exp(coef)"] if "prio" in cox_vars else None
                prio_txt = (f" Each additional prior conviction multiplies the "
                            f"re-arrest hazard by <b>{prio_hr:.2f}×</b>."
                            if prio_hr else "")
                st.markdown(f"""<div class="ibox">
<b>Key finding:</b> Financial aid is associated with a
<b>{"%.1f" % (abs(1 - fin_hr) * 100)} % {"lower" if fin_hr < 1 else "higher"}
hazard of re-arrest</b> (HR = {fin_hr:.3f}, p = {fin_p:.4f}).
{"This is statistically significant (α = 0.05), supporting the hypothesis that economic support post-release reduces recidivism." if fin_p < 0.05 else "This does not reach statistical significance at α = 0.05."}{prio_txt}
</div>""", unsafe_allow_html=True)
        else:
            st.warning("Select at least one covariate for the Cox model.")

        with st.expander("Show raw dataset (first 20 rows)"):
            st.dataframe(rossi.head(20), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LINEAR MIXED EFFECTS (Sleep Deprivation)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not sleep_ok:
        st.warning("⚠️ Could not download sleepstudy from R datasets. "
                   "Showing synthetic fallback data with the same structure.")

    st.markdown("""<div class="ds-card">
<h3>📂 Sleep Deprivation Study &nbsp;<span class="tag">Neuroscience</span><span class="tag">Longitudinal</span><span class="tag">Balanced panel</span></h3>
<p>Belenky et al. (2003) measured psychomotor vigilance task (PVT) reaction times in
<b>18 subjects</b> over <b>10 consecutive days</b> of sleep restriction (3 h/night).
This is the canonical reference dataset for Linear Mixed Effects models in the lme4
documentation (Bates et al., 2015). Random intercepts capture the baseline speed
differences across subjects; random slopes capture individual trajectories of
cognitive decline.</p>
<div class="meta">
  <span><b>N</b> = 180 obs (18 subjects × 10 days)</span>
  <span><b>Outcome</b> = Reaction time (ms)</span>
  <span><b>Source</b> = Belenky et al. (2003) · <em>lme4</em> R package</span>
</div>
<b>Variables:</b>&nbsp; <code>Reaction</code> (ms) · <code>Days</code> (0–9 days of deprivation) ·
<code>Subject</code> (participant ID)
</div>""", unsafe_allow_html=True)

    ctrl2, main2 = st.columns([1, 3])

    with ctrl2:
        st.markdown("#### ⚙️ Controls")
        n_show_subj = st.slider("Subjects to show in trajectory plot:",
                                3, 18, 10, 1, key="t2_ns")
        show_random_slopes = st.checkbox("Add random slopes (Days|Subject)",
                                         value=False, key="t2_rs")
        st.info("Random intercepts model how fast each subject is at baseline. "
                "Random slopes capture how differently subjects respond to sleep loss.")

    with main2:
        # Fit LME
        @st.cache_data
        def fit_lme(use_random_slopes, _sleep_df):
            if use_random_slopes:
                try:
                    mdl = smf.mixedlm("Reaction ~ Days", _sleep_df,
                                      groups=_sleep_df["Subject"],
                                      re_formula="~Days")
                    res = mdl.fit(reml=True, method="lbfgs")
                    return res
                except Exception:
                    pass
            mdl = smf.mixedlm("Reaction ~ Days", _sleep_df,
                               groups=_sleep_df["Subject"])
            return mdl.fit(reml=True, method="lbfgs")

        result = fit_lme(show_random_slopes, sleep_df)

        # Fixed effects table
        st.markdown('<p class="sh">Fixed Effects</p>', unsafe_allow_html=True)
        fe_names = list(result.fe_params.index)
        fe_coef  = result.fe_params.values
        try:
            fe_se = result.bse[fe_names].values
        except Exception:
            fe_se = result.bse.iloc[:len(fe_names)].values
        fe_z = fe_coef / np.where(fe_se > 1e-9, fe_se, np.nan)
        fe_p = 2 * stats.norm.sf(np.abs(fe_z))

        fe_tbl = pd.DataFrame({
            "Estimate":    fe_coef.round(3),
            "Std Error":   fe_se.round(3),
            "z":           fe_z.round(3),
            "p-value":     fe_p.round(4),
            "95 % CI":     [f"[{fe_coef[i]-1.96*fe_se[i]:.2f}, "
                             f"{fe_coef[i]+1.96*fe_se[i]:.2f}]"
                             for i in range(len(fe_coef))],
        }, index=fe_names)
        st.dataframe(fe_tbl, use_container_width=True)

        # Variance components + ICC
        re_var_est  = float(result.cov_re.iloc[0, 0])
        res_var_est = float(result.scale)
        icc = re_var_est / (re_var_est + res_var_est)

        v1, v2, v3 = st.columns(3)
        v1.metric("Random Intercept Variance", f"{re_var_est:.2f} ms²")
        v2.metric("Residual Variance",          f"{res_var_est:.2f} ms²")
        v3.metric("ICC (Intraclass Corr.)",     f"{icc:.3f}",
                  help="Proportion of total variance attributable to between-subject differences")

        st.markdown(f"""<div class="ibox">
<b>Interpretation:</b> Each additional day of sleep deprivation adds approximately
<b>{result.fe_params["Days"]:.1f} ms</b> to reaction time (p = {fe_p[1]:.4f}).
The ICC of <b>{icc:.2f}</b> means {icc*100:.0f} % of total variability in reaction
times is due to stable between-subject differences — subjects differ substantially
in baseline speed, independent of the sleep manipulation.
</div>""", unsafe_allow_html=True)

        # ── Observed vs Fitted scatter ────────────────────────────────────
        st.markdown('<p class="sh">Model Diagnostics</p>', unsafe_allow_html=True)
        fitted = result.fittedvalues
        lo, hi = sleep_df["Reaction"].min(), sleep_df["Reaction"].max()

        d1, d2 = st.columns(2)

        with d1:
            fig_ov = go.Figure()
            fig_ov.add_trace(go.Scatter(
                x=sleep_df["Reaction"], y=fitted, mode="markers",
                marker=dict(color=P["blue"], opacity=0.5, size=6,
                            line=dict(color="white", width=0.5)),
                name="Observations",
                hovertemplate="Observed: %{x:.1f} ms<br>Fitted: %{y:.1f} ms<extra></extra>",
            ))
            fig_ov.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines",
                line=dict(color="#94A3B8", dash="dash", width=1.5),
                name="Perfect fit",
            ))
            fig_ov.update_layout(
                title="Observed vs Fitted Values",
                xaxis_title="Observed (ms)", yaxis_title="Fitted (ms)",
                legend=LEGEND_BASE,
                **CT, height=330,
            )
            st.plotly_chart(fig_ov, use_container_width=True)

        with d2:
            # Random effects histogram
            re_vals = [float(result.random_effects[s].iloc[0])
                       for s in sleep_df["Subject"].unique()]
            fig_re = go.Figure()
            fig_re.add_trace(go.Histogram(
                x=re_vals, nbinsx=10,
                marker_color=P["teal"], opacity=0.75,
                name="Random intercepts",
            ))
            # Overlay Normal
            mu_re, sd_re = np.mean(re_vals), np.std(re_vals)
            xx = np.linspace(min(re_vals)-10, max(re_vals)+10, 200)
            yy = stats.norm.pdf(xx, mu_re, sd_re) * len(re_vals) * (max(re_vals)-min(re_vals))/10
            fig_re.add_trace(go.Scatter(
                x=xx, y=yy, mode="lines",
                line=dict(color=P["amber"], width=2, dash="dot"),
                name="Normal fit",
            ))
            fig_re.update_layout(
                title="Distribution of Subject Random Intercepts",
                xaxis_title="Random Intercept (ms offset)",
                yaxis_title="Count",
                legend=LEGEND_BASE,
                **CT, height=330,
            )
            st.plotly_chart(fig_re, use_container_width=True)

        # ── Subject trajectories ──────────────────────────────────────────
        st.markdown('<p class="sh">Subject-Level Trajectories</p>', unsafe_allow_html=True)
        fig_tj = go.Figure()
        days_arr = np.arange(sleep_df["Days"].max() + 1)
        fe = result.fe_params

        subjects_sample = list(sleep_df["Subject"].unique())[:n_show_subj]
        cmap = [f"hsl({int(h)},65%,50%)"
                for h in np.linspace(220, 10, n_show_subj)]

        for k, sid in enumerate(subjects_sample):
            try:
                re_val = float(result.random_effects[sid].iloc[0])
            except Exception:
                re_val = 0.0
            y_ind = fe["Intercept"] + fe["Days"] * days_arr + re_val
            # Actual data
            sub_data = sleep_df[sleep_df["Subject"] == sid]
            fig_tj.add_trace(go.Scatter(
                x=sub_data["Days"], y=sub_data["Reaction"],
                mode="markers", marker=dict(color=cmap[k], size=6, opacity=0.6),
                showlegend=False, hovertemplate=f"Subject {sid}<br>Day %{{x}}: %{{y:.0f}} ms<extra></extra>",
            ))
            fig_tj.add_trace(go.Scatter(
                x=days_arr, y=y_ind, mode="lines",
                line=dict(color=cmap[k], width=1.5),
                name=f"Subject {sid}", opacity=0.7,
            ))

        # Population average
        y_pop = fe["Intercept"] + fe["Days"] * days_arr
        fig_tj.add_trace(go.Scatter(
            x=days_arr, y=y_pop, mode="lines",
            line=dict(color=P["dark"], width=3, dash="solid"),
            name="Population Average",
        ))

        fig_tj.update_layout(
            title=f"Individual Reaction Time Trajectories ({n_show_subj} subjects shown)",
            xaxis_title="Days of sleep deprivation",
            yaxis_title="Reaction time (ms)",
            legend=LEGEND_BASE,
            **CT, height=400,
        )
        st.plotly_chart(fig_tj, use_container_width=True)

        with st.expander("Show raw dataset"):
            st.dataframe(sleep_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CAUSAL INFERENCE (UCI Diabetes + Fatalities DiD)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""<div class="ds-card">
<h3>📂 Two datasets are used in this tab</h3>
<p><b>Propensity Score Matching</b> — UCI Diabetes Dataset (Efron et al., 2004):
442 patients with baseline clinical measurements; all features are standardised.
We define a binary treatment as <em>above-median BMI</em> and ask whether higher BMI
<em>causally</em> elevates disease progression at one year after controlling for
age, sex, blood pressure, and glucose via propensity score matching.</p>
<p><b>Difference-in-Differences</b> — US Traffic Fatalities (Ruhm, 1996 / Stock &amp; Watson):
48 US states × 7 years (1982–1988). We test whether states with <em>higher beer taxes</em>
experienced a greater reduction in traffic fatalities, the canonical DiD illustration from
<em>Introduction to Econometrics</em>.</p>
<div class="meta">
  <span><b>PSM N</b> = 442 patients</span>
  <span><b>DiD N</b> = 336 state-years</span>
  <span><b>Sources</b> = sklearn · AER R package (Kleiber &amp; Zeileis)</span>
</div>
</div>""", unsafe_allow_html=True)

    ctrl3, main3 = st.columns([1, 3])

    with ctrl3:
        st.markdown("#### ⚙️ Controls")
        bmi_pct = st.slider("Treatment threshold (BMI percentile):",
                            40, 70, 50, 5, key="t3_pct")
        confounders = st.multiselect(
            "Propensity score covariates:",
            ["age", "sex", "bp", "s1", "s6"],
            default=["age", "sex", "bp", "s6"],
            key="t3_conf",
        )
        feat_labels = {"age": "Age", "sex": "Sex", "bmi": "BMI",
                       "bp": "Blood Pressure", "s1": "Total Cholesterol",
                       "s2": "LDL", "s3": "HDL", "s4": "Chol/HDL ratio",
                       "s5": "log(Triglycerides)", "s6": "Glucose"}
        st.info("All features are normalised (mean 0). "
                "Treatment = BMI above selected percentile.")
        with st.expander("Feature legend"):
            for k, v in feat_labels.items():
                st.markdown(f"`{k}` → {v}")

    # ── PSM ──────────────────────────────────────────────────────────────────
    @st.cache_data
    def run_psm(pct_threshold, confounder_list, _df):
        df = _df.copy()
        thr = np.percentile(df["bmi"], pct_threshold)
        df["treatment"] = (df["bmi"] >= thr).astype(int)

        if not confounder_list:
            confounder_list = ["age", "bp"]

        X = df[confounder_list].values
        y = df["treatment"].values
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X, y)
        df["ps"] = lr.predict_proba(X)[:, 1]

        t_idx = np.where(df["treatment"] == 1)[0]
        c_idx = np.where(df["treatment"] == 0)[0]
        t_ps  = df["ps"].values[t_idx]
        c_ps  = df["ps"].values[c_idx]

        used, pairs = set(), []
        for ti, tp in sorted(zip(t_idx, t_ps), key=lambda x: x[1]):
            bd, bc = float("inf"), None
            for j, ci in enumerate(c_idx):
                if ci not in used:
                    d = abs(tp - c_ps[j])
                    if d < bd:
                        bd, bc = d, ci
            if bc is not None:
                pairs.append((ti, bc))
                used.add(bc)

        all_m = [p[0] for p in pairs] + [p[1] for p in pairs]
        return df, df.iloc[all_m].copy()

    if confounders:
        df_psm, df_matched = run_psm(bmi_pct, confounders, diab_df)
    else:
        st.warning("Select at least one propensity score covariate.")
        df_psm, df_matched = run_psm(bmi_pct, ["age", "bp"], diab_df)

    with main3:
        st.markdown('<p class="sh">Part 1 — Propensity Score Matching</p>',
                    unsafe_allow_html=True)

        # ── PS Distribution ───────────────────────────────────────────────
        fig_ps = make_subplots(rows=1, cols=2,
                               subplot_titles=["Before Matching", "After Matching"],
                               horizontal_spacing=0.1)
        for is_t, label, color in [(0, "Control", P["blue"]),
                                    (1, "Treated",  P["red"])]:
            for ci, dfsrc in [(1, df_psm), (2, df_matched)]:
                sub_ps = dfsrc[dfsrc["treatment"] == is_t]["ps"]
                fig_ps.add_trace(
                    go.Histogram(x=sub_ps, name=label, nbinsx=25,
                                 marker_color=color, opacity=0.6,
                                 legendgroup=label, showlegend=(ci == 1)),
                    row=1, col=ci,
                )
        fig_ps.update_layout(
            barmode="overlay",
            title="Propensity Score Distributions — Before and After Matching",
            legend=dict(**LEGEND_BASE, orientation="h", yanchor="top", y=-0.2),
            **CT, height=320,
        )
        st.plotly_chart(fig_ps, use_container_width=True)

        # ── Love plot (SMD) ───────────────────────────────────────────────
        st.markdown('<p class="sh">Covariate Balance — Standardised Mean Differences</p>',
                    unsafe_allow_html=True)
        all_covs = [c for c in confounders + ["bmi"] if c != "treatment"]
        all_covs = list(dict.fromkeys(all_covs))  # deduplicate

        def smd(df_src, cov):
            t = df_src[df_src["treatment"] == 1][cov]
            c = df_src[df_src["treatment"] == 0][cov]
            sp = np.sqrt((t.std()**2 + c.std()**2) / 2)
            return (t.mean() - c.mean()) / (sp if sp > 0 else 1)

        smd_before = [smd(df_psm, c) for c in all_covs]
        smd_after  = [smd(df_matched, c) for c in all_covs]

        fig_love = go.Figure()
        fig_love.add_vline(x=0,  line_dash="solid", line_color="#E5E7EB", line_width=1)
        fig_love.add_vline(x=0.1,  line_dash="dot", line_color="#FCA5A5", line_width=1)
        fig_love.add_vline(x=-0.1, line_dash="dot", line_color="#FCA5A5", line_width=1)
        fig_love.add_trace(go.Scatter(
            x=smd_before, y=all_covs, mode="markers",
            marker=dict(color=P["red"], size=10, symbol="circle-open", line_width=2),
            name="Before matching",
        ))
        fig_love.add_trace(go.Scatter(
            x=smd_after, y=all_covs, mode="markers",
            marker=dict(color=P["blue"], size=10),
            name="After matching",
        ))
        for i, cov in enumerate(all_covs):
            fig_love.add_shape(type="line",
                x0=smd_before[i], x1=smd_after[i], y0=cov, y1=cov,
                line=dict(color="#CBD5E1", width=1))
        fig_love.update_layout(
            title="Love Plot — |SMD| < 0.10 indicates good balance",
            xaxis_title="Standardised Mean Difference (SMD)",
            xaxis=dict(zeroline=False),
            legend=LEGEND_BASE,
            **CT, height=280 + 30 * len(all_covs),
        )
        st.plotly_chart(fig_love, use_container_width=True)

        # SMD table
        smd_tbl = pd.DataFrame({
            "Covariate":       all_covs,
            "SMD Before":      [f"{v:.3f}" for v in smd_before],
            "SMD After":       [f"{v:.3f}" for v in smd_after],
            "|SMD| reduction": [f"{abs(b)-abs(a):.3f} ({'+' if abs(b)>abs(a) else '-'})"
                                for b, a in zip(smd_before, smd_after)],
            "Balanced (< 0.1)": ["✅" if abs(a) < 0.1 else "❌" for a in smd_after],
        }).set_index("Covariate")
        st.dataframe(smd_tbl, use_container_width=True)

        # ── ATE estimates ─────────────────────────────────────────────────
        st.markdown('<p class="sh">Average Treatment Effect (ATE)</p>', unsafe_allow_html=True)

        def ate_and_ci(df_src):
            t = df_src[df_src["treatment"] == 1]["target"]
            c = df_src[df_src["treatment"] == 0]["target"]
            ate = t.mean() - c.mean()
            se  = np.sqrt(t.var() / len(t) + c.var() / len(c))
            return ate, se, len(t), len(c)

        ate_b, se_b, nt_b, nc_b = ate_and_ci(df_psm)
        ate_a, se_a, nt_a, nc_a = ate_and_ci(df_matched)

        ate_tbl = pd.DataFrame({
            "ATE Estimate":     [f"{ate_b:.2f}", f"{ate_a:.2f}"],
            "Std. Error":       [f"{se_b:.2f}",  f"{se_a:.2f}"],
            "95 % CI":          [
                f"[{ate_b-1.96*se_b:.2f}, {ate_b+1.96*se_b:.2f}]",
                f"[{ate_a-1.96*se_a:.2f}, {ate_a+1.96*se_a:.2f}]",
            ],
            "n (treated/control)": [f"{nt_b}/{nc_b}", f"{nt_a}/{nc_a}"],
        }, index=["Naïve (unmatched)", "After PS Matching"])
        st.dataframe(ate_tbl, use_container_width=True)

        ate_change = ate_a - ate_b
        st.markdown(f"""<div class="ibox">
<b>Interpretation:</b> The naïve difference ({ate_b:.2f} units, 95 % CI
{ate_b-1.96*se_b:.2f}–{ate_b+1.96*se_b:.2f}) conflates the causal effect with
selection bias: patients with high BMI also tend to be older or have elevated
glucose. After 1:1 propensity score matching, the estimate shifts to
<b>{ate_a:.2f} units</b> (95 % CI {ate_a-1.96*se_a:.2f}–{ate_a+1.96*se_a:.2f}),
{"reducing" if abs(ate_a) < abs(ate_b) else "increasing"} the raw association by
{abs(ate_change):.2f} units once confounders are balanced.
The matched estimate is the more credible approximation of the causal effect
under the <em>ignorability assumption</em> (no unmeasured confounders).
</div>""", unsafe_allow_html=True)

        # ── DiD ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="sh">Part 2 — Difference-in-Differences (Ruhm 1996 / Beer Tax)</p>',
                    unsafe_allow_html=True)

        if fat_ok and fat_df is not None:
            did_src = fat_df
            st.caption("Dataset: US traffic fatalities per 10,000 population · "
                       "Treatment = above-median beer tax state · Pre = 1982–85 · Post = 1986–88")

            # 2×2 means
            means = (did_src.groupby(["high_tax", "post"])["fatal_rate"]
                     .mean().unstack())
            means.index = pd.Index(["Low-tax states", "High-tax states"], name="Group")
            means.columns = pd.Index(["Pre (1982–85)", "Post (1986–88)"])
            means["Δ (Post − Pre)"] = means["Post (1986–88)"] - means["Pre (1982–85)"]

            st.markdown("**2 × 2 Means Table (fatalities per 10,000 population):**")
            st.dataframe(means.round(4), use_container_width=True)

            # OLS DiD
            did_ols = sm.OLS(
                did_src["fatal_rate"],
                sm.add_constant(did_src[["high_tax", "post", "interaction"]]),
            ).fit(cov_type="HC1")  # heteroskedasticity-robust SEs
            did_est = did_ols.params["interaction"]
            did_se  = did_ols.bse["interaction"]
            did_p   = did_ols.pvalues["interaction"]

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("DiD Estimate",
                      f"{did_est:.4f}",
                      delta="fatalities / 10 k")
            d2.metric("Robust SE", f"{did_se:.4f}")
            d3.metric("95 % CI",
                      f"[{did_est-1.96*did_se:.4f}, {did_est+1.96*did_se:.4f}]")
            d4.metric("p-value", f"{did_p:.4f}",
                      delta="Significant" if did_p < 0.05 else "Not sig.")

            st.markdown(f"""<div class="ibox">
<b>Interpretation:</b> The DiD estimate of <b>{did_est:.4f}</b> fatalities per 10,000
represents the <em>differential</em> change in fatality rates for high-tax vs low-tax
states between the pre- and post-1985 periods.
{"This is statistically significant (HC1-robust SE), consistent with Ruhm's (1996) finding that higher beer taxes reduce traffic fatalities." if did_p < 0.05 else "This does not reach statistical significance; the beer tax effect may be confounded by other state-level trends."}
The DiD estimator is valid under the <b>parallel trends assumption</b>: absent the
policy difference, both groups would have followed identical trajectories.
</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="wbox">⚠️ Fatalities dataset could not be downloaded. '
                        'Check internet connectivity to access the AER R package repository.'
                        '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BAYESIAN INFERENCE (UCI Diabetes)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""<div class="ds-card">
<h3>📂 UCI Diabetes Dataset &nbsp;<span class="tag">Biostatistics</span><span class="tag">Conjugate Bayesian</span></h3>
<p>442 diabetes patients with baseline clinical measurements (Efron et al., 2004).
We perform <b>Bayesian linear regression</b> of disease progression (continuous,
25–346) on standardised BMI using the <em>conjugate Normal-Normal model</em>
(no MCMC required — analytical posterior). The interactive sample-size slider
demonstrates how data progressively overwhelm the prior: a core principle of
Bayesian statistics.</p>
<div class="meta">
  <span><b>N</b> = up to 442 patients (adjustable)</span>
  <span><b>X</b> = BMI (z-scored)</span>
  <span><b>Y</b> = disease progression at 1 year (z-scored)</span>
  <span><b>True slope ≈ correlation(BMI, progression) ≈ 0.59</b></span>
</div>
</div>""", unsafe_allow_html=True)

    ctrl4, main4 = st.columns([1, 3])

    with ctrl4:
        st.markdown("#### ⚙️ Controls")
        prior_mu  = st.slider("Prior mean  μ₀",  -2.0, 2.0,  0.0, 0.1, key="t4_pm")
        prior_var = st.slider("Prior variance τ₀²", 0.05, 5.0, 1.0, 0.05, key="t4_pv")
        n_bayes   = st.slider("Sample size  n",   10, 442, 100, 5, key="t4_n")
        st.info("Slope is on the **correlation scale** (both X and Y are z-scored). "
                "Increase n to watch the posterior sharpen around the data-driven estimate. "
                "Set τ₀² large for a diffuse (non-informative) prior.")
        with st.expander("Model details"):
            st.markdown(r"""
**Model** &nbsp;(centred, standardised variables)

$$y_i = \beta x_i + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0,\sigma^2)$$

**Conjugate prior** &nbsp; $\beta \sim \mathcal{N}(\mu_0, \tau_0^2)$

**Posterior** (closed-form)

$$\beta \mid \mathbf{y} \sim \mathcal{N}(\mu_n, \tau_n^2)$$

$$\tau_n^{-2} = \tau_0^{-2} + S_{xx}/\sigma^2, \quad
\mu_n = \tau_n^2\!\left(\frac{\mu_0}{\tau_0^2} + \frac{S_{xy}}{\sigma^2}\right)$$
""")

    # ── Bayesian computation ──────────────────────────────────────────────────
    # z-score both x (bmi) and y (target) so slope ≈ correlation
    bmi_z = (diab_df["bmi"] - diab_df["bmi"].mean()) / diab_df["bmi"].std()
    tgt_z = (diab_df["target"] - diab_df["target"].mean()) / diab_df["target"].std()

    x_all = bmi_z.values
    y_all = tgt_z.values

    x = x_all[:n_bayes]
    y = y_all[:n_bayes]

    # Sufficient statistics (already zero-mean by z-scoring, but re-centre per subsample)
    xc = x - x.mean()
    yc = y - y.mean()
    Sxx = float(np.sum(xc**2))
    Sxy = float(np.sum(xc * yc))
    b_ols = Sxy / Sxx if Sxx > 0 else 0.0
    resid = yc - b_ols * xc
    sig2  = max(float(np.sum(resid**2)) / max(n_bayes - 2, 1), 1e-6)

    tau_n_sq = 1.0 / (1.0 / prior_var + Sxx / sig2)
    mu_n     = tau_n_sq * (prior_mu / prior_var + Sxy / sig2)
    se_ols   = np.sqrt(sig2 / Sxx) if Sxx > 0 else 1e3

    with main4:
        # ── Distribution overlay ──────────────────────────────────────────
        lo4 = min(prior_mu - 3.5*prior_var**.5, mu_n - 4*tau_n_sq**.5, b_ols - 3*se_ols) - 0.05
        hi4 = max(prior_mu + 3.5*prior_var**.5, mu_n + 4*tau_n_sq**.5, b_ols + 3*se_ols) + 0.05
        brange = np.linspace(max(lo4, -3), min(hi4, 3), 600)

        pri_pdf  = stats.norm.pdf(brange, prior_mu, prior_var**.5)
        post_pdf = stats.norm.pdf(brange, mu_n,     tau_n_sq**.5)
        like_raw = stats.norm.pdf(brange, b_ols,    se_ols)
        pk = max(pri_pdf.max(), post_pdf.max())
        like_pdf = like_raw / max(like_raw.max(), 1e-12) * pk

        fig_bay = go.Figure()
        fig_bay.add_trace(go.Scatter(
            x=brange, y=pri_pdf, mode="lines",
            line=dict(color=P["blue"], width=2, dash="dot"),
            fill="tozeroy", fillcolor=P["blue_lt"],
            name=f"Prior  𝒩({prior_mu:.1f}, {prior_var:.2f})",
        ))
        fig_bay.add_trace(go.Scatter(
            x=brange, y=like_pdf, mode="lines",
            line=dict(color=P["amber"], width=2, dash="dash"),
            name=f"Likelihood (normalised, n = {n_bayes})",
        ))
        fig_bay.add_trace(go.Scatter(
            x=brange, y=post_pdf, mode="lines",
            line=dict(color=P["red"], width=2.5),
            fill="tozeroy", fillcolor=P["red_lt"],
            name=f"Posterior  𝒩({mu_n:.3f}, {tau_n_sq:.4f})",
        ))
        # OLS true line
        corr_full = float(np.corrcoef(x_all, y_all)[0, 1])
        fig_bay.add_vline(x=corr_full, line_dash="longdash", line_color="#059669",
                          annotation_text=f"Full-data OLS  β = {corr_full:.3f}",
                          annotation_position="top left",
                          annotation_font_color="#059669")
        fig_bay.update_layout(
            title=f"Bayesian Prior → Posterior Update  (n = {n_bayes} of 442 patients)",
            xaxis_title="Slope β (= standardised effect of BMI on disease progression)",
            yaxis_title="Probability density",
            legend=dict(**LEGEND_BASE, orientation="h", yanchor="top", y=-0.22),
            **CT, height=390,
        )
        st.plotly_chart(fig_bay, use_container_width=True)

        # ── Comparison table ──────────────────────────────────────────────
        st.markdown('<p class="sh">Frequentist OLS vs Bayesian Posterior Comparison</p>',
                    unsafe_allow_html=True)

        ci_lo = b_ols - 1.96 * se_ols
        ci_hi = b_ols + 1.96 * se_ols
        cr_lo = mu_n  - 1.96 * tau_n_sq**.5
        cr_hi = mu_n  + 1.96 * tau_n_sq**.5

        comp_df = pd.DataFrame({
            "Point Estimate":    [f"{b_ols:.4f}", f"{mu_n:.4f}"],
            "Lower Bound":       [f"{ci_lo:.4f}", f"{cr_lo:.4f}"],
            "Upper Bound":       [f"{ci_hi:.4f}", f"{cr_hi:.4f}"],
            "Interval Width":    [f"{ci_hi-ci_lo:.4f}", f"{cr_hi-cr_lo:.4f}"],
            "Std / Post. SD":    [f"{se_ols:.4f}", f"{tau_n_sq**.5:.4f}"],
            "Interval Type":     ["95 % Confidence Interval", "95 % Credible Interval"],
        }, index=["Frequentist OLS", "Bayesian Posterior"])
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("""<div class="ibox">
<b>Confidence interval</b> (frequentist): "If this study were repeated infinitely under the
same conditions, 95 % of such intervals would contain the true β." — a statement about the
<em>sampling procedure</em>, not about this specific interval.

<b>Credible interval</b> (Bayesian): "Given the observed data and our prior beliefs,
there is a 95 % posterior probability that β lies in this range." — a direct probability
statement about the parameter.

As n → ∞ the posterior mean converges to the OLS estimate and the credible interval
converges to the confidence interval: <em>data dominate the prior</em>.
The gap between the two estimates reflects how far the prior mean is from the likelihood peak,
weighted by the relative precision of prior vs data.
</div>""", unsafe_allow_html=True)

        # ── Scatter + regression lines ─────────────────────────────────────
        st.markdown('<p class="sh">Data and Fitted Lines</p>', unsafe_allow_html=True)

        d1c, d2c = st.columns(2)

        with d1c:
            xline = np.linspace(x.min(), x.max(), 200)
            xcline = xline - x.mean()
            fig_dat = go.Figure()
            fig_dat.add_trace(go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(color=P["blue"], opacity=0.3, size=5),
                name="Observed data",
                hovertemplate="BMI z: %{x:.3f}<br>Progression z: %{y:.3f}<extra></extra>",
            ))
            fig_dat.add_trace(go.Scatter(
                x=xline, y=y.mean() + b_ols * xcline, mode="lines",
                line=dict(color=P["amber"], width=2, dash="dash"),
                name=f"OLS  β̂ = {b_ols:.3f}",
            ))
            fig_dat.add_trace(go.Scatter(
                x=xline, y=y.mean() + mu_n * xcline, mode="lines",
                line=dict(color=P["red"], width=2.5),
                name=f"Posterior mean  E[β|data] = {mu_n:.3f}",
            ))
            fig_dat.update_layout(
                title=f"BMI vs Disease Progression  (n = {n_bayes})",
                xaxis_title="BMI (z-scored)",
                yaxis_title="Disease progression (z-scored)",
                legend=dict(**LEGEND_BASE, orientation="h", yanchor="top", y=-0.25),
                **CT, height=330,
            )
            st.plotly_chart(fig_dat, use_container_width=True)

        with d2c:
            # Posterior width as a function of n
            ns = np.arange(10, 443, 5)
            widths_ols  = []
            widths_bayes = []
            for ni in ns:
                xi = xc[:ni] if ni <= len(xc) else xc
                yi = yc[:ni] if ni <= len(yc) else yc
                Sx = max(float(np.sum((xi - xi.mean())**2)), 1e-9)
                Sy = float(np.sum((xi - xi.mean()) * (yi - yi.mean())))
                bh = Sy / Sx
                s2 = max(float(np.sum((yi - yi.mean() - bh*(xi - xi.mean()))**2)) / max(ni-2,1), 1e-6)
                se = np.sqrt(s2 / Sx)
                tn2 = 1 / (1/prior_var + Sx/s2)
                widths_ols.append(2 * 1.96 * se)
                widths_bayes.append(2 * 1.96 * tn2**.5)

            fig_n = go.Figure()
            fig_n.add_trace(go.Scatter(
                x=ns, y=widths_ols, mode="lines",
                line=dict(color=P["amber"], width=2, dash="dash"),
                name="Confidence interval width",
            ))
            fig_n.add_trace(go.Scatter(
                x=ns, y=widths_bayes, mode="lines",
                line=dict(color=P["red"], width=2),
                name="Credible interval width",
            ))
            fig_n.add_vline(x=n_bayes, line_dash="dot", line_color="#9CA3AF",
                            annotation_text=f"n = {n_bayes}")
            fig_n.update_layout(
                title="Interval Width vs Sample Size",
                xaxis_title="Sample size n",
                yaxis_title="95 % interval width",
                legend=dict(**LEGEND_BASE, orientation="h", yanchor="top", y=-0.25),
                **CT, height=330,
            )
            st.plotly_chart(fig_n, use_container_width=True)

        with st.expander("Show diabetes data (first 20 rows)"):
            display_cols = ["age", "sex", "bmi", "bp", "s1", "s6", "target"]
            st.dataframe(diab_df[display_cols].head(20).round(4), use_container_width=True)
