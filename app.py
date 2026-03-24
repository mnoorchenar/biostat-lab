import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioStat Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🔬 BioStat Lab")
st.caption(
    "Interactive biostatistical modeling suite — all datasets synthetic, "
    "generated at runtime · all charts Plotly · all computation server-side"
)

C = {
    "blue":   ("#2E86AB", "rgba(46,134,171,0.15)"),
    "red":    ("#E84855", "rgba(232,72,85,0.15)"),
    "green":  ("#44BBA4", "rgba(68,187,164,0.15)"),
    "orange": ("#F18F01", "rgba(241,143,1,0.15)"),
    "purple": ("#7B2D8B", "rgba(123,45,139,0.15)"),
}

tab1, tab2, tab3, tab4 = st.tabs([
    "⚕️ Survival Analysis",
    "📊 Linear Mixed Effects",
    "🎯 Causal Inference",
    "🎲 Bayesian Inference",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_p, col_m = st.columns([1, 3])

    with col_p:
        st.markdown("### Parameters")
        n_surv   = st.slider("Sample size",          100, 500, 300, 50,  key="t1_n")
        cens_pct = st.slider("Censoring rate (%)",    10,  50,  30,  5,  key="t1_c") / 100
        hr_val   = st.slider("Hazard ratio (Trt/Ctrl)", 0.5, 2.0, 0.7, 0.1, key="t1_hr")
        st.info("HR < 1 → treatment improves survival  \nHR > 1 → treatment worsens survival")

    @st.cache_data
    def gen_survival(n, cens, hr, seed=42):
        rng = np.random.default_rng(seed)
        n_ctrl = n // 2
        n_trt  = n - n_ctrl
        group  = np.array([0] * n_ctrl + [1] * n_trt)
        lam    = np.where(group == 1, 0.04 * hr, 0.04)
        event_t = rng.exponential(1.0 / lam)
        avg_lam = 0.04 * (1 + hr) / 2
        lam_c   = (cens * avg_lam) / max(1 - cens, 1e-6)
        cens_t  = rng.exponential(1.0 / lam_c, size=n)
        obs_t   = np.minimum(event_t, cens_t)
        ev      = (event_t <= cens_t).astype(int)
        return pd.DataFrame({
            "time": obs_t, "event": ev, "group": group,
            "group_label": np.where(group == 1, "Treatment", "Control"),
        })

    df_s = gen_survival(n_surv, cens_pct, hr_val)

    with col_m:
        # ── KM figure ────────────────────────────────────────────────────────
        fig_km = go.Figure()
        for grp, (c_line, c_fill) in zip(
            ["Control", "Treatment"], [C["blue"], C["red"]]
        ):
            sub  = df_s[df_s["group_label"] == grp]
            kmf  = KaplanMeierFitter()
            kmf.fit(sub["time"], sub["event"], label=grp)
            tl   = kmf.timeline
            sf   = kmf.survival_function_[grp].values
            cl   = kmf.confidence_interval_[f"{grp}_lower_0.95"].values
            cu   = kmf.confidence_interval_[f"{grp}_upper_0.95"].values
            # CI band
            fig_km.add_trace(go.Scatter(
                x=np.concatenate([tl, tl[::-1]]),
                y=np.concatenate([cu, cl[::-1]]),
                fill="toself", fillcolor=c_fill,
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{grp} 95% CI", hoverinfo="skip", showlegend=False,
            ))
            # Survival step
            fig_km.add_trace(go.Scatter(
                x=tl, y=sf, mode="lines",
                line=dict(color=c_line, width=2.5, shape="hv"),
                name=grp,
            ))

        fig_km.update_layout(
            title="Kaplan-Meier Survival Curves with 95% Confidence Intervals",
            xaxis_title="Time", yaxis_title="Survival Probability",
            yaxis=dict(range=[0, 1.05]), template="plotly_white", height=380,
            legend=dict(orientation="h", yanchor="top", y=-0.15),
        )
        st.plotly_chart(fig_km, use_container_width=True)

        # ── Metrics ──────────────────────────────────────────────────────────
        ctrl_s = df_s[df_s["group"] == 0]
        trt_s  = df_s[df_s["group"] == 1]
        lr_res = logrank_test(
            ctrl_s["time"], trt_s["time"],
            event_observed_A=ctrl_s["event"],
            event_observed_B=trt_s["event"],
        )
        kmf_c = KaplanMeierFitter().fit(ctrl_s["time"], ctrl_s["event"])
        kmf_t = KaplanMeierFitter().fit(trt_s["time"], trt_s["event"])

        m1, m2, m3, m4 = st.columns(4)
        p_val = lr_res.p_value
        m1.metric("Log-rank p-value", f"{p_val:.4f}",
                  delta="p < 0.05" if p_val < 0.05 else "NS")
        m2.metric("Median Survival — Control",   f"{kmf_c.median_survival_time_:.2f}")
        m3.metric("Median Survival — Treatment", f"{kmf_t.median_survival_time_:.2f}")
        m4.metric("Observed Censoring Rate", f"{1 - df_s['event'].mean():.1%}")

        # ── Cox PH ───────────────────────────────────────────────────────────
        st.markdown("#### Cox Proportional Hazards Model")
        cph = CoxPHFitter()
        cph.fit(df_s[["time", "event", "group"]], duration_col="time", event_col="event")
        s = cph.summary
        cox_df = pd.DataFrame({
            "Hazard Ratio":        s["exp(coef)"].round(3),
            "HR 95% CI Lower":     s["exp(coef) lower 95%"].round(3),
            "HR 95% CI Upper":     s["exp(coef) upper 95%"].round(3),
            "z-stat":              s["z"].round(3),
            "p-value":             s["p"].round(4),
        })
        st.dataframe(cox_df, use_container_width=True)
        st.caption(
            f"Concordance index (C-statistic): **{cph.concordance_index_:.3f}**  "
            f"— 0.5 = random, 1.0 = perfect discrimination"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LINEAR MIXED EFFECTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_p2, col_m2 = st.columns([1, 3])

    with col_p2:
        st.markdown("### Parameters")
        n_subj  = st.slider("Number of subjects",  20, 100, 50,  5,   key="t2_n")
        n_tp    = st.slider("Time points",           3,   8,  5,  1,   key="t2_tp")
        re_sd   = st.slider("Random effect SD",    1.0, 10.0, 4.0, 0.5, key="t2_re")
        st.info(
            "Random intercepts simulate subject-level "
            "baseline differences in blood pressure."
        )

    @st.cache_data
    def gen_lme(n_subjects, n_timepoints, rand_sd, seed=42):
        rng = np.random.default_rng(seed)
        subjects   = np.repeat(np.arange(n_subjects), n_timepoints)
        timepoints = np.tile(np.arange(n_timepoints), n_subjects)
        treatment  = np.repeat(rng.integers(0, 2, size=n_subjects), n_timepoints)
        rand_int   = rng.normal(0, rand_sd, size=n_subjects)
        rand_obs   = rand_int[subjects]
        bp = (120
              + (-1.5) * timepoints
              + (-5.0) * treatment
              + rand_obs
              + rng.normal(0, 3.0, size=len(subjects)))
        return pd.DataFrame({
            "subject": subjects, "time_pt": timepoints,
            "treatment": treatment, "outcome": bp,
        })

    df_lme = gen_lme(n_subj, n_tp, re_sd)

    with col_m2:
        # Fit model
        try:
            mdl    = smf.mixedlm("outcome ~ time_pt + treatment", df_lme,
                                  groups=df_lme["subject"])
            result = mdl.fit(reml=True, method="lbfgs")
        except Exception:
            mdl    = smf.mixedlm("outcome ~ time_pt + treatment", df_lme,
                                  groups=df_lme["subject"])
            result = mdl.fit(reml=True)

        # ── Fixed effects table ───────────────────────────────────────────
        st.markdown("#### Fixed Effects")
        fe_names = list(result.fe_params.index)
        fe_coef  = result.fe_params.values
        try:
            fe_se = result.bse[fe_names].values
        except Exception:
            fe_se = result.bse.iloc[:len(fe_names)].values
        fe_z  = fe_coef / np.where(fe_se > 0, fe_se, np.nan)
        fe_p  = 2 * stats.norm.sf(np.abs(fe_z))

        fe_df = pd.DataFrame({
            "Estimate":      fe_coef.round(4),
            "Std Error":     fe_se.round(4),
            "z":             fe_z.round(3),
            "p-value":       fe_p.round(4),
            "95% CI Lower":  (fe_coef - 1.96 * fe_se).round(4),
            "95% CI Upper":  (fe_coef + 1.96 * fe_se).round(4),
        }, index=fe_names)
        st.dataframe(fe_df, use_container_width=True)

        # ── Variance components ───────────────────────────────────────────
        re_var_est   = float(result.cov_re.iloc[0, 0])
        resid_var_est = float(result.scale)
        icc          = re_var_est / (re_var_est + resid_var_est)

        v1, v2, v3 = st.columns(3)
        v1.metric("Random Intercept Variance", f"{re_var_est:.3f}")
        v2.metric("Residual Variance",          f"{resid_var_est:.3f}")
        v3.metric("ICC",                        f"{icc:.3f}",
                  help="Intraclass Correlation Coefficient — fraction of total variance "
                       "attributable to between-subject differences")

        # ── Observed vs Fitted ────────────────────────────────────────────
        fitted = result.fittedvalues
        lo, hi = df_lme["outcome"].min(), df_lme["outcome"].max()
        fig_ov = go.Figure()
        fig_ov.add_trace(go.Scatter(
            x=df_lme["outcome"], y=fitted, mode="markers",
            marker=dict(color=C["blue"][0], opacity=0.45, size=5),
            name="Observations",
        ))
        fig_ov.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="gray", dash="dash", width=1.5), name="y = x",
        ))
        fig_ov.update_layout(
            title="Observed vs Fitted Values",
            xaxis_title="Observed", yaxis_title="Fitted",
            template="plotly_white", height=320,
        )
        st.plotly_chart(fig_ov, use_container_width=True)

        # ── Subject-level trajectories ────────────────────────────────────
        st.markdown("#### Subject-Level Trajectories")
        fig_tj = go.Figure()
        tp_arr = np.arange(n_tp)
        fe     = result.fe_params
        n_show = min(n_subj, 25)
        legend_shown = {0: False, 1: False}

        for sid in range(n_show):
            sub_rows = df_lme[df_lme["subject"] == sid]
            trt_val  = int(sub_rows["treatment"].iloc[0])
            try:
                re_val = float(result.random_effects[sid].iloc[0])
            except Exception:
                re_val = 0.0
            y_ind = (fe["Intercept"]
                     + fe["time_pt"]   * tp_arr
                     + fe["treatment"] * trt_val
                     + re_val)
            color = C["red"][0] if trt_val == 1 else C["blue"][0]
            label = "Treatment" if trt_val == 1 else "Control"
            show  = not legend_shown[trt_val]
            if show:
                legend_shown[trt_val] = True
            fig_tj.add_trace(go.Scatter(
                x=tp_arr, y=y_ind, mode="lines",
                line=dict(color=color, width=1),
                opacity=0.35,
                name=label, showlegend=show,
                legendgroup=label,
            ))

        # Population averages
        for trt_val, label, color in [
            (0, "Control (pop. avg.)",   C["blue"][0]),
            (1, "Treatment (pop. avg.)", C["red"][0]),
        ]:
            y_pop = (fe["Intercept"]
                     + fe["time_pt"]   * tp_arr
                     + fe["treatment"] * trt_val)
            fig_tj.add_trace(go.Scatter(
                x=tp_arr, y=y_pop, mode="lines",
                line=dict(color=color, width=3),
                name=label, legendgroup=label,
            ))

        fig_tj.update_layout(
            title="Individual Trajectories + Population Average",
            xaxis_title="Time Point", yaxis_title="Blood Pressure",
            template="plotly_white", height=380,
            legend=dict(orientation="h", yanchor="top", y=-0.15),
        )
        st.plotly_chart(fig_tj, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CAUSAL INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_p3, col_m3 = st.columns([1, 3])

    with col_p3:
        st.markdown("### Parameters")
        n_ci      = st.slider("Sample size", 200, 800, 400, 50, key="t3_n")
        conf_str  = st.radio(
            "Confounding strength", ["Low", "Medium", "High"], index=1, key="t3_cs"
        )
        st.info(
            "Confounding: treatment assignment depends on age "
            "and comorbidity score. True ATE = **5.0**."
        )

    @st.cache_data
    def gen_and_match(n, confounding, seed=42):
        rng      = np.random.default_rng(seed)
        gamma    = {"Low": 0.5, "Medium": 1.5, "High": 3.0}[confounding]
        age      = rng.normal(50, 10, n)
        comorbid = rng.normal(0,   1,  n)
        logit    = -1.0 + gamma * 0.03 * (age - 50) + gamma * 0.5 * comorbid
        ps_true  = 1 / (1 + np.exp(-logit))
        treatment = rng.binomial(1, ps_true)
        outcome  = (50
                    + 0.2  * (age - 50)
                    + 2.0  * comorbid
                    + 5.0  * treatment
                    + rng.normal(0, 5, n))
        df = pd.DataFrame({
            "age": age, "comorbidity": comorbid,
            "treatment": treatment, "outcome": outcome,
        })
        # Propensity score via logistic regression
        lr = LogisticRegression(random_state=42, max_iter=500)
        lr.fit(df[["age", "comorbidity"]].values, df["treatment"].values)
        df["ps"] = lr.predict_proba(df[["age", "comorbidity"]].values)[:, 1]

        # 1:1 NN matching without replacement
        t_idx  = np.where(df["treatment"] == 1)[0]
        c_idx  = np.where(df["treatment"] == 0)[0]
        t_ps   = df["ps"].values[t_idx]
        c_ps   = df["ps"].values[c_idx]

        used   = set()
        pairs  = []
        for ti, tp in sorted(zip(t_idx, t_ps), key=lambda x: x[1]):
            best_ci, best_d = None, float("inf")
            for j, ci in enumerate(c_idx):
                if ci not in used:
                    d = abs(tp - c_ps[j])
                    if d < best_d:
                        best_d, best_ci = d, ci
            if best_ci is not None:
                pairs.append((ti, best_ci))
                used.add(best_ci)

        all_matched = [p[0] for p in pairs] + [p[1] for p in pairs]
        df_matched  = df.iloc[all_matched].copy()
        return df, df_matched

    df_ci, df_m = gen_and_match(n_ci, conf_str)

    with col_m3:
        # ── PS Distribution ───────────────────────────────────────────────
        fig_ps = make_subplots(rows=1, cols=2,
                               subplot_titles=["Before Matching", "After Matching"])
        for is_trt, label, color in [
            (0, "Control",   C["blue"][0]),
            (1, "Treated",   C["red"][0]),
        ]:
            for col_idx, df_src in [(1, df_ci), (2, df_m)]:
                sub = df_src[df_src["treatment"] == is_trt]["ps"]
                fig_ps.add_trace(
                    go.Histogram(x=sub, name=label, nbinsx=30,
                                 marker_color=color, opacity=0.6,
                                 legendgroup=label,
                                 showlegend=(col_idx == 1)),
                    row=1, col=col_idx,
                )
        fig_ps.update_layout(
            title="Propensity Score Distribution Before and After Matching",
            barmode="overlay", template="plotly_white", height=320,
            legend=dict(orientation="h", yanchor="top", y=-0.15),
        )
        st.plotly_chart(fig_ps, use_container_width=True)

        # ── SMD Table ─────────────────────────────────────────────────────
        st.markdown("#### Standardized Mean Differences (SMD)")
        smd_rows = []
        for cov in ["age", "comorbidity"]:
            def smd_calc(df_src, cov=cov):
                t = df_src[df_src["treatment"] == 1][cov]
                c = df_src[df_src["treatment"] == 0][cov]
                sd_pool = np.sqrt((t.std()**2 + c.std()**2) / 2)
                return (t.mean() - c.mean()) / (sd_pool if sd_pool > 0 else 1)
            before = smd_calc(df_ci)
            after  = smd_calc(df_m)
            smd_rows.append({
                "Covariate": cov,
                "SMD Before": round(before, 3),
                "SMD After":  round(after, 3),
                "Balanced (|SMD|<0.1)": "✓" if abs(after) < 0.1 else "✗",
            })
        st.dataframe(pd.DataFrame(smd_rows).set_index("Covariate"),
                     use_container_width=True)

        # ── ATE estimates ─────────────────────────────────────────────────
        st.markdown("#### Average Treatment Effect (ATE)")

        def ate_ci(df_src):
            t_o = df_src[df_src["treatment"] == 1]["outcome"]
            c_o = df_src[df_src["treatment"] == 0]["outcome"]
            ate = t_o.mean() - c_o.mean()
            se  = np.sqrt(t_o.var() / len(t_o) + c_o.var() / len(c_o))
            return ate, se

        ate_b, se_b = ate_ci(df_ci)
        ate_a, se_a = ate_ci(df_m)
        ate_df = pd.DataFrame({
            "ATE Estimate": [f"{ate_b:.3f}", f"{ate_a:.3f}", "5.000"],
            "95% CI": [
                f"[{ate_b-1.96*se_b:.3f}, {ate_b+1.96*se_b:.3f}]",
                f"[{ate_a-1.96*se_a:.3f}, {ate_a+1.96*se_a:.3f}]",
                "—",
            ],
            "Bias from Truth": [
                f"{abs(ate_b-5):.3f}", f"{abs(ate_a-5):.3f}", "0.000"
            ],
        }, index=["Naive (unmatched)", "After PS Matching", "True Effect"])
        st.dataframe(ate_df, use_container_width=True)
        st.info(
            f"**Interpretation:** The naive estimate ({ate_b:.2f}) is biased by "
            f"{abs(ate_b-5):.2f} units from the true ATE (5.0) due to {conf_str.lower()} confounding. "
            f"After 1:1 propensity score matching, the bias drops to "
            f"{abs(ate_a-5):.2f} units, as the matched sample is better balanced "
            f"on age and comorbidity."
        )

        # ── Difference-in-Differences ─────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Difference-in-Differences (DiD)")

        @st.cache_data
        def gen_did(seed=99):
            rng = np.random.default_rng(seed)
            n_cell = 100
            true_effect = 6.0
            trend = 2.0
            y = np.concatenate([
                rng.normal(50,           5, n_cell),   # ctrl  pre
                rng.normal(50 + trend,   5, n_cell),   # ctrl  post
                rng.normal(50,           5, n_cell),   # treat pre
                rng.normal(50 + trend + true_effect, 5, n_cell),  # treat post
            ])
            treated = np.array([0]*n_cell*2 + [1]*n_cell*2)
            post    = np.array([0]*n_cell + [1]*n_cell + [0]*n_cell + [1]*n_cell)
            return pd.DataFrame({"y": y, "treated": treated, "post": post}), true_effect

        df_did, true_did = gen_did()
        df_did["interaction"] = df_did["treated"] * df_did["post"]

        means = df_did.groupby(["treated", "post"])["y"].mean().unstack()
        means.index = pd.Index(["Control", "Treatment"], name="Group")
        means.columns = pd.Index(["Pre", "Post"])
        means["Δ (Post−Pre)"] = means["Post"] - means["Pre"]
        st.markdown("**2 × 2 Means Table:**")
        st.dataframe(means.round(3), use_container_width=True)

        did_ols = sm.OLS(
            df_did["y"],
            sm.add_constant(df_did[["treated", "post", "interaction"]]),
        ).fit()
        did_est = did_ols.params["interaction"]
        did_se  = did_ols.bse["interaction"]

        d1, d2, d3 = st.columns(3)
        d1.metric("DiD Estimate",      f"{did_est:.3f}")
        d2.metric("Std Error",         f"{did_se:.3f}")
        d3.metric("True DiD Effect",   f"{true_did:.1f}")
        st.caption(
            "DiD assumes **parallel trends**: absent treatment, "
            "both groups would have changed by the same amount. "
            "The interaction coefficient (treated × post) identifies the causal effect."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BAYESIAN INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    col_p4, col_m4 = st.columns([1, 3])

    with col_p4:
        st.markdown("### Parameters")
        prior_mu  = st.slider("Prior mean (μ₀)",      -3.0, 3.0, 0.0, 0.1, key="t4_pm")
        prior_var = st.slider("Prior variance (τ₀²)",  0.1, 10.0, 1.0, 0.1, key="t4_pv")
        n_bayes   = st.slider("Sample size (n)",         10, 500, 100,  10,  key="t4_n")
        st.info("True slope = **1.5**. Increase n to watch the posterior "
                "concentrate around the truth and overwhelm the prior.")

    @st.cache_data
    def gen_bayes_full(seed=42, n_total=500, true_b=1.5, true_a=2.0, noise=3.0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-3, 3, n_total)
        y = true_a + true_b * x + rng.normal(0, noise, n_total)
        return x, y

    x_full, y_full = gen_bayes_full()
    x = x_full[:n_bayes]
    y = y_full[:n_bayes]

    # Center so intercept cancels for slope posterior
    xc = x - x.mean()
    yc = y - y.mean()
    Sxx = np.sum(xc ** 2)
    Sxy = np.sum(xc * yc)

    beta_mle = Sxy / Sxx if Sxx > 0 else 0.0
    resid    = yc - beta_mle * xc
    sigma_sq = max(np.sum(resid**2) / max(n_bayes - 1, 1), 1e-6)

    # Conjugate posterior: beta | data ~ N(mu_n, tau_n^2)
    tau_n_sq  = 1.0 / (1.0 / prior_var + Sxx / sigma_sq)
    mu_n      = tau_n_sq * (prior_mu / prior_var + Sxy / sigma_sq)
    se_ols    = np.sqrt(sigma_sq / Sxx) if Sxx > 0 else 1e3

    with col_m4:
        # ── Distribution plot ─────────────────────────────────────────────
        # Compute sensible x-range
        lo4 = min(prior_mu - 3*prior_var**0.5, mu_n - 3*tau_n_sq**0.5,
                  beta_mle - 2*se_ols, 1.5) - 0.3
        hi4 = max(prior_mu + 3*prior_var**0.5, mu_n + 3*tau_n_sq**0.5,
                  beta_mle + 2*se_ols, 1.5) + 0.3
        brange = np.linspace(lo4, hi4, 600)

        prior_pdf   = stats.norm.pdf(brange, prior_mu, prior_var**0.5)
        post_pdf    = stats.norm.pdf(brange, mu_n,     tau_n_sq**0.5)
        like_pdf_raw = stats.norm.pdf(brange, beta_mle, se_ols)
        # Normalize likelihood to same peak as tallest distribution for readability
        scale = max(prior_pdf.max(), post_pdf.max())
        like_pdf = like_pdf_raw / max(like_pdf_raw.max(), 1e-12) * scale

        fig_bay = go.Figure()
        fig_bay.add_trace(go.Scatter(
            x=brange, y=prior_pdf, mode="lines",
            line=dict(color=C["blue"][0], width=2, dash="dot"),
            fill="tozeroy", fillcolor=C["blue"][1],
            name=f"Prior  N({prior_mu:.1f}, {prior_var:.1f})",
        ))
        fig_bay.add_trace(go.Scatter(
            x=brange, y=like_pdf, mode="lines",
            line=dict(color=C["orange"][0], width=2, dash="dash"),
            name=f"Likelihood (scaled, n={n_bayes})",
        ))
        fig_bay.add_trace(go.Scatter(
            x=brange, y=post_pdf, mode="lines",
            line=dict(color=C["red"][0], width=2.5),
            fill="tozeroy", fillcolor=C["red"][1],
            name=f"Posterior N({mu_n:.3f}, {tau_n_sq:.4f})",
        ))
        fig_bay.add_vline(
            x=1.5, line_dash="longdash", line_color="black",
            annotation_text="True β = 1.5", annotation_position="top right",
        )
        fig_bay.update_layout(
            title=f"Prior → Posterior Bayesian Update  (n = {n_bayes})",
            xaxis_title="Slope β₁", yaxis_title="Density",
            template="plotly_white", height=370,
            legend=dict(orientation="h", yanchor="top", y=-0.15),
        )
        st.plotly_chart(fig_bay, use_container_width=True)

        # ── Comparison table ──────────────────────────────────────────────
        st.markdown("#### Frequentist OLS vs Bayesian Posterior")
        ci_lo = beta_mle - 1.96 * se_ols
        ci_hi = beta_mle + 1.96 * se_ols
        cr_lo = mu_n - 1.96 * tau_n_sq**0.5
        cr_hi = mu_n + 1.96 * tau_n_sq**0.5

        comp_df = pd.DataFrame({
            "Point Estimate":     [f"{beta_mle:.4f}", f"{mu_n:.4f}"],
            "Lower Bound":        [f"{ci_lo:.4f}",    f"{cr_lo:.4f}"],
            "Upper Bound":        [f"{ci_hi:.4f}",    f"{cr_hi:.4f}"],
            "Width":              [f"{ci_hi-ci_lo:.4f}", f"{cr_hi-cr_lo:.4f}"],
            "Interval Type":      ["95% Confidence Interval", "95% Credible Interval"],
        }, index=["Frequentist OLS", "Bayesian Posterior"])
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("""
**Confidence interval:** "If this experiment were repeated infinitely, 95% of such
intervals would contain the true β." — a statement about the *procedure*, not this interval.

**Credible interval:** "Given the data and prior, there is 95% posterior probability that
β lies in this range." — a direct probability statement about the parameter.

As n → ∞, the posterior mean → MLE and the credible interval → confidence interval
(prior is overwhelmed by data).
        """)

        # ── Data scatter with regression lines ────────────────────────────
        x_line = np.linspace(x.min(), x.max(), 200)
        xc_line = x_line - x.mean()

        fig_dat = go.Figure()
        fig_dat.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(color=C["blue"][0], opacity=0.35, size=5),
            name="Data",
        ))
        fig_dat.add_trace(go.Scatter(
            x=x_line, y=y.mean() + beta_mle * xc_line, mode="lines",
            line=dict(color=C["orange"][0], width=2, dash="dash"),
            name=f"OLS  β̂ = {beta_mle:.3f}",
        ))
        fig_dat.add_trace(go.Scatter(
            x=x_line, y=y.mean() + mu_n * xc_line, mode="lines",
            line=dict(color=C["red"][0], width=2),
            name=f"Bayes  E[β|data] = {mu_n:.3f}",
        ))
        fig_dat.update_layout(
            title=f"Data and Regression Lines  (n = {n_bayes})",
            xaxis_title="x", yaxis_title="y",
            template="plotly_white", height=320,
            legend=dict(orientation="h", yanchor="top", y=-0.15),
        )
        st.plotly_chart(fig_dat, use_container_width=True)
