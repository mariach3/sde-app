# app2_improved.py
# Streamlit app: Volatility & SDE Explorer — with contextual help, Theory page, and a dedicated Performance window
# Author: Maria Manitara Christodoulidou
# Improvements in this version:
# - NEW: "📚 Theory & Notes" page with collapsible sections for Brownian Motion, GBM/Black–Scholes, Heston, Risk-Neutral pricing, Greeks,
#        Calibration tips, and a searchable Glossary. Uses st.latex and st.expander for a clean, syllabus-like layout.
# - Reuses your help blocks and robust image handling on the Theory page (e.g., to show helpful diagrams if present).
# - Leaves the SDE Visualiser, Performance, and Vol Smile pages unchanged except for the router to include the new page.

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from pathlib import Path  # robust file handling

# --------------------------------------------------
# App config (only once)
# --------------------------------------------------
st.set_page_config(page_title="Volatility & SDE Explorer", layout="wide", page_icon="📈")

# ==================================================
# ===============  SHARED UTILITIES  ===============
# ==================================================

@st.cache_data(show_spinner=False)
def load_clean(path: str) -> pd.DataFrame:
    """Load and lightly clean the options CSV used in the Vol Smile Explorer page."""
    df = pd.read_csv(path, skipinitialspace=True, on_bad_lines="skip")
    df.columns = [c.strip().lstrip("[").rstrip("]") for c in df.columns]
    for col in ["QUOTE_READTIME", "QUOTE_DATE", "EXPIRE_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    numeric_cols = [
        "QUOTE_UNIXTIME", "QUOTE_TIME_HOURS", "UNDERLYING_LAST",
        "DTE", "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO", "C_IV",
        "C_VOLUME", "C_LAST", "C_SIZE", "C_BID", "C_ASK",
        "STRIKE", "P_BID", "P_ASK", "P_SIZE", "P_LAST",
        "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA", "P_RHO", "P_IV",
        "P_VOLUME", "STRIKE_DISTANCE", "STRIKE_DISTANCE_PCT"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["C_SIZE", "P_SIZE"]:
        if col in df.columns:
            df = df.drop(columns=col)
    return df


def _subset(df: pd.DataFrame, date, expiry) -> pd.DataFrame:
    d = pd.Timestamp(date).normalize()
    e = pd.Timestamp(expiry).normalize()
    lhs_date   = df["QUOTE_DATE"].dt.normalize()
    lhs_expiry = df["EXPIRE_DATE"].dt.normalize()
    sub = df.loc[(lhs_date == d) & (lhs_expiry == e)].copy()
    return sub


def _atm_iv(sub: pd.DataFrame, S0: float) -> float:
    if "C_IV" not in sub.columns:
        raise ValueError("C_IV column missing for market IVs.")
    mny = sub["STRIKE"] / S0
    near = sub.loc[mny.between(0.95, 1.05) & sub["C_IV"].notna(), "C_IV"]
    return float(np.median(near)) if not near.empty else float(np.nanmedian(sub["C_IV"]))


# Convenience: set controls programmatically
def set_state(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v


# Small helper to attach a Help block after plots
def help_block(title: str, body_md: str, presets: list | None = None):
    """Render a collapsible help section with optional preset buttons.
    presets: list of dicts {label: str, on_click: callable or None}
    """
    with st.expander(title):
        st.markdown(body_md)
        if presets:
            cols = st.columns(min(4, len(presets)))
            for i, p in enumerate(presets):
                with cols[i % len(cols)]:
                    st.button(p.get("label", "Use preset"), on_click=p.get("on_click"))

# ---- Image helpers (robust display with fallback) ----
def _find_asset(filename: str) -> Path | None:
    """Search common locations for an asset (alongside app, ./assets, CWD)."""
    try:
        here = Path(__file__).parent
    except NameError:
        # __file__ may be undefined in some environments; fall back to CWD
        here = Path.cwd()
    candidates = [
        here / filename,
        here / "assets" / filename,
        Path.cwd() / filename,
        Path.cwd() / "assets" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def show_image_safe(filename: str, caption: str = ""):
    """Try to open image via Pillow; if missing/corrupt, draw a fallback figure instead of erroring."""
    img_path = _find_asset(filename)
    if img_path is not None:
        try:
            from PIL import Image  # Pillow
            with Image.open(img_path) as im:
                st.image(im, use_container_width=True, caption=caption)
                return
        except Exception as e:
            st.warning(f"Found image at {img_path}, but couldn’t open it: {e}. Showing a fallback chart below.")
    else:
        st.info(f"Image '{filename}' not found in common locations (./, ./assets). Showing a fallback chart below.")

    # Fallback chart so the UI remains informative
    N = np.array([50, 100, 200, 400, 800, 1600])
    # Illustrative times (arbitrary scale) to show relative growth only
    t_gbm = 0.002 * N
    t_hes = 0.006 * N
    fig, ax = plt.subplots()
    ax.plot(N, t_gbm, label="GBM (illustrative)")
    ax.plot(N, t_hes, label="Heston (illustrative)")
    ax.set_xlabel("Time steps N")
    ax.set_ylabel("Runtime (relative units)")
    ax.set_title("GBM vs Heston timing vs time steps — fallback view")
    ax.grid(True); ax.legend()
    st.pyplot(fig)
    st.caption(
        "Place the real PNG as './assets/benchmark_sde_timing.png' or alongside 'app2_improved.py' to display it here."
    )

# ==================================================
# ===============  PAGE 0: THEORY  =================
# ==================================================

def _latex_block(title: str, latex_lines: list[str], prose_md: str = "", tips_md: str = ""):
    """Common pattern: collapsible with LaTeX + optional prose + optional tips."""
    with st.expander(title, expanded=False):
        if prose_md:
            st.markdown(prose_md)
        for L in latex_lines:
            st.latex(L)
        if tips_md:
            help_block("Notes & tips", tips_md)

def page_theory():
    st.title("📚 Theory & Notes")
    st.caption(
        "Quick reference for core concepts used in the app. Use the **searchable glossary** below or open specific sections."
    )

    # --- Quick jump & glossary search ---
    colA, colB = st.columns([1.2, 0.8])
    with colA:
        jump = st.selectbox(
            "Jump to section",
            ["—", "Brownian Motion", "GBM & Black–Scholes", "Heston Model",
             "Risk-Neutral Pricing", "Option Greeks", "Calibration Tips", "Glossary"],
            index=0
        )
    with colB:
        query = st.text_input("Glossary quick search", value="", placeholder="e.g., martingale, moneyness, Feller condition")

    # --- Brownian Motion ---
    _latex_block(
        "Brownian Motion (Wiener process)",
        latex_lines=[
            r"W_0 = 0,\quad W_t - W_s \sim \mathcal{N}(0,\, t-s)\ \text{for } t>s,\quad \text{independent increments},",
            r"dW_t \text{ is the formal increment used in Itô calculus.}"
        ],
        prose_md=(
            "Brownian motion is the fundamental noise driving continuous-time models. "
            "Its independent, Gaussian increments make Itô’s lemma and stochastic integration tractable."
        ),
        tips_md="- Scaling: for any \(c>0\), \(W_{ct} \overset{d}{=} \sqrt{c}\,W_t\)."
    )

    # --- GBM & Black–Scholes ---
    _latex_block(
        "Geometric Brownian Motion & Black–Scholes",
        latex_lines=[
            r"dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,\quad S_t = S_0 \exp\!\Big((\mu-\tfrac{1}{2}\sigma^2)t + \sigma W_t\Big),",
            r"C^{\mathrm{BS}} = S_0 e^{-q\tau}\Phi(d_1) - K e^{-r\tau}\Phi(d_2),",
            r"d_1 = \frac{\ln(S_0 e^{-q\tau}/K) + \tfrac{1}{2}\sigma^2 \tau}{\sigma \sqrt{\tau}},\quad d_2 = d_1 - \sigma \sqrt{\tau}."
        ],
        prose_md=(
            "Under risk-neutral dynamics with dividend yield \(q\), GBM implies log-normal prices and closed-form Black–Scholes "
            "values for European options."
        ),
        tips_md=(
            "- Flat IV surface is a **model assumption**; market smiles/skews indicate model misspecification.\n"
            "- Use this page’s ‘GBM vs Heston’ plots to visualise when GBM is too simple."
        )
    )

    # --- Heston Model ---
    _latex_block(
        "Heston Stochastic Volatility",
        latex_lines=[
            r"\begin{aligned}"
            r"dS_t &= S_t\big((r-q)\,dt + \sqrt{v_t}\,dW_t^{(1)}\big),\\"
            r"dv_t &= \kappa(\theta - v_t)\,dt + \sigma_v \sqrt{v_t}\,dW_t^{(2)},\quad d\langle W^{(1)},W^{(2)}\rangle_t = \rho\,dt."
            r"\end{aligned}",
            r"\text{Feller condition: } 2\kappa\theta \ge \sigma_v^2 \ \text{(keeps } v_t>0 \text{ a.s.).}"
        ],
        prose_md=(
            "Variance \(v_t\) mean-reverts to \(\\theta\) at speed \(\\kappa\) with volatility-of-vol \(\\sigma_v\). "
            "The correlation \(\\rho<0\) typically generates equity skew."
        ),
        tips_md=(
            "- Increasing \(\\sigma_v\) fattens tails; more negative \(\\rho\) increases downside skew.\n"
            "- Calibration aims to match observed IV smiles/skews for a date–expiry slice."
        )
    )

    # --- Risk-Neutral Pricing ---
    _latex_block(
        "Risk-Neutral Pricing (sketch)",
        latex_lines=[
            r"V_0 = e^{-r\tau}\,\mathbb{E}^{\mathbb{Q}}\!\left[\text{payoff}(S_\tau)\,\middle|\,\mathcal{F}_0\right].",
            r"\text{For Heston: } C_0 = S_0 e^{-q\tau} P_1 - K e^{-r\tau} P_2,\quad \text{with } P_i \text{ from the characteristic function.}"
        ],
        prose_md="Under \(\\mathbb{Q}\), discounted asset prices are martingales. Heston prices follow from Fourier methods."
    )

    # --- Greeks ---
    with st.expander("Option Greeks (intuition + key formulas)"):
        st.markdown(
            "- **Delta (Δ):** sensitivity to spot; **Gamma (Γ):** curvature; **Vega (ν):** sensitivity to volatility; "
            "**Theta (Θ):** time decay; **Rho (ρ):** sensitivity to rates.\n"
            "Closed-form BS Greeks exist; Heston Greeks are obtained via differentiation under the integral or bump-and-reval."
        )
        st.latex(r"\Delta_{\mathrm{call}}^{\mathrm{BS}} = e^{-q\tau}\Phi(d_1),\quad \Gamma^{\mathrm{BS}} = \frac{e^{-q\tau}\phi(d_1)}{S_0\sigma\sqrt{\tau}}")

    # --- Calibration Tips ---
    with st.expander("Calibration Tips (practical)"):
        st.markdown(
            "- Start from **ATM IV** to seed \(v_0\approx \sigma_{\text{ATM}}^2\).\n"
            "- Constrain params to stable ranges (e.g., \(\\rho\\in[-0.99,0.0]\), \(\\kappa>0\), \(\\theta>0\), \(\\sigma_v>0\)).\n"
            "- Use a **robust loss** (e.g., RMSE on IVs) and remove outliers/no-volume strikes.\n"
            "- Inspect residuals by strike/moneyness; systematic shape → re-specify or use term-structure (Heston with time-dependent params)."
        )

    # --- Optional figures if available ---
    with st.expander("Optional figures (auto-detected)"):
        st.markdown("If present in `./assets/`, the app will display them below.")
        show_image_safe("help_example.png", caption="Illustrative figure")
        show_image_safe("return_distr.png", caption="Return distribution illustration")

    # --- Glossary (searchable) ---
    glossary = {
        "moneyness": "Ratio K/F or K/S₀ describing how in/out of the money a strike is.",
        "martingale": "A process with zero conditional drift under a given measure; discounted prices under Q are martingales.",
        "characteristic function": "Fourier transform of a distribution; Heston pricing uses it to compute P1, P2.",
        "feller condition": "Constraint 2κθ ≥ σ_v^2 ensuring v_t stays strictly positive.",
        "implied volatility": "Vol parameter that matches a model price to the observed market price.",
        "risk-neutral measure": "Measure under which discounted asset prices have zero drift.",
        "term structure": "Dependence of model parameters/vol with maturity.",
        "leverage effect": "Negative stock–vol correlation (ρ<0) often seen in equities.",
    }

    if query.strip():
        q = query.strip().lower()
        hits = {k: v for k, v in glossary.items() if q in k.lower() or q in v.lower()}
    else:
        hits = glossary

    st.markdown("### Glossary")
    if not hits:
        st.info("No glossary entries match your search.")
    else:
        for k, v in hits.items():
            st.markdown(f"**{k.capitalize()}** — {v}")

    # Jump button to other pages for hands-on demos
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Open SDE Visualiser ▶"):
            st.session_state["__page_override"] = "🧪 SDE Visualiser"
    with c2:
        if st.button("Open Performance ▶"):
            st.session_state["__page_override"] = "⚡ Performance & Benchmark"
    with c3:
        if st.button("Open Vol Smile ▶"):
            st.session_state["__page_override"] = "📊 Vol Smile Explorer"


# ==================================================
# ============  PAGE 1: SDE VISUALISER  ============
# ==================================================

def simulate_gbm_paths(S0, T, mu, sigma, M, N):
    dt = T / N
    t_local = np.linspace(0, T, N)
    paths = np.zeros((M, N))
    for i in range(M):
        W = np.cumsum(np.random.randn(N)) * np.sqrt(dt)
        paths[i] = S0 * np.exp((mu - 0.5 * sigma**2) * t_local + sigma * W)
    return paths


def simulate_heston_paths(S0, T, r, v0, kappa, theta, sigma_v, rho, M, N, return_variance: bool = False):
    """Euler scheme for Heston. If return_variance=True, also returns the variance paths array (M x N)."""
    dt = T / N
    paths = np.zeros((M, N))
    vars_ = np.zeros((M, N)) if return_variance else None
    for j in range(M):
        S = np.zeros(N)
        v = np.zeros(N)
        S[0], v[0] = S0, max(v0, 1e-12)
        for i in range(1, N):
            z1 = np.random.normal()
            z2 = rho * z1 + np.sqrt(max(1 - rho**2, 0.0)) * np.random.normal()
            v_prev = max(v[i-1], 1e-12)
            v[i] = np.abs(v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * np.sqrt(dt) * z2)
            S[i] = S[i-1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev) * np.sqrt(dt) * z1)
        paths[j] = S
        if return_variance:
            vars_[j] = v
    return (paths, vars_) if return_variance else paths


def page_sde_visualiser():
    st.title("Stochastic Differential Equation Visualiser (Black–Scholes & Heston)")

    # Sidebar parameters
    with st.sidebar:
        st.header("🧮 Global Parameters")
        S0 = st.number_input("Initial Asset Price (S₀)", value=st.session_state.get("S0", 100.0), key="S0")
        K = st.number_input("Strike Price (K)", value=st.session_state.get("K", 100.0), key="K")
        r = st.slider("Risk-Free Rate (r)", 0.0, 0.1, float(st.session_state.get("r", 0.03)), key="r")
        T = st.slider("Time Horizon (Years)", 0.5, 5.0, float(st.session_state.get("T", 1.0)), key="T")
        N = st.slider("Time Steps", 100, 1000, int(st.session_state.get("N", 250)), key="N")
        M = st.slider("Simulations (Monte Carlo)", 10, 2000, int(st.session_state.get("M", 250)), key="M")
        t = np.linspace(0, T, N)

    with st.sidebar.expander("Black–Scholes Parameters"):
        mu = st.slider("Drift (μ)", -0.1, 0.2, float(st.session_state.get("mu", 0.05)), key="mu")
        sigma = st.slider("Volatility (σ)", 0.01, 1.0, float(st.session_state.get("sigma", 0.2)), key="sigma")

    with st.sidebar.expander("Heston Parameters"):
        v0 = st.slider("Initial Variance (v₀)", 0.01, 0.5, float(st.session_state.get("v0", 0.04)), key="v0")
        kappa = st.slider("Mean Reversion (κ)", 0.1, 5.0, float(st.session_state.get("kappa", 2.0)), key="kappa")
        theta = st.slider("Long-Term Variance (θ)", 0.01, 0.5, float(st.session_state.get("theta", 0.04)), key="theta")
        sigma_v = st.slider("Volatility of Volatility (σᵥ)", 0.01, 1.0, float(st.session_state.get("sigma_v", 0.3)), key="sigma_v")
        rho = st.slider("Correlation (ρ)", -1.0, 1.0, float(st.session_state.get("rho", -0.7)), key="rho")

    # Simulations
    gbm_paths = simulate_gbm_paths(S0, T, mu, sigma, M, N)
    heston_paths, heston_vars = simulate_heston_paths(S0, T, r, v0, kappa, theta, sigma_v, rho, M, N, return_variance=True)

    # --- Normalize prices by S0 so S0 ≡ 1 ---
    _eps = 1e-12
    gbm_norm = gbm_paths / max(S0, _eps)
    heston_norm = heston_paths / max(S0, _eps)

    # Layout: two columns
    col1, col2 = st.columns(2)

    # Left column: clearer plots (quantile fan, sample paths, variance)
    with col1:
        st.subheader("📈 Simulated Paths — normalized to S₀")
        logy = st.checkbox("Use log y-scale (prices)", value=False)

        def fan(ax, tvals, paths, label):
            q = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
            ax.plot(tvals, q[2], linewidth=2, label=f"{label} median")
            ax.fill_between(tvals, q[1], q[3], alpha=0.25, label=f"{label} IQR (25–75%)")
            ax.fill_between(tvals, q[0], q[4], alpha=0.12, label=f"{label} 5–95%")

        tab1, tab2, tab3 = st.tabs(["Quantile fan", "Sample paths", "Variance (Heston)"])

        with tab1:
            fig, ax = plt.subplots()
            fan(ax, t, gbm_norm, "GBM")
            fan(ax, t, heston_norm, "Heston")
            ax.set_title("Distribution over time (median, IQR, 5–95%)")
            ax.set_xlabel("Time"); ax.set_ylabel("S(t) / S₀")
            if logy: ax.set_yscale("log")
            ax.grid(True); ax.legend()
            st.pyplot(fig)

        with tab2:
            fig, ax = plt.subplots()
            show = min(12, M)
            idx = np.random.choice(M, show, replace=False)
            for i in idx: ax.plot(t, gbm_norm[i], linewidth=1, alpha=0.7)
            for i in idx: ax.plot(t, heston_norm[i], linewidth=1, alpha=0.7)
            ax.set_title(f"{show} sample paths (normalized)")
            ax.set_xlabel("Time"); ax.set_ylabel("S(t) / S₀")
            if logy: ax.set_yscale("log")
            ax.grid(True)
            st.pyplot(fig)

        with tab3:
            fig, ax = plt.subplots()
            qv = np.quantile(heston_vars, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
            ax.plot(t, qv[2], linewidth=2, label="v(t) median")
            ax.fill_between(t, qv[1], qv[3], alpha=0.25, label="v(t) IQR")
            ax.fill_between(t, qv[0], qv[4], alpha=0.12, label="v(t) 5–95%")
            ax.set_title("Heston variance dynamics v(t)")
            ax.set_xlabel("Time"); ax.set_ylabel("Variance")
            ax.grid(True); ax.legend()
            st.pyplot(fig)

        help_block(
            "Help — why this is clearer",
            "- Fan charts show the **distribution** (median & quantiles), not a potentially misleading ±σ band.\n"
            "- The **variance tab** reveals mean reversion and volatility-of-vol; wider price fans come from higher v(t).\n"
            "- Use **log scale** to compare multiplicative moves fairly."
        )

        # Explicit note about normalization:
        st.caption("Note: All price plots on this page are **normalized by S₀** (so S₀ ≡ 1).")

    # Right column: return distribution plot
    with col2:
        st.subheader("📊 Return Distribution vs Normal Distribution")

        # Terminal log-returns relative to S0
        returns_gbm = np.log(np.maximum(gbm_paths[:, -1], 1e-12) / max(S0, 1e-12))
        returns_heston = np.log(np.maximum(heston_paths[:, -1], 1e-12) / max(S0, 1e-12))

        fig2, ax2 = plt.subplots()
        ax2.hist(returns_gbm, bins=30, alpha=0.5, label="GBM Terminal Log-Returns", density=True)
        ax2.hist(returns_heston, bins=30, alpha=0.5, label="Heston Terminal Log-Returns", density=True)

        x_min = float(min(returns_gbm.min(), returns_heston.min()))
        x_max = float(max(returns_gbm.max(), returns_heston.max()))
        x = np.linspace(x_min, x_max, 200)
        ax2.plot(x, norm.pdf(x, np.mean(returns_gbm), np.std(returns_gbm)), linestyle="--", label="Normal (GBM match)")
        ax2.plot(x, norm.pdf(x, np.mean(returns_heston), np.std(returns_heston)), linestyle="--", label="Normal (Heston match)")

        ax2.set_xlabel("Log Returns")
        ax2.set_ylabel("Density")
        ax2.set_title("Comparison of Terminal Return Distributions (GBM vs Heston)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        help_block(
            "Help — Reading the Return Distributions",
            (
                "- GBM (Black-Scholes) → log-returns follow a Normal distribution by design; histogram should match dashed curve.\n"
                "- Heston → stochastic volatility creates fat tails, meaning more extreme returns than Normal predicts.\n"
                "- Why it matters: fat tails are common in real markets and imply higher risk of large price moves.\n"
                "- How to read: if histogram tails are taller than the dashed curve, the model is producing heavier tails.\n"
                "- Experiment: increase σᵥ or |ρ| to make fat tails more pronounced.\n\n"
            ),
            presets=[
                {"label": "Fatter tails: σᵥ=1.0, ρ=-0.9", "on_click": lambda: set_state(sigma_v=1.0, rho=-0.9)},
                {"label": "Near-GBM: σᵥ=0.05, ρ=0.0", "on_click": lambda: set_state(sigma_v=0.05, rho=0.0)},
            ]
        )


# ==================================================
# =======  PAGE 2: PERFORMANCE & BENCHMARK  ========
# ==================================================

def page_performance():
    st.title("⚡ Performance & Benchmark")
    st.caption("Interactive timing plus a reference chart for larger simulation sizes.")

    # Two columns: left = interactive, right = help/remark with precomputed image
    left, right = st.columns([1.2, 0.8])

    with left:
        st.subheader("Interactive Performance Benchmark")
        bm_M = st.slider("Number of Paths (M)", 10, 5000, st.session_state.get("bm_M", 100), step=10, key="bm_M")
        bm_N = st.slider("Number of Time Steps (N)", 10, 5000, st.session_state.get("bm_N", 100), step=10, key="bm_N")

        # Try to reuse current SDE params if present
        S0 = st.session_state.get("S0", 100.0)
        r = st.session_state.get("r", 0.03)
        mu = st.session_state.get("mu", 0.05)
        sigma = st.session_state.get("sigma", 0.2)
        T = st.session_state.get("T", 1.0)
        v0 = st.session_state.get("v0", 0.04)
        kappa = st.session_state.get("kappa", 2.0)
        theta = st.session_state.get("theta", 0.04)
        sigma_v = st.session_state.get("sigma_v", 0.3)
        rho = st.session_state.get("rho", -0.7)

        if st.button("Run Performance Test"):
            start = time.perf_counter()
            _ = simulate_gbm_paths(S0, T, mu, sigma, M=bm_M, N=bm_N)
            t_gbm = time.perf_counter() - start

            start = time.perf_counter()
            _ = simulate_heston_paths(S0, T, r, v0, kappa, theta, sigma_v, rho, M=bm_M, N=bm_N)
            t_heston = time.perf_counter() - start

            st.success(f"GBM (M={bm_M}, N={bm_N}) took {t_gbm:.3f}s | Heston took {t_heston:.3f}s")
            st.caption("Note: timings vary by hardware and current load; Heston is typically several× slower.")

        help_block(
            "Help — What affects runtime?",
            (
                "- Complexity grows roughly O(M·N).\n"
                "- Heston costs more per step (variance dynamics + correlation).\n"
                "- Use fewer steps for quick exploration; increase M last for stability.\n"
            ),
            presets=[
                {"label": "Fast demo: M=100, N=100", "on_click": lambda: set_state(bm_M=100, bm_N=100)},
                {"label": "Stress: M=2000, N=2000", "on_click": lambda: set_state(bm_M=2000, bm_N=2000)},
            ]
        )

    with right:
        st.subheader("Reference (Precomputed)")
        with st.expander("Show expected scaling remarks & chart"):
            st.markdown(
                "This image illustrates typical scaling for GBM vs Heston as N grows. Use it to gauge ballpark runtimes before launching very large simulations."
            )
            # Robust image display with graceful fallback
            show_image_safe(
                "benchmark_sde_timing.png",
                caption="GBM vs Heston timing vs time steps (if file present)."
            )
            st.markdown(
                "**Rule of thumb:** double either M or N → ≈2× runtime. Doubling both → ≈4×."
            )


# ==================================================
# ==============  PAGE 2: VOL SMILE  ===============
# ==================================================

def page_vol_smile():
    st.title("Volatility Smile Explorer")
    st.caption("This tab uses **real AAPL options data** to compare observed market implied volatility with two models: Black–Scholes (flat σ at-the-money) and the Heston stochastic-volatility model. Heston parameters are **calibrated to the selected trade date and expiry** using the market IVs shown below (QuantLib, if available).")

    # Sidebar I/O
    st.sidebar.subheader("📁 Data")
    dataset_dir = st.sidebar.text_input(
        "Dataset folder",
        value=st.session_state.get("dataset_dir", "/Users/mariachristodoulidou/Desktop/research project"),
        key="dataset_dir"
    )
    filename = st.sidebar.text_input("CSV filename", value=st.session_state.get("csv_filename", "aapl_2016_2020.csv"), key="csv_filename")
    csv_path = os.path.join(dataset_dir, filename)

    st.sidebar.subheader("🎛️ Filters")
    side = st.sidebar.selectbox("Side", ["both", "C", "P"], index=["both", "C", "P"].index(st.session_state.get("vs_side", "both")), key="vs_side")
    r = st.sidebar.number_input("Risk-free rate r", value=float(st.session_state.get("vs_r", 0.02)), step=0.005, format="%.4f", key="vs_r")
    q = st.sidebar.number_input("Dividend yield q", value=float(st.session_state.get("vs_q", 0.00)), step=0.005, format="%.4f", key="vs_q")
    run_heston = st.sidebar.checkbox("Include Heston fit (QuantLib)", value=st.session_state.get("vs_heston", True), key="vs_heston")

    # Load
    try:
        df = load_clean(csv_path)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

    # Validate required columns
    if "QUOTE_DATE" not in df.columns or "EXPIRE_DATE" not in df.columns:
        st.error("Required columns QUOTE_DATE and EXPIRE_DATE are missing.")
        st.stop()

    # Build date/expiry choices
    qd = pd.to_datetime(df["QUOTE_DATE"], errors="coerce").dt.normalize()
    ed = pd.to_datetime(df["EXPIRE_DATE"], errors="coerce").dt.normalize()

    date_options = [pd.Timestamp(x) for x in sorted(qd.dropna().unique().tolist())]
    if not date_options:
        st.error("No valid QUOTE_DATE values in the file.")
        st.stop()

    date_label = lambda d: pd.Timestamp(d).date().isoformat()
    sel_date = st.sidebar.selectbox("Trade date", date_options, format_func=date_label, index=0, key="vs_date")

    expiries_for_day = [pd.Timestamp(x) for x in sorted(ed[qd == sel_date].dropna().unique().tolist())]
    if not expiries_for_day:
        st.warning("No expiries found for that date; choose a different trade date.")
        st.stop()

    expiry_label = lambda d: pd.Timestamp(d).date().isoformat()
    sel_expiry = st.sidebar.selectbox("Expiry date", expiries_for_day, format_func=expiry_label, index=0, key="vs_expiry")

    # Layout: three columns to give more room
    col1, col2, col3 = st.columns([1.1, 1.1, 0.9])

    # ---- Plot 1: Market Smile ----
    with col1:
        st.subheader("Market Volatility Smile")
        sub = _subset(df, sel_date, sel_expiry)
        if sub.empty:
            st.warning("No rows for that date/expiry.")
        elif "STRIKE" not in sub.columns:
            st.error("STRIKE column missing.")
        else:
            fig1, ax1 = plt.subplots()
            plotted = False
            if side in ("C", "both") and "C_IV" in sub.columns:
                ax1.scatter(sub["STRIKE"], sub["C_IV"], s=12, label="Call IV")
                plotted = True
            if side in ("P", "both") and "P_IV" in sub.columns:
                ax1.scatter(sub["STRIKE"], sub["P_IV"], s=12, marker="x", label="Put IV")
                plotted = True
            if plotted:
                ax1.set_xlabel("Strike")
                ax1.set_ylabel("Implied Volatility")
                label_side = {"both": "Call/Put", "C": "Call", "P": "Put"}[side]
                ax1.set_title(f"{label_side} Smile on {pd.Timestamp(sel_date).date()} expiring {pd.Timestamp(sel_expiry).date()}")
                ax1.legend()
                fig1.tight_layout()
                st.pyplot(fig1)

                help_block(
                    "Help — How to read the Market Smile",
                    (
                        "- **Real AAPL data**: each point is a market IV at a strike for the chosen date & expiry.\n"
                        "- A U-shape indicates smile/skew. Black–Scholes would be flat.\n"
                        "- Compare calls vs puts (markers) to spot skew.\n"
                        "\n**Tips**\n\n"
                        "• Narrow to one side if dots overlap.\n"
                        "• If the smile is noisy, filter to larger volumes before export.\n"
                    )
                )
            else:
                st.info("Nothing to plot (missing C_IV/P_IV columns).")

    # ---- Plot 2: Overlay ----
    with col2:
        st.subheader("Overlay: Market vs BS vs Heston")
        if sub.empty or "STRIKE" not in sub.columns or "UNDERLYING_LAST" not in sub.columns:
            st.info("Not enough data to build overlay plot.")
        else:
            iv_col = "C_IV" if side in ("C", "both") else "P_IV"
            if iv_col not in sub.columns:
                st.info(f"Column {iv_col} not found.")
            else:
                sub_iv = sub.loc[sub[iv_col].notna() & np.isfinite(sub[iv_col])]
                if sub_iv.empty:
                    st.info("No finite market IVs to plot.")
                else:
                    S0 = float(np.nanmedian(sub_iv["UNDERLYING_LAST"]))
                    # BS flat-σ at ATM
                    try:
                        sigma_bs = _atm_iv(sub_iv, S0)
                    except Exception as e:
                        st.warning(f"ATM IV estimate failed: {e}")
                        sigma_bs = float(np.nanmedian(sub_iv[iv_col]))

                    x = sub_iv["STRIKE"].values
                    y = sub_iv[iv_col].values

                    # Optional Heston (QuantLib)
                    heston_points = None
                    heston_params = None
                    heston_rmse = np.nan

                    if run_heston:
                        with st.spinner("Calibrating Heston (QuantLib)…"):
                            try:
                                import QuantLib as ql
                                calendar = ql.TARGET()
                                dc = ql.Actual365Fixed()
                                d0 = pd.Timestamp(sel_date)
                                e0 = pd.Timestamp(sel_expiry)
                                todays_date = ql.Date(d0.day, d0.month, d0.year)
                                exercise_date = ql.Date(e0.day, e0.month, e0.year)
                                ql.Settings.instance().evaluationDate = todays_date

                                spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
                                r_ts = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, float(r), dc))
                                q_ts = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, float(q), dc))

                                days_to_expiry = exercise_date - todays_date
                                maturity = ql.Period(int(days_to_expiry), ql.Days)

                                helpers = []
                                for _, row in sub_iv.iterrows():
                                    K = float(row["STRIKE"])
                                    vol = float(row[iv_col])
                                    if not np.isfinite(vol) or vol <= 0:
                                        continue
                                    quote = ql.QuoteHandle(ql.SimpleQuote(vol))
                                    helpers.append(ql.HestonModelHelper(maturity, calendar, float(S0), K, quote, r_ts, q_ts))
                                if helpers:
                                    v0    = max(sigma_bs**2, 1e-6)
                                    kappa = 1.5; theta = v0; sigma = 0.5; rho = -0.5
                                    process = ql.HestonProcess(r_ts, q_ts, spot_handle, v0, kappa, theta, sigma, rho)
                                    model   = ql.HestonModel(process)
                                    engine_for_helpers = ql.AnalyticHestonEngine(model)
                                    for h in helpers:
                                        h.setPricingEngine(engine_for_helpers)
                                    om = ql.LevenbergMarquardt()
                                    model.calibrate(helpers, om, ql.EndCriteria(400, 50, 1e-8, 1e-8, 1e-8))

                                    # Invert to BS IVs on same strikes
                                    init_vol = ql.BlackVolTermStructureHandle(
                                        ql.BlackConstantVol(todays_date, calendar, max(sigma_bs, 1e-4), dc)
                                    )
                                    bs_process = ql.BlackScholesMertonProcess(spot_handle, q_ts, r_ts, init_vol)
                                    opt_engine = ql.AnalyticHestonEngine(model)
                                    opt_type = ql.Option.Call if side in ("C", "both") else ql.Option.Put

                                    pts = []
                                    for _, row in sub_iv.iterrows():
                                        K = float(row["STRIKE"])
                                        payoff   = ql.PlainVanillaPayoff(opt_type, K)
                                        exercise = ql.EuropeanExercise(exercise_date)
                                        opt = ql.VanillaOption(payoff, exercise)
                                        opt.setPricingEngine(opt_engine)
                                        price = opt.NPV()
                                        try:
                                            iv = opt.impliedVolatility(price, bs_process, 1e-7, 500, 1e-6, 4.0)
                                        except RuntimeError:
                                            iv = np.nan
                                        if np.isfinite(iv) and iv > 0:
                                            pts.append((K, float(iv)))
                                    if pts:
                                        heston_points = sorted(pts, key=lambda p: p[0])

                                    try:
                                        kappa_cal, theta_cal, sigma_cal, rho_cal, v0_cal = [float(x) for x in model.params()]
                                        heston_params = dict(
                                            v0=v0_cal, kappa=kappa_cal, theta=theta_cal, sigma=sigma_cal, rho=rho_cal
                                        )
                                        if heston_points:
                                            mkt_by_k = sub_iv.groupby("STRIKE")[iv_col].median().to_dict()
                                            market_ivs = []
                                            model_ivs = []
                                            for K, iv_model in heston_points:
                                                if K in mkt_by_k and np.isfinite(mkt_by_k[K]) and mkt_by_k[K] > 0:
                                                    market_ivs.append(float(mkt_by_k[K]))
                                                    model_ivs.append(float(iv_model))
                                            if market_ivs:
                                                market_ivs = np.array(market_ivs)
                                                model_ivs = np.array(model_ivs)
                                                heston_rmse = float(np.sqrt(np.mean((market_ivs - model_ivs) ** 2)))
                                    except Exception as e:
                                        st.info(f"Heston calibration insights unavailable: {e}")

                            except ImportError:
                                st.info("QuantLib not installed. Skipping Heston. (pip install QuantLib-Python)")
                            except Exception as e:
                                st.info(f"Heston calibration skipped: {e}")

                    # Plot overlay
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(x, y, s=12, label=f"Market {'Call' if side in ('C','both') else 'Put'} IV")
                    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                    ax2.plot(x_line, np.full_like(x_line, sigma_bs), linestyle="--", label=f"BS fit (σ_ATM≈{sigma_bs:.3f})")
                    if heston_points:
                        xh = np.array([p[0] for p in heston_points])
                        yh = np.array([p[1] for p in heston_points])
                        ax2.plot(xh, yh, label="Heston fit")

                    rmse_str = f" | Heston RMSE={heston_rmse:.4f}" if np.isfinite(heston_rmse) else ""
                    ax2.set_xlabel("Strike"); ax2.set_ylabel("Implied Volatility")
                    ax2.set_title(
                        f"Market vs BS vs Heston{rmse_str}\n{pd.Timestamp(sel_date).date()} → {pd.Timestamp(sel_expiry).date()}"
                    )
                    ax2.legend()
                    fig2.tight_layout()
                    st.pyplot(fig2)

                    help_body = (
                        "**What you see**\n\n"
                        "• Blue dots: market IVs.\n"
                        "• Dashed line: flat BS σ at ATM.\n"
                        "• Heston curve (if enabled): model IVs from calibrated parameters.\n\n"
                        "**Practical notes**\n\n"
                        "• Use risk-free r and dividend q to reflect the trade date.\n"
                        "• RMSE is in absolute IV units (e.g., 0.02 = 2 vol points).\n"
                    )

                    presets = [
                        {"label": "Preset: Calls, r=2%, q=0%", "on_click": lambda: set_state(vs_side="C", vs_r=0.02, vs_q=0.00)},
                        {"label": "Preset: Puts, r=0.5%, q=0.5%", "on_click": lambda: set_state(vs_side="P", vs_r=0.005, vs_q=0.005)},
                        {"label": "Preset: Both sides, r=3%, q=1%", "on_click": lambda: set_state(vs_side="both", vs_r=0.03, vs_q=0.01)},
                    ]
                    help_block("Help — Interpreting Overlay & Calibration", help_body, presets)

                    if heston_params is not None:
                        st.markdown("### Heston Calibration Insights / Params")
                        st.write(f"**v₀** (Initial variance): {heston_params['v0']:.6f}")
                        st.write(f"**κ** (Mean reversion speed): {heston_params['kappa']:.6f}")
                        st.write(f"**θ** (Long-run variance): {heston_params['theta']:.6f}")
                        st.write(f"**σ** (Vol-of-vol): {heston_params['sigma']:.6f}")
                        st.write(f"**ρ** (Spot/Vol correlation): {heston_params['rho']:.6f}")
                        if np.isfinite(heston_rmse):
                            st.write(f"**Calibration RMSE** (Market IV vs Heston IV): {heston_rmse:.6f}")
                        else:
                            st.write("**Calibration RMSE**: n/a")
                        st.caption(
                            f"Calibrated on {len(heston_points) if heston_points else 0} model points; RMSE is absolute IV units."
                        )

    # ---- Help & Guided Example (right column) ----
    with col3:
        st.subheader("📘 Help & Guided Example")
        with st.expander("Show guide"):
            st.write(
                "This guide walks through a minimal example so you can interpret the two plots."
            )
            example_date = pd.Timestamp("2019-01-02")
            example_expiry = pd.Timestamp("2019-01-04")
            st.markdown(
                f"**Example setup**  \n"
                f"• Trade date = {example_date.date()}  \n"
                f"• Expiry date = {example_expiry.date()}  \n"
                f"• Side = Call (C)  \n"
                f"• r = 0.02, q = 0.00"
            )
            example_sub = _subset(df, example_date, example_expiry)
            if example_sub.empty:
                st.info("Example data not found in this dataset — pick any visible date/expiry on the left.")
            else:
                fig_ex1, ax_ex1 = plt.subplots()
                if "C_IV" in example_sub.columns:
                    ax_ex1.scatter(example_sub["STRIKE"], example_sub["C_IV"], s=12, label="Call IV")
                ax_ex1.set_xlabel("Strike"); ax_ex1.set_ylabel("Implied Volatility"); ax_ex1.legend()
                ax_ex1.set_title("Market Volatility Smile (Example)")
                st.pyplot(fig_ex1)
                st.markdown(
                    "**Reading tips:** The U-shape is the smile; BS assumes flat σ, so deviations show skew/smile."
                )

                S0 = float(np.nanmedian(example_sub["UNDERLYING_LAST"])) if "UNDERLYING_LAST" in example_sub.columns else np.nan
                try:
                    sigma_bs = _atm_iv(example_sub, S0)
                except Exception:
                    sigma_bs = float(np.nanmedian(example_sub.get("C_IV", pd.Series([np.nan]))))

                x = example_sub["STRIKE"].values
                y = example_sub.get("C_IV", pd.Series(np.full_like(x, np.nan))).values
                fig_ex2, ax_ex2 = plt.subplots()
                ax_ex2.scatter(x, y, s=12, label="Market Call IV")
                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                if np.isfinite(sigma_bs):
                    ax_ex2.plot(x_line, np.full_like(x_line, sigma_bs), linestyle="--", label=f"BS fit (σ≈{sigma_bs:.3f})")
                ax_ex2.set_xlabel("Strike"); ax_ex2.set_ylabel("Implied Volatility"); ax_ex2.legend()
                ax_ex2.set_title("Market vs BS (Example)")
                st.pyplot(fig_ex2)

    st.caption("Tip: toggle the ‘Include Heston fit’ switch if QuantLib isn’t installed, or tweak r and q to see sensitivity.")



# ==================================================
# ================  PAGE ROUTER  ===================
# ==================================================

with st.sidebar:
    st.markdown("---")
    # Now includes the Theory page first
    page = st.radio(
        "Navigate",
        ["📚 Theory & Notes", "🧪 SDE Visualiser", "⚡ Performance & Benchmark", "📊 Vol Smile Explorer"],
        index=0,
    )

# Allow internal jump buttons to switch page this run
if "__page_override" in st.session_state:
    page = st.session_state.pop("__page_override")

if page.startswith("📚"):
    page_theory()
elif page.startswith("📊"):
    page_vol_smile()
elif page.startswith("🧪"):
    page_sde_visualiser()
else:
    page_performance()
