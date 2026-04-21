import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="Monte Carlo Option Pricing", layout="wide")

# --- GLOBAL CSS STYLING ---
st.markdown("""
    <style>
    /* Global Font Style: Times New Roman for professional academic look */
    * {
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* Force Tabs to Stretch Full Width */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
        justify-content: center;
        font-size: 18px;
    }

    /* Subtle styling for theory containers */
    .theory-container {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 30px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
        color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE MANAGEMENT ---
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

def trigger_analysis():
    st.session_state.run_analysis = True

def reset_app():
    st.session_state.run_analysis = False

# --- MATH & LOGIC FUNCTIONS ---
@st.cache_data
def calculate_data(ticker):
    df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
    prices = df["Close"].dropna().squeeze()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    mu_daily = float(log_returns.mean())
    sigma_daily = float(log_returns.std())
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)

    S0 = float(prices.iloc[-1])
    dt = 1 / 252
    return prices, log_returns, mu_daily, sigma_daily, mu_annual, sigma_annual, S0, dt

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price, d1, d2

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price, d1, d2

def bs_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

def simulate_gbm(S0, r, sigma, T, dt, n_paths):
    n_steps = int(T / dt)
    Z = np.random.standard_normal((n_steps, n_paths))
    dW = np.sqrt(dt) * Z
    step = ((r - (0.5 * sigma ** 2)) * dt) + (sigma * dW)
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(step, axis=0)])
    return S0 * np.exp(log_paths)

def monte_carlo_option(S0, K, T, r, sigma, dt, n_paths, option_type="call", n_batches=10):
    payoffs = []
    batch_size = n_paths // n_batches
    for _ in range(n_batches):
        paths = simulate_gbm(S0, r, sigma, T, dt, batch_size)
        S_T = paths[-1]
        if option_type == "call":
            payoff = np.maximum(S_T - K, 0)
        else:
            payoff = np.maximum(K - S_T, 0)
        payoffs.append(np.exp(-r * T) * payoff)
    all_payoffs = np.concatenate(payoffs)
    return all_payoffs.mean(), all_payoffs.std() / np.sqrt(len(all_payoffs))

# ==========================================
# VIEW 1: LANDING / HOME PAGE
# ==========================================
if not st.session_state.run_analysis:

    st.markdown("""
        <style>
        .stApp {
            background-color: #0f1419;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: white; margin-top: 20px;'>Monte Carlo Option Pricing Model</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #b3c5d6;'>Group 16 — Financial Engineering Dashboard</h4><br>", unsafe_allow_html=True)

    # Input Form Centered
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.08); padding: 30px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2);">
            <h3 style='color: white; text-align: center; margin-bottom: 20px;'>Model Configuration Parameters</h3>
        """, unsafe_allow_html=True)

        st.session_state.ticker_input = st.text_input("Underlying Asset Ticker Symbol", value="NATIONALUM.NS")
        st.session_state.r_input = st.number_input("Annual Risk-Free Rate (r)", value=0.037, format="%.3f")
        st.session_state.T_input = st.number_input("Time to Expiration in Years (T)", value=0.25, format="%.2f")
        st.session_state.paths_input = st.slider("Number of Monte Carlo Simulation Paths", min_value=100, max_value=100000, step=100, value=10000)

        st.write("")
        st.button("Execute Model Analysis", type="primary", use_container_width=True, on_click=trigger_analysis)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# VIEW 2: POST-RUN DASHBOARD
# ==========================================
else:
    # --- TOP RIGHT HOME BUTTON ---
    col_empty, col_home = st.columns([5,1])
    with col_home:
        st.button("Home", on_click=reset_app, use_container_width=True)

    # --- DASHBOARD BANNER IMAGE STRIP ---
    DASHBOARD_BANNER_URL = "https://iitgn.ac.in/assets/img/IITGN-5.png"

    st.markdown(f"""
        <div style="
            width: 100%;
            height: 140px;
            background-image: url('{DASHBOARD_BANNER_URL}');
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            border-radius: 8px;
            margin-bottom: 20px;
            margin-top: -10px;
        "></div>
    """, unsafe_allow_html=True)

    with st.spinner("Processing computational models and executing Monte Carlo iterations..."):

        prices, log_returns, mu_daily, sigma_daily, mu_annual, sigma_annual, S0, dt = calculate_data(
            st.session_state.ticker_input
        )

        np.random.seed(42)
        strikes = np.round(np.arange(0.50, 1.5, 0.05) * S0, 2)
        results = []

        for K in strikes:
            bs_call, _, _ = black_scholes_call(S0, K, st.session_state.T_input, st.session_state.r_input, sigma_annual)
            mc_call, mc_se_call = monte_carlo_option(S0, K, st.session_state.T_input, st.session_state.r_input, sigma_annual, dt, st.session_state.paths_input, "call")

            bs_put, _, _ = black_scholes_put(S0, K, st.session_state.T_input, st.session_state.r_input, sigma_annual)
            mc_put, mc_se_put = monte_carlo_option(S0, K, st.session_state.T_input, st.session_state.r_input, sigma_annual, dt, st.session_state.paths_input, "put")

            moneyness = "ATM" if abs(K - S0) < 0.05 * S0 else ("ITM" if K < S0 else "OTM")
            results.append({
                "Strike": K, "Moneyness": moneyness,
                "BS Call": bs_call, "MC Call": mc_call, "MC Call SE": mc_se_call,
                "BS Put": bs_put, "MC Put": mc_put, "MC Put SE": mc_se_put
            })

        results_df = pd.DataFrame(results)

    # --- TOP NAVIGATION BAR (TABS) ---
    tab_theory, tab_market, tab_table, tab_greeks, tab_charts = st.tabs([
        "Theory and Methodology",
        "Market Parameters",
        "Pricing Matrix",
        "Option Greeks",
        "Visualizations"
    ])

    with tab_theory:
        # 1. Theory of Geometric Brownian Motion (GBM)
        st.header("1. Theory of Geometric Brownian Motion (GBM)")

        st.subheader("1.1. Introduction")
        st.write(r"""
        Geometric Brownian Motion (GBM) is a continuous-time stochastic process in which the logarithm of the randomly varying quantity follows a Brownian motion (also known as a Wiener process) with drift. It is widely used in mathematical finance, most notably as the standard model for asset prices in the Black-Scholes framework, because it ensures that asset prices remain strictly positive.
        """)

        st.subheader("1.2. The Stochastic Differential Equation (SDE)")
        st.write(r"""
        A stochastic process $S_t$ is said to follow a Geometric Brownian Motion if it satisfies the following stochastic differential equation:

        $$ dS_t = \mu S_t \, dt + \sigma S_t \, dW_t $$

        **Variable Definitions:**
        * **$S_t$:** The value of the process (e.g., the stock price) at time $t$.
        * **$dS_t$:** The infinitesimal change in the process value over a tiny time interval.
        * **$\mu$:** The percentage drift rate. It is a deterministic constant representing the expected rate of return of the asset per unit of time.
        * **$\sigma$:** The percentage volatility. It is a strictly positive constant representing the standard deviation of the asset's returns, characterizing the magnitude of random fluctuations.
        * **$t$:** Time.
        * **$dt$:** An infinitesimal increment in time.
        * **$W_t$:** A standard Brownian motion or Wiener process.
        * **$dW_t$:** The increment of the Wiener process. It is a random variable that is normally distributed with a mean of $0$ and a variance of $dt$, denoted as $dW_t \sim \mathcal{N}(0, dt)$.
        """)

        st.subheader("1.3. Solving the SDE using Itô's Lemma")
        st.write(r"""
        To find the explicit, analytical solution for $S_t$, standard calculus cannot be used because $W_t$ is nowhere differentiable. Instead, Itô's Lemma must be applied to the natural logarithm of the process. Let $f(S_t, t) = \ln(S_t)$.

        Itô's Lemma states that for a twice-differentiable function $f(S_t, t)$, the differential $df$ is given by:

        $$ df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial S} dS_t + \frac{1}{2} \frac{\partial^2 f}{\partial S^2} (dS_t)^2 $$

        For the function $f(S_t) = \ln(S_t)$, the partial derivatives are calculated as follows:
        """)

        st.markdown("Itô's Lemma for $f(S_t) = \\ln S_t$")
        st.latex(r"""
        \begin{aligned}
        \frac{\partial f}{\partial t} &= 0 \\
        \frac{\partial f}{\partial S} &= \frac{1}{S_t} \\
        \frac{\partial^2 f}{\partial S^2} &= -\frac{1}{S_t^2}
        \end{aligned}
        """)

        st.markdown("Quadratic variation")

        st.latex(r"""
        (dS_t)^2 = (\mu S_t dt + \sigma S_t dW_t)^2 = \sigma^2 S_t^2 dt
        """)

        st.markdown("Applying Itô's Lemma")

        st.latex(r"""
        \begin{aligned}
        d(\ln S_t) &= 0 \cdot dt 
        + \frac{1}{S_t} (\mu S_t dt + \sigma S_t dW_t) \\
        &\quad - \frac{1}{2} \frac{1}{S_t^2} (\sigma^2 S_t^2 dt) \\
        &= \mu dt + \sigma dW_t - \frac{1}{2} \sigma^2 dt \\
        &= \left( \mu - \frac{\sigma^2}{2} \right) dt + \sigma dW_t
        \end{aligned}
        """)

        st.markdown("### Integrating from $0$ to $t$")

        st.latex(r"""
        \int_0^t d(\ln S_u)
        =
        \int_0^t \left( \mu - \frac{\sigma^2}{2} \right) du
        +
        \int_0^t \sigma dW_u
        """)

        st.markdown("### Simplifying")

        st.latex(r"""
        \ln(S_t) - \ln(S_0)
        =
        \left( \mu - \frac{\sigma^2}{2} \right) t + \sigma W_t
        """)

        st.markdown("### Final solution")

        st.latex(r"""
        S_t = S_0 \exp\left(
        \left( \mu - \frac{\sigma^2}{2} \right)t + \sigma W_t
        \right)
        """)

        st.subheader("1.4. Statistical Properties")
        st.write(r"""
        Because the standard Brownian motion $W_t$ is normally distributed such that $W_t \sim \mathcal{N}(0, t)$, it follows that $\ln(S_t)$ is also normally distributed:

        $$ \ln(S_t) \sim \mathcal{N}\left( \ln(S_0) + \left( \mu - \frac{\sigma^2}{2} \right)t, \, \sigma^2 t \right) $$

        Given that the logarithm of $S_t$ is normally distributed, the asset price $S_t$ itself follows a **log-normal distribution**. The expected value (mean) and variance of the log-normally distributed process $S_t$ are evaluated as:
        """)

        st.latex(r"""
        \begin{aligned}
        \mathbb{E}[S_t] &= S_0 e^{\mu t} \\
        \text{Var}(S_t) &= S_0^2 e^{2\mu t} \left( e^{\sigma^2 t} - 1 \right)
        \end{aligned}
        """)

        st.divider()

        # 2. Black-Scholes Theory
        st.header("2. Theory of the Black-Scholes Model")

        st.subheader("2.1. Introduction")
        st.write(r"""
        The Black-Scholes model is a mathematical framework for pricing an options contract. Building upon the assumption that the underlying asset follows a Geometric Brownian Motion (GBM), the model provides a theoretical estimate of the price of European-style options. It demonstrates that a perfectly hedged portfolio can eliminate market risk, implying that the option's value depends on the risk-free rate rather than the asset's expected return.
        """)

        st.subheader("2.2. Underlying Assumptions")
        st.write(r"""
        The derivation relies on several key market assumptions:
        * The underlying asset price $S_t$ follows a Geometric Brownian Motion: $dS_t = \mu S_t dt + \sigma S_t dW_t$.
        * The risk-free interest rate $r$ and the asset's volatility $\sigma$ are known and strictly constant.
        * The underlying asset pays no dividends during the life of the option.
        * There are no transaction costs or taxes (frictionless market).
        * It is possible to short-sell the underlying asset without restriction.
        * The option is European, meaning it can only be exercised at expiration $T$.
        """)

        st.subheader("2.3. The Black-Scholes PDE")
        st.write(r"""
        To price a derivative $V(S, t)$, we construct a risk-free portfolio $\Pi$ consisting of one short position in the derivative and a long position in $\Delta$ shares of the underlying asset:

        $$ \Pi = -V + \Delta S $$

        The infinitesimal change in the value of this portfolio is:

        $$ d\Pi = -dV + \Delta dS $$

        By applying Itô's Lemma to the derivative price $V(S, t)$, its differential is:

        $$ dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} dt $$

        Substituting $dV$ into the portfolio change equation yields:

        $$ d\Pi = -\left( \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt + \left( \Delta - \frac{\partial V}{\partial S} \right) dS $$

        To make the portfolio perfectly risk-free, the stochastic term $dS$ must be eliminated. This is achieved by continuously choosing $\Delta$ such that:

        $$ \Delta = \frac{\partial V}{\partial S} $$

        This is known as delta hedging. The change in the risk-free portfolio simplifies to:

        $$ d\Pi = -\left( \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt $$

        Because the portfolio is risk-free, the no-arbitrage principle dictates that it must earn exactly the risk-free interest rate $r$. Therefore, $d\Pi = r\Pi dt$:

        $$ -\left( \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt = r(-V + \Delta S) dt $$

        Dividing by $dt$ and substituting $\Delta = \frac{\partial V}{\partial S}$, the fundamental **Black-Scholes Partial Differential Equation (PDE)** is obtained:

        $$ \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0 $$
        """)

        st.subheader("2.4. The Black-Scholes Pricing Formulas")
        st.write(r"""
        The PDE can be solved subject to specific boundary conditions at expiration $T$. For a European Call option $C(S,t)$, the payoff at expiration is $\max(S_T - K, 0)$. For a European Put option $P(S,t)$, the payoff is $\max(K - S_T, 0)$.

        **Call Option Formula:**
        $$ C(S, t) = S \cdot N(d_1) - K \cdot e^{-r(T-t)} \cdot N(d_2) $$

        **Put Option Formula:**
        $$ P(S, t) = K \cdot e^{-r(T-t)} \cdot N(-d_2) - S \cdot N(-d_1) $$

        **Variable Definitions:**
        Where the intermediate variables $d_1$ and $d_2$ are defined as:
        """)
        st.latex(r"""
        \begin{aligned}
        d_1 &= \frac{\ln\left(\frac{S}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)(T-t)}{\sigma \sqrt{T-t}} \\
        d_2 &= d_1 - \sigma \sqrt{T-t}
        \end{aligned}
        """)

        # Text explanation
        st.markdown("""
        - **$C(S, t), P(S, t)$:** Price of the European call and put option
        - **$S$:** Current price of the underlying asset
        - **$K$:** Strike (exercise) price
        - **$T - t$:** Time to expiration (years)
        - **$r$:** Risk-free interest rate
        - **$\\sigma$:** Volatility
        - **$N(\\cdot)$:** Standard normal CDF
        """)
        st.divider()

        # 3. Monte Carlo Theory
        st.header("3. Theory of Monte Carlo Simulations for Option Pricing")

        st.subheader("3.1. Introduction")
        st.write(r"""
        Monte Carlo simulation is a computational algorithm that relies on repeated random sampling to obtain numerical results. In mathematical finance, it is primarily used to price complex derivatives where analytical solutions (like the Black-Scholes formula) do not exist. The method works by simulating thousands or millions of possible future paths for the underlying asset, calculating the option payoff for each path, and then computing the discounted average of these payoffs.
        """)

        st.subheader("3.2. Asset Price Dynamics and Discretization")
        st.write(r"""
        To simulate the future paths of an asset $S_t$, we typically assume it follows a Geometric Brownian Motion (GBM). Under the risk-neutral measure $\mathbb{Q}$, the real-world drift rate $\mu$ is replaced by the risk-free interest rate $r$. The risk-neutral Stochastic Differential Equation (SDE) is:

        $$ dS_t = r S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}} $$

        Using Itô's Lemma, the exact analytical solution for the asset price at a future time $t + \Delta t$, given its price at time $t$, is:

        $$ S_{t+\Delta t} = S_t \exp\left( \left( r - \frac{\sigma^2}{2} \right)\Delta t + \sigma \left( W_{t+\Delta t} - W_t \right) \right) $$

        Because the increment of a Wiener process $W_{t+\Delta t} - W_t$ is normally distributed with mean $0$ and variance $\Delta t$, we can express it as $\sqrt{\Delta t} \, Z$, where $Z$ is a standard normal random variable. This yields the **discrete-time simulation equation**:

        $$ S_{t+\Delta t} = S_t \exp\left( \left( r - \frac{\sigma^2}{2} \right)\Delta t + \sigma \sqrt{\Delta t} \, Z \right) $$

        For a European option that only depends on the final price at expiration $T$, we can simulate the final price directly in a single step (setting $\Delta t = T$ and $t = 0$):

        $$ S_T = S_0 \exp\left( \left( r - \frac{\sigma^2}{2} \right)T + \sigma \sqrt{T} \, Z \right) $$
        """)

        st.subheader("3.3. Risk-Neutral Valuation")
        st.write(r"""
        The Fundamental Theorem of Asset Pricing states that the current value of a derivative $V_0$ is the expected value of its future discounted payoff under the risk-neutral probability measure $\mathbb{Q}$:

        $$ V_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[f(S_T)] $$

        Where $f(S_T)$ is the payoff function of the option (e.g., $f(S_T) = \max(S_T - K, 0)$ for a call option).
        """)

        st.subheader("3.4. The Monte Carlo Estimator")
        st.write(r"""
        Since calculating the exact expectation is often impossible for complex derivatives, we estimate it using the Law of Large Numbers. By generating $N$ independent random samples of the standard normal variable $Z$, we create $N$ simulated terminal asset prices $S_T^{(i)}$.

        The Monte Carlo estimator for the option price is the sample mean of the discounted payoffs:

        $$ \hat{V}_0 = e^{-rT} \frac{1}{N} \sum_{i=1}^{N} f\left(S_T^{(i)}\right) $$

        As the number of simulations $N \to \infty$, the estimator $\hat{V}_0$ converges to the true option value $V_0$.
        """)

        st.subheader("3.5. Standard Error and Convergence")
        st.write(r"""
        Because $\hat{V}_0$ is a statistical estimate, it contains sampling error. The standard deviation of the simulated payoffs is:

        $$ s = \sqrt{ \frac{1}{N-1} \sum_{i=1}^{N} \left( f\left(S_T^{(i)}\right) - \bar{f} \right)^2 } $$

        where $\bar{f}$ is the sample mean of the un-discounted payoffs.

        The accuracy of the Monte Carlo estimate is given by the **Standard Error (SE)**, which determines the confidence interval of the estimated price:

        $$ \text{SE} = \frac{e^{-rT} \cdot s}{\sqrt{N}} $$

        Notice that the error decays at a rate of $1/\sqrt{N}$. To halve the error, one must quadruple the number of simulations.

        **Variable Definitions:**
        * **$S_t$:** Price of the underlying asset at time $t$.
        * **$S_T^{(i)}$:** The $i$-th simulated final asset price at expiration $T$.
        * **$V_0$:** True theoretical price of the option at time $t=0$.
        * **$\hat{V}_0$:** Monte Carlo estimated price of the option.
        * **$r$:** Annualized continuously compounded risk-free interest rate.
        * **$\sigma$:** Volatility of the underlying asset.
        * **$T$:** Time to expiration (in years).
        * **$\Delta t$:** Time step size used in the simulation.
        * **$Z$:** A random variable drawn from a standard normal distribution, $Z \sim \mathcal{N}(0, 1)$.
        * **$N$:** The total number of simulation paths generated.
        * **$f(S_T)$:** The payoff function of the derivative at expiration.
        * **$\text{SE}$:** The standard error of the Monte Carlo estimate.
        """)

    with tab_market:
        st.subheader("Empirical Market Parameters")
        st.write(f"Parameters extracted from 1-year historical pricing data for **{st.session_state.ticker_input}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spot Price (S₀)", f"Rs {S0:,.2f}")
        m2.metric("Annualized Volatility (σ)", f"{sigma_annual*100:.2f}%")
        m3.metric("Annualized Drift (μ)", f"{mu_annual*100:.2f}%")
        m4.metric("Risk-Free Rate (r)", f"{st.session_state.r_input*100:.2f}%")

        fig = plt.figure(figsize=(14, 5))
        gs  = gridspec.GridSpec(1, 2, hspace=0.38, wspace=0.30)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(prices.index, prices.values, color="blue", lw=1.5, label=f"{st.session_state.ticker_input} Close")
        ax1.axhline(S0, color="red", ls="--", lw=1.2, label=f"Latest: Rs{S0:.2f}")
        ax1.set_title(f"{st.session_state.ticker_input} Historical Closing Price (1 Year)", fontsize=13)
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(log_returns, bins=50, color="blue", edgecolor="white", density=True, label="Log Returns")
        xr = np.linspace(log_returns.min(), log_returns.max(), 2000)
        ax2.plot(xr, norm.pdf(xr, mu_daily, sigma_daily), color="red", lw=2, label="Normal Fit")
        ax2.set_title("Daily Logarithmic Return Distribution", fontsize=13)
        ax2.legend()
        st.pyplot(fig)

        st.markdown("""
        ---
        **Analytical Summary:** The left subplot illustrates the historical trajectory of the asset over the preceding year. The right subplot presents a histogram of the asset's daily logarithmic returns, overlaid with a theoretical normal probability density function. This visualizes the degree to which the empirical data aligns with the log-normal assumptions required by standard pricing models.
        """)

    with tab_table:
        st.subheader("Options Pricing Summary Matrix")
        st.dataframe(
            results_df.style.format({
                "BS Call": "{:.2f}", "MC Call": "{:.2f}", "MC Call SE": "{:.4f}",
                "Strike": "{:.2f}", "BS Put": "{:.2f}", "MC Put": "{:.2f}", "MC Put SE": "{:.4f}"
            }),
            use_container_width=True
        )

        fig = plt.figure(figsize=(14, 5))
        gs  = gridspec.GridSpec(1, 2, figure=fig, hspace=0.38, wspace=0.30)

        ax5 = fig.add_subplot(gs[0, 0])
        ax5.plot(results_df["Strike"], results_df["BS Call"], color="blue", marker="o", lw=2, label="Black-Scholes Call")
        ax5.plot(results_df["Strike"], results_df["MC Call"], color="orange", marker="s", lw=2, ls="--", label="Monte Carlo Call")
        ax5.fill_between(results_df["Strike"], results_df["MC Call"] - 1.96 * results_df["MC Call SE"], results_df["MC Call"] + 1.96 * results_df["MC Call SE"], alpha=0.20, color="orange")
        ax5.axvline(S0, color="red", ls=":", label=f"Spot Price = Rs{S0:.0f}")
        ax5.set_title("European Call Price: Analytical vs Simulated", fontsize=13)
        ax5.legend()

        ax6 = fig.add_subplot(gs[0, 1])
        ax6.plot(results_df["Strike"], results_df["BS Put"], color="purple", marker="o", lw=2, label="Black-Scholes Put")
        ax6.plot(results_df["Strike"], results_df["MC Put"], color="green", marker="s", lw=2, ls="--", label="Monte Carlo Put")
        ax6.fill_between(results_df["Strike"], results_df["MC Put"] - 1.96 * results_df["MC Put SE"], results_df["MC Put"] + 1.96 * results_df["MC Put SE"], alpha=0.20, color="green")
        ax6.axvline(S0, color="red", ls=":", label=f"Spot Price = Rs{S0:.0f}")
        ax6.set_title("European Put Price: Analytical vs Simulated", fontsize=13)
        ax6.legend()
        st.pyplot(fig)

        st.markdown("""
        ---
        **Analytical Summary:** These charts compare the option premiums calculated via the analytical Black-Scholes formula against the simulated Monte Carlo results across a range of strike prices. The shaded regions denote the 95% Confidence Interval corresponding to the Monte Carlo standard error. As expected, call values decline monotonically with increasing strike prices, while put values increase.
        """)

    with tab_greeks:
        K_atm = round(S0)
        greeks = bs_greeks(S0, K_atm, st.session_state.T_input, st.session_state.r_input, sigma_annual)
        st.subheader(f"First and Second-Order Sensitivity Measures (ATM Strike = {K_atm})")

        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Delta", f"{greeks['Delta']:.4f}")
        g2.metric("Gamma", f"{greeks['Gamma']:.4f}")
        g3.metric("Vega", f"{greeks['Vega']:.4f}")
        g4.metric("Theta", f"{greeks['Theta']:.4f}")
        g5.metric("Rho", f"{greeks['Rho']:.4f}")

        st.write("") # Spacing
        st.subheader("The Black-Scholes Greeks: Theoretical Foundations")
        st.write(r"""
        The "Greeks" represent the sensitivity of an option's price ($V$) to various underlying variables. Below are the partial differential formulas representing each metric.
        """)

        st.divider()

        st.write(r"**1. Delta ($\Delta$)**")
        st.latex(r"\Delta = \frac{\partial V}{\partial S}")
        st.write("**Meaning:** The rate of change of the option price with respect to changes in the underlying asset's price.")

        st.write(r"**2. Gamma ($\Gamma$)**")
        st.latex(r"\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{\partial \Delta}{\partial S}")
        st.write("**Meaning:** The rate of change of Delta with respect to changes in the underlying asset's price (measuring convexity).")

        st.write(r"**3. Theta ($\Theta$)**")
        st.latex(r"\Theta = \frac{\partial V}{\partial t}")
        st.write("**Meaning:** The rate of change of the option price with respect to the passage of time (commonly referred to as time decay).")

        st.write(r"**4. Vega ($\mathcal{V}$)**")
        st.latex(r"\mathcal{V} = \frac{\partial V}{\partial \sigma}")
        st.write("**Meaning:** The rate of change of the option price with respect to changes in the underlying asset's implied volatility.")

        st.write(r"**5. Rho ($\rho$)**")
        st.latex(r"\rho = \frac{\partial V}{\partial r}")
        st.write("**Meaning:** The rate of change of the option price with respect to changes in the risk-free interest rate.")

    with tab_charts:
        st.subheader("Simulation Analytics and Convergence")

        K_conv = strikes[len(strikes) // 2]
        bs_ref_call, _, _ = black_scholes_call(S0, K_conv, st.session_state.T_input, st.session_state.r_input, sigma_annual)
        bs_ref_put, _, _  = black_scholes_put(S0, K_conv, st.session_state.T_input, st.session_state.r_input, sigma_annual)

        path_counts = [500, 1000, 5000, 10000, 50000]
        conv_prices_call, conv_errors_call = [], []
        conv_prices_put, conv_errors_put   = [], []

        for n in path_counts:
            mc_c, mc_ec = monte_carlo_option(S0, K_conv, st.session_state.T_input, st.session_state.r_input, sigma_annual, dt, n, "call", n_batches=5)
            mc_p, mc_ep = monte_carlo_option(S0, K_conv, st.session_state.T_input, st.session_state.r_input, sigma_annual, dt, n, "put", n_batches=5)
            conv_prices_call.append(mc_c); conv_errors_call.append(mc_ec)
            conv_prices_put.append(mc_p); conv_errors_put.append(mc_ep)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams['font.family'] = 'serif'

        # --- PLOT 1: GBM Paths ---
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        sample_paths = simulate_gbm(S0, st.session_state.r_input, sigma_annual, st.session_state.T_input, dt, 5)
        t_axis = np.linspace(0, st.session_state.T_input, sample_paths.shape[0])
        ax1.plot(t_axis, sample_paths[:,:5], alpha=1, lw=1)
        ax1.axhline(S0, color="red", ls="--", lw=1, label=f"Spot Price = Rs{S0:.2f}")
        ax1.set_title("Simulated Geometric Brownian Motion Trajectories (n=5)", fontsize=13)
        ax1.legend()
        st.pyplot(fig1)

        st.markdown("""
        **Analytical Summary:**
        This graph displays 5 hypothetical futures for the stock price. Because the asset's trajectory is inherently stochastic, the Geometric Brownian Motion model simulates potential evolutionary paths from the present point to the option's expiration date. Each line constitutes one simulated scenario, with the dashed baseline representing the current spot price.
        """)
        st.markdown("---")

        # --- PLOT 2: Terminal Distribution ---
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        term_paths = simulate_gbm(S0, st.session_state.r_input, sigma_annual, st.session_state.T_input, dt, 50_000)
        S_T_dist   = term_paths[-1]
        ax2.hist(S_T_dist, bins=80, density=True, color="blue", edgecolor="white", alpha=0.75, label="Simulated Terminal Distribution")
        mu_ln  = np.log(S0) + (st.session_state.r_input - 0.5 * sigma_annual**2) * st.session_state.T_input
        sig_ln = sigma_annual * np.sqrt(st.session_state.T_input)
        xs     = np.linspace(S_T_dist.min(), S_T_dist.max(), 400)
        pdf_ln = (1 / (xs * sig_ln * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(xs) - mu_ln) / sig_ln)**2)
        ax2.plot(xs, pdf_ln, color="red", lw=2, label="Theoretical Log-Normal PDF")
        ax2.axvline(S0, color="green", ls="--", lw=1.3, label=f"Spot Price = Rs{S0:.0f}")
        ax2.set_title(f"Terminal Asset Price Distribution at T={st.session_state.T_input}", fontsize=13)
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("""
        **Analytical Summary:**
        Rather than observing single trajectories, this visualization aggregates the terminal asset prices at expiration across 50,000 discrete simulation paths. The histogram illustrates the empirical distribution of these endpoints. The superimposed theoretical curve confirms that the aggregate simulation accurately converges to the continuous log-normal probability density function demanded by the model's analytical framework.
        """)
        st.markdown("---")

        # --- PLOT 3: Call Convergence ---
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.semilogx(path_counts, conv_prices_call, color="blue", marker="o", lw=2, label="Monte Carlo Estimation")
        ax3.axhline(bs_ref_call, color="red", ls="--", lw=1.5, label=f"Analytical Target = Rs{bs_ref_call:.2f}")
        ax3.fill_between(path_counts, np.array(conv_prices_call) - 1.96 * np.array(conv_errors_call), np.array(conv_prices_call) + 1.96 * np.array(conv_errors_call), alpha=0.20, color="blue")
        ax3.set_title("Call Option Pricing: Convergence Analysis", fontsize=13)
        ax3.legend()
        st.pyplot(fig3)

        st.markdown("""
        **Analytical Summary:**
        This graph maps the computational convergence of the Monte Carlo technique for the Call option. The analytical Black-Scholes benchmark is represented by the horizontal axis. As the simulation dimension expands along the logarithmic x-axis, the discrete estimates smoothly converge toward the theoretical value. Furthermore, the corresponding 95% confidence interval progressively narrows, confirming the inverse-square-root reduction in standard error.
        """)
        st.markdown("---")

        # --- PLOT 4: Put Convergence ---
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        ax4.semilogx(path_counts,conv_prices_put, color="purple", marker="o", lw=2, label="Monte Carlo Estimation")
        ax4.axhline(bs_ref_put, color="red", ls="--", lw=1.5, label=f"Analytical Target = Rs{bs_ref_put:.2f}")
        ax4.fill_between(path_counts, np.array(conv_prices_put) - 1.96 * np.array(conv_errors_put), np.array(conv_prices_put) + 1.96 * np.array(conv_errors_put), alpha=0.20, color="purple")
        ax4.set_title("Put Option Pricing: Convergence Analysis", fontsize=13)
        ax4.legend()
        st.pyplot(fig4)

        st.markdown("""
        **Analytical Summary:**
        Analogous to the preceding convergence analysis, this chart tracks the stabilization of the Put option estimate. It visually validates the robustness of the Monte Carlo approach: increasing computational iterations yields a definitive statistical convergence toward the closed-form Black-Scholes valuation, minimizing variance and bounding the solution within an increasingly precise confidence interval.
        """)
