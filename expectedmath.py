"""
Monte Carlo simulation comparing different early-career paths:

- RIQ founder (you) – no salary in year 0, founder salary + potential exits
- J&J-style frontier AI role – high salary + small equity
- GS relationship lending analyst – 50k base (July 2025 start)
- IBD analyst – 65k base (July 2025 start)
- Oxford Economics grad – 44k base (July 2025 start)
- Millennium ops/middle office – 60k base (July 2025 start)

Horizon: 5 years from July 2025 to July 2030
Output: wealth curves (percentiles) for each path
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------
# Global simulation settings
# ---------------------------

YEARS = 5              # 0 = Jul 2025–Jul 2026, ..., 5 = Jul 2029–Jul 2030
N_SCENARIOS = 20000    # Monte Carlo runs
INV_RETURN = 0.05      # annual investment return on saved money (after inflation)
START_WEALTH = 0.0     # assume everyone starts with £0 net worth

years = np.arange(YEARS + 1)

# ---------------------------
# Helper functions
# ---------------------------

def simulate_trad_path(
    years,
    n_scenarios,
    base_salary_year0,
    salary_growth_mean,
    salary_growth_std,
    bonus_rate_mean,
    bonus_rate_std,
    save_rate_mean,
    save_rate_std,
    start_delay_years=0,
    inv_return=INV_RETURN,
):
    """
    Generic traditional path:
    - salary + bonus
    - random growth & bonus
    - fixed saving rate (fraction of total compensation)
    - delayed start (for your path if needed)
    """
    wealth = np.zeros((n_scenarios, years + 1))

    for i in range(n_scenarios):
        w = START_WEALTH
        salary = base_salary_year0

        for t in range(1, years + 1):
            # Invest existing wealth
            w *= (1 + inv_return)

            # Optionally delay start (e.g. still in uni, no salary)
            if t <= start_delay_years:
                wealth[i, t] = w
                continue

            # Salary growth
            growth = np.random.normal(salary_growth_mean, salary_growth_std)
            growth = max(-0.2, growth)  # clamp extreme negative years
            salary *= (1 + growth)

            # Bonus as % of salary
            bonus_rate = np.random.normal(bonus_rate_mean, bonus_rate_std)
            bonus_rate = max(0.0, bonus_rate)
            bonus = salary * bonus_rate

            # Saving rate on total comp
            save_rate = np.random.normal(save_rate_mean, save_rate_std)
            save_rate = np.clip(save_rate, 0.05, 0.6)  # keep it realistic
            savings = (salary + bonus) * save_rate

            w += savings
            wealth[i, t] = w

    return wealth


def simulate_riq_founder(
    years,
    n_scenarios,
    inv_return=INV_RETURN,
):
    """
    RIQ founder path (you):

    - Year 0: still in LSE, effectively £0 salary (building RIQ)
    - Years 1–5: founder salary + optional exit scenarios

    Exit scenarios (illustrative, tweak as you like):
    - 50%: no liquidity event yet by year 5 (still building, only salary)
    - 30%: mid exit ~£5m–£15m to you (after dilution, tax ignored)
    - 20%: big exit ~£20m–£80m to you

    Exit year (if any): uniform between years 3–5 (you need time to build)
    Founder salary: Normal(£80k, 15k) per year from t>=1
    Saving rate: Normal(0.4, 0.05)
    """
    wealth = np.zeros((n_scenarios, years + 1))

    exit_types = ["none", "mid", "big"]
    exit_probs = [0.5, 0.3, 0.2]

    for i in range(n_scenarios):
        w = START_WEALTH

        # Sample exit scenario
        etype = np.random.choice(exit_types, p=exit_probs)
        exit_year = None
        if etype != "none":
            exit_year = np.random.randint(3, years + 1)  # years 3–5

        # Key parameters
        founder_salary_mean = 80_000
        founder_salary_std = 15_000
        save_rate_mean = 0.4
        save_rate_std = 0.05

        # Exit size ranges (post-dilution to you)
        if etype == "mid":
            exit_mean = 10_000_000
            exit_std = 3_000_000
        elif etype == "big":
            exit_mean = 40_000_000
            exit_std = 15_000_000
        else:
            exit_mean = 0
            exit_std = 0

        for t in range(1, years + 1):
            # Invest existing wealth
            w *= (1 + inv_return)

            # Year 0 (t=0) was LSE final year, no salary; from t>=1, you can pay yourself
            # (in reality you might keep it low initially, but this is EV modelling)
            salary = np.random.normal(founder_salary_mean, founder_salary_std)
            salary = max(40_000, salary)

            save_rate = np.random.normal(save_rate_mean, save_rate_std)
            save_rate = np.clip(save_rate, 0.1, 0.7)
            savings = salary * save_rate
            w += savings

            # If exit happens this year, add liquidity
            if exit_year is not None and t == exit_year:
                payout = np.random.normal(exit_mean, exit_std)
                payout = max(0, payout)
                w += payout

            wealth[i, t] = w

    return wealth


def simulate_jj_frontier_role(
    years,
    n_scenarios,
    inv_return=INV_RETURN,
):
    """
    Jack & Jill frontier AI role:

    - Start immediately at t=0 (they'd have hired you July 2025)
    - Total comp ~£240k+ per year, with modest growth
    - Equity: small % (e.g. 0.15%), high potential but low probability

    Exit scenarios (illustrative):
    - 70%: no meaningful liquidity by year 5 (equity = 0 for now)
    - 25%: good exit: company ~£1B, your equity ~£1–3m
    - 5%: very strong exit: company ~£3B, your equity ~£3–7m

    Exit year (if any): uniform between years 3–5
    """

    wealth = np.zeros((n_scenarios, years + 1))

    exit_types = ["none", "good", "great"]
    exit_probs = [0.7, 0.25, 0.05]

    for i in range(n_scenarios):
        w = START_WEALTH

        etype = np.random.choice(exit_types, p=exit_probs)
        exit_year = None
        if etype != "none":
            exit_year = np.random.randint(3, years + 1)

        # Salary & bonus assumptions
        base_salary = 200_000   # part of the £240k TC
        bonus_rate_mean = 0.2
        bonus_rate_std = 0.1
        salary_growth_mean = 0.08
        salary_growth_std = 0.05
        save_rate_mean = 0.45
        save_rate_std = 0.1

        # Equity payout assumptions (you as mid-level founding staff)
        if etype == "good":
            exit_mean = 2_000_000
            exit_std = 800_000
        elif etype == "great":
            exit_mean = 5_000_000
            exit_std = 2_000_000
        else:
            exit_mean = 0
            exit_std = 0

        salary = base_salary

        for t in range(1, years + 1):
            # Invest existing wealth
            w *= (1 + inv_return)

            # Salary growth
            growth = np.random.normal(salary_growth_mean, salary_growth_std)
            growth = max(-0.1, growth)
            salary *= (1 + growth)

            # Bonus
            bonus_rate = np.random.normal(bonus_rate_mean, bonus_rate_std)
            bonus_rate = max(0.0, bonus_rate)
            bonus = salary * bonus_rate

            # Savings
            save_rate = np.random.normal(save_rate_mean, save_rate_std)
            save_rate = np.clip(save_rate, 0.1, 0.7)
            savings = (salary + bonus) * save_rate
            w += savings

            # Equity event
            if exit_year is not None and t == exit_year:
                payout = np.random.normal(exit_mean, exit_std)
                payout = max(0, payout)
                w += payout

            wealth[i, t] = w

    return wealth


# ---------------------------
# Run simulations
# ---------------------------

paths = {}

# RIQ founder (you) – start_delay_years handled inside (year 0 = no salary)
paths["RIQ Founder"] = simulate_riq_founder(YEARS, N_SCENARIOS)

# J&J frontier role
paths["J&J Frontier"] = simulate_jj_frontier_role(YEARS, N_SCENARIOS)

# GS Relationship Lending (started Jul 2025, so no delay)
paths["GS RL (50k)"] = simulate_trad_path(
    years=YEARS,
    n_scenarios=N_SCENARIOS,
    base_salary_year0=50_000,
    salary_growth_mean=0.06,
    salary_growth_std=0.03,
    bonus_rate_mean=0.25,
    bonus_rate_std=0.10,
    save_rate_mean=0.25,
    save_rate_std=0.05,
    start_delay_years=0,
)

# IBD Analyst (sub-BB, 65k)
paths["IBD Analyst (65k)"] = simulate_trad_path(
    years=YEARS,
    n_scenarios=N_SCENARIOS,
    base_salary_year0=65_000,
    salary_growth_mean=0.08,
    salary_growth_std=0.04,
    bonus_rate_mean=0.70,
    bonus_rate_std=0.25,
    save_rate_mean=0.35,
    save_rate_std=0.08,
    start_delay_years=0,
)

# Oxford Economics Grad (44k)
paths["OE Grad (44k)"] = simulate_trad_path(
    years=YEARS,
    n_scenarios=N_SCENARIOS,
    base_salary_year0=44_000,
    salary_growth_mean=0.05,
    salary_growth_std=0.02,
    bonus_rate_mean=0.10,
    bonus_rate_std=0.05,
    save_rate_mean=0.20,
    save_rate_std=0.05,
    start_delay_years=0,
)

# Millennium Ops/MO Grad (60k)
paths["Millennium Ops (60k)"] = simulate_trad_path(
    years=YEARS,
    n_scenarios=N_SCENARIOS,
    base_salary_year0=60_000,
    salary_growth_mean=0.07,
    salary_growth_std=0.03,
    bonus_rate_mean=0.30,
    bonus_rate_std=0.15,
    save_rate_mean=0.30,
    save_rate_std=0.07,
    start_delay_years=0,
)

# ---------------------------
# Visualisation
# ---------------------------

plt.figure(figsize=(11, 7))

for name, wealth in paths.items():
    median = np.percentile(wealth, 50, axis=0)
    p25 = np.percentile(wealth, 25, axis=0)
    p75 = np.percentile(wealth, 75, axis=0)

    plt.plot(years, median / 1e6, label=f"{name} (median)")
    plt.fill_between(years, p25 / 1e6, p75 / 1e6, alpha=0.08)

plt.axhline(0, color="black", linewidth=0.5)
plt.xticks(years, [f"Y{t}" for t in years])
plt.ylabel("Net worth (£ millions)")
plt.xlabel("Years from Jul 2025")
plt.title("Monte Carlo Net Worth Trajectories (Median & IQR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
