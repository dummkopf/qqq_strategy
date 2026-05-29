"""
Validate the leveraged-ETF simulation against the REAL TQQQ (2010-02-11 .. 2019-10-04).

We fit (expense_ratio, swap_spread) to minimise the difference between the
simulated and real TQQQ cumulative total return, then report:
  * daily return correlation
  * annualised tracking difference (CAGR sim - CAGR real)
  * annualised tracking error (stdev of daily diff * sqrt(252))
A good fit justifies using the SAME model to back-fill TQQQ before 2010 and
QLD before 2006.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import qqqlib as q


def metrics(sim: pd.Series, real: pd.Series):
    j = pd.concat([sim.rename("sim"), real.rename("real")], axis=1).dropna()
    corr = j["sim"].corr(j["real"])
    n = len(j)
    yrs = n / 252.0
    cagr_sim = (1 + j["sim"]).prod() ** (1 / yrs) - 1
    cagr_real = (1 + j["real"]).prod() ** (1 / yrs) - 1
    te = (j["sim"] - j["real"]).std() * np.sqrt(252)
    return dict(n=n, corr=corr, cagr_sim=cagr_sim, cagr_real=cagr_real,
                cagr_diff=cagr_sim - cagr_real, tracking_error=te)


def main():
    qqq = q.load_qqq()
    real = q.load_real_tqqq()
    lo, hi = real.index.min(), real.index.max()
    pr = qqq.loc[lo:hi, "price_ret"]
    real_ret = real.loc[lo:hi, "ret"]

    # grid-search expense_ratio + swap_spread to best match cumulative return
    best = None
    for er in np.arange(0.0070, 0.0131, 0.0005):
        for sp in np.arange(0.0, 0.0081, 0.0010):
            sim = q.simulate_letf(pr, 3.0, er, sp)
            m = metrics(sim, real_ret)
            score = abs(m["cagr_diff"])
            if best is None or score < best[0]:
                best = (score, er, sp, m)
    _, er, sp, m = best
    print("=== TQQQ simulation validation (2010-02-11 .. 2019-10-04) ===")
    print(f"best-fit expense_ratio = {er*100:.2f}%/yr, swap_spread = {sp*100:.2f}%/yr")
    print(f"daily-return correlation : {m['corr']:.5f}")
    print(f"sim  CAGR                 : {m['cagr_sim']*100:7.2f}%")
    print(f"real CAGR                 : {m['cagr_real']*100:7.2f}%")
    print(f"CAGR difference           : {m['cagr_diff']*100:7.2f}%/yr")
    print(f"annualised tracking error : {m['tracking_error']*100:7.2f}%")

    # also report a "round-number published" parameterisation for reference
    sim_pub = q.simulate_letf(pr, 3.0, 0.0095, 0.0040)
    mp = metrics(sim_pub, real_ret)
    print("\n--- reference (ER=0.95%, spread=0.40%) ---")
    print(f"corr {mp['corr']:.5f} | sim CAGR {mp['cagr_sim']*100:.2f}% | "
          f"real {mp['cagr_real']*100:.2f}% | diff {mp['cagr_diff']*100:.2f}%/yr | "
          f"TE {mp['tracking_error']*100:.2f}%")

    # cumulative levels endpoint check
    lvl_sim = q.levels_from_returns(sim_pub)
    lvl_real = q.levels_from_returns(real_ret)
    print(f"\nEndpoint multiple over period: sim {lvl_sim.iloc[-1]/100:.2f}x  "
          f"real {lvl_real.iloc[-1]/100:.2f}x")
    return er, sp


if __name__ == "__main__":
    main()
