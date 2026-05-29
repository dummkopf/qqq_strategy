"""
Validate the leveraged-ETF simulation against the REAL funds over their full
Yahoo history (TQQQ 2010-2026, QLD 2006-2026 -- both include the 2022 bear and
the 2023-25 high-rate regime, the regimes that matter most).

For each fund we grid-fit (expense_ratio, swap_spread) to match cumulative
total return, and report daily correlation, CAGR difference, and annualised
tracking error.  A tight fit justifies using the same model to back-fill the
fund before it existed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import qqqlib as q


def metrics(sim: pd.Series, real: pd.Series) -> dict:
    j = pd.concat([sim.rename("s"), real.rename("r")], axis=1).dropna()
    yrs = len(j) / 252.0
    return dict(
        n=len(j),
        corr=j["s"].corr(j["r"]),
        cagr_sim=(1 + j["s"]).prod() ** (1 / yrs) - 1,
        cagr_real=(1 + j["r"]).prod() ** (1 / yrs) - 1,
        te=(j["s"] - j["r"]).std() * np.sqrt(252),
        mult_sim=(1 + j["s"]).prod(),
        mult_real=(1 + j["r"]).prod(),
    )


def fit(qqq, ticker, leverage):
    real = q.load_real(ticker)
    base = qqq["tr_ret"].reindex(real.index)
    best = None
    for er in np.arange(0.0070, 0.0131, 0.0005):
        for sp in np.arange(-0.001, 0.0101, 0.0005):
            sim = q.simulate_letf(base, leverage, er, sp)
            m = metrics(sim, real["ret"])
            score = abs(m["cagr_diff"]) if False else abs(m["cagr_sim"] - m["cagr_real"])
            if best is None or score < best[0]:
                best = (score, er, sp, m)
    return best, real


def report(name, leverage, best, qqq, real):
    _, er, sp, m = best
    print(f"\n=== {name} ({leverage:.0f}x)  validation over {real.index.min().date()} .. {real.index.max().date()} ===")
    print(f"  best-fit:  expense_ratio={er*100:.2f}%/yr  swap_spread={sp*100:+.2f}%/yr")
    print(f"  daily-return correlation : {m['corr']:.5f}")
    print(f"  simulated CAGR           : {m['cagr_sim']*100:7.2f}%")
    print(f"  real      CAGR           : {m['cagr_real']*100:7.2f}%")
    print(f"  CAGR difference          : {(m['cagr_sim']-m['cagr_real'])*100:+7.2f}%/yr")
    print(f"  annualised tracking error: {m['te']*100:7.2f}%")
    print(f"  total growth multiple    : sim {m['mult_sim']:8.1f}x   real {m['mult_real']:8.1f}x")
    # reference with the params hard-coded in qqqlib.CALIB
    cp = q.CALIB[name]
    sim_c = q.simulate_letf(qqq["tr_ret"].reindex(real.index), leverage, cp["expense_ratio"], cp["swap_spread"])
    mc = metrics(sim_c, real["ret"])
    print(f"  [CALIB ER={cp['expense_ratio']*100:.2f}% sp={cp['swap_spread']*100:.2f}%] "
          f"corr {mc['corr']:.5f}  CAGR diff {(mc['cagr_sim']-mc['cagr_real'])*100:+.2f}%/yr  TE {mc['te']*100:.2f}%")


def main():
    qqq = q.load_qqq()
    bt, rt = fit(qqq, "TQQQ", 3.0)
    report("TQQQ", 3.0, bt, qqq, rt)
    bq, rq = fit(qqq, "QLD", 2.0)
    report("QLD", 2.0, bq, qqq, rq)


if __name__ == "__main__":
    main()
