"""
Same QQQ all-time-high recovery episodes as ath_recovery.py, but the investor
DCA's $1,000/month THROUGH the episode instead of holding a lump sum.

Why it can differ: DCA-ing down into a crash buys cheap shares that catch the
amplified recovery bounce -- this works *for* leverage, partly offsetting the
volatility decay that hurts a lump-sum holder.  So we ask, per episode
(QQQ peak -> back to that peak):
  * Is DCA-TQQQ profitable at the recovery (multiple > 1)?
  * Does DCA-TQQQ beat DCA-QQQ over the same span?
and contrast with the lump-sum holder result.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import qqqlib as q
import run_study as rs
from ath_recovery import find_episodes

CONTRIB = 1000.0


def main():
    qqq = q.load_qqq()
    returns = rs.build_returns(qqq)
    qlvl = qqq["tr_level"]
    tlvl = q.levels_from_returns(returns["TQQQ"], 100.0)

    ep = find_episodes(qlvl, tlvl)
    ep = ep[(ep["depth_pct"] <= -1.0) & ep["recovered"]].reset_index(drop=True)

    rows = []
    for _, e in ep.iterrows():
        t_p = pd.Timestamp(e["peak_date"]); t_r = pd.Timestamp(e["recovery_date"])
        dates = qqq.index[(qqq.index >= t_p) & (qqq.index <= t_r)]
        if len(dates) < 2:
            continue
        rT = rs.run_one(returns, qqq, dates, aggressive="TQQQ", defensive=None)
        rQ = rs.run_one(returns, qqq, dates, aggressive="QQQ",  defensive=None)
        # lump-sum holder TQQQ over the same span, for contrast
        lump_T = tlvl.loc[dates[-1]] / tlvl.loc[dates[0]] - 1.0
        rows.append(dict(
            peak_date=e["peak_date"], recovery_date=e["recovery_date"], era=e["era"],
            dd_years=e["dd_years"], depth_pct=e["depth_pct"],
            invested=rT["invested"],
            dca_qqq_mult=rQ["final"] / rQ["invested"],
            dca_tqqq_mult=rT["final"] / rT["invested"],
            dca_tqqq_irr=rT["irr"] * 100,
            dca_tqqq_better=rT["final"] > rQ["final"],
            dca_tqqq_underwater=rT["final"] < rT["invested"],
            lump_tqqq_rt_pct=lump_T * 100,
        ))
    d = pd.DataFrame(rows)
    d.to_csv("../results/ath_recovery_dca.csv", index=False)

    print("=" * 100)
    print('TEST: DCA $1,000/mo THROUGH each QQQ peak->recovery episode  (TQQQ vs QQQ)')
    print("=" * 100)
    n = len(d)
    print(f"\nEpisodes: {n}")
    print(f">>> DCA-TQQQ ended ABOVE total invested in {(~d.dca_tqqq_underwater).sum()}/{n} "
          f"({(~d.dca_tqqq_underwater).mean()*100:.0f}%)")
    print(f">>> DCA-TQQQ beat DCA-QQQ in {d.dca_tqqq_better.sum()}/{n} "
          f"({d.dca_tqqq_better.mean()*100:.0f}%)")
    print("   (compare lump-sum holder: TQQQ was underwater in ~26% of episodes overall)\n")

    print("By how deep QQQ fell  (DCA buys cheap shares into the dip):")
    print(f"  {'QQQ drawdown':>16} | {'n':>3} | {'med DCA-QQQ':>11} | {'med DCA-TQQQ':>12} | "
          f"{'DCA-TQQQ beats QQQ':>18} | {'med lump-sum TQQQ':>17}")
    for lo, hi in [(-5, 0), (-10, -5), (-20, -10), (-35, -20), (-100, -35)]:
        b = d[(d.depth_pct > lo) & (d.depth_pct <= hi)]
        if len(b) == 0:
            continue
        print(f"  {hi:>6.0f}% to {lo:>5.0f}% | {len(b):>3} | {b.dca_qqq_mult.median():>10.2f}x | "
              f"{b.dca_tqqq_mult.median():>11.2f}x | {b.dca_tqqq_better.mean()*100:>16.0f}% | "
              f"{b.lump_tqqq_rt_pct.median():>16.1f}%")

    print("\nDeepest episodes (DCA through the whole drawdown-and-recovery):")
    cols = ["peak_date", "recovery_date", "era", "dd_years", "depth_pct",
            "invested", "dca_qqq_mult", "dca_tqqq_mult", "dca_tqqq_irr", "lump_tqqq_rt_pct"]
    print(d.sort_values("depth_pct").head(10)[cols]
          .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\n-> results/ath_recovery_dca.csv")


if __name__ == "__main__":
    main()
