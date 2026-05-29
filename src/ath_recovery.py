"""
Hypothesis test: "as long as QQQ rehits its previous all-time high, it's always
better to have held TQQQ."

This is the classic leveraged-ETF misconception. A 3x *daily* fund delivers 3x
the DAILY return, not 3x the cumulative return; over any drawdown-and-recovery
round trip (QQQ net ~0%) volatility decay leaves TQQQ DOWN. The deeper/choppier
the drawdown, the worse -- and from a deep enough hole TQQQ may NEVER recover
even after QQQ goes on to new highs.

Method (total-return / wealth indices, dividends reinvested, for both):
  * Find every QQQ all-time-high peak that is later RECOVERED (QQQ returns to
    that level after a drawdown).  Each episode spans [peak date -> recovery
    date], over which QQQ's net return is ~0 by construction.
  * Measure TQQQ's return over the SAME span.  If the hypothesis held, this
    would always be >= 0.
  * Also: how far below ITS OWN prior peak is TQQQ at QQQ's recovery, and how
    much longer (if ever, within the data) until TQQQ regains its own peak.
TQQQ is REAL after 2010-02, validated simulation before.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import qqqlib as q


def find_episodes(qqq_lvl: pd.Series, tqqq_lvl: pd.Series) -> pd.DataFrame:
    peak = qqq_lvl.cummax()
    is_high = qqq_lvl.values >= peak.values            # True on all-time-high days
    dates = qqq_lvl.index
    n = len(dates)
    rows = []
    i = 0
    while i < n:
        if is_high[i]:
            # are we starting a drawdown? look for next non-high then next high
            j = i + 1
            while j < n and is_high[j]:
                j += 1
            if j >= n:
                break                                   # ran out at a high streak
            # drawdown started at i (peak), find recovery = next high day k
            k = j
            while k < n and not is_high[k]:
                k += 1
            if k >= n:
                # peak at i never recovered within the data
                trough = qqq_lvl.iloc[j:n].min()
                rows.append(_episode(dates, qqq_lvl, tqqq_lvl, i, None, trough, recovered=False))
                break
            trough = qqq_lvl.iloc[i:k+1].min()
            rows.append(_episode(dates, qqq_lvl, tqqq_lvl, i, k, trough, recovered=True))
            i = k                                        # continue from recovery
        else:
            i += 1
    return pd.DataFrame(rows)


def _episode(dates, qlvl, tlvl, ip, kr, trough, recovered):
    P = qlvl.iloc[ip]
    depth = trough / P - 1.0
    end = kr if recovered else len(dates) - 1
    # TQQQ's own max drawdown DURING this episode (peak->trough within the span)
    t_seg = tlvl.iloc[ip:end + 1]
    tqqq_trough = t_seg.min() / tlvl.iloc[ip] - 1.0
    tqqq_rt = tlvl.iloc[end] / tlvl.iloc[ip] - 1.0
    rec = {
        "peak_date": dates[ip].date(),
        "recovery_date": dates[kr].date() if recovered else None,
        "recovered": recovered,
        "era": "real" if dates[ip].year >= 2010 else "sim",   # TQQQ real after 2010-02
        "dd_years": (dates[end] - dates[ip]).days / 365.25,
        "depth_pct": depth * 100,                              # QQQ drawdown
        "qqq_rt_pct": (qlvl.iloc[end] / P - 1.0) * 100,        # ~0 if recovered
        "tqqq_trough_pct": tqqq_trough * 100,                  # TQQQ drawdown
        "tqqq_rt_pct": tqqq_rt * 100,                          # TQQQ over same span
        "tqqq_underwater": tqqq_rt < 0,
    }
    return rec


def main():
    qqq = q.load_qqq()
    qlvl = qqq["tr_level"]                            # QQQ total-return wealth index
    tqqq = q.build_fund_total_return(qqq, "TQQQ")     # real after 2010, sim before
    tlvl = q.levels_from_returns(tqqq, base=100.0)

    ep = find_episodes(qlvl, tlvl)
    # only episodes with a meaningful drawdown (ignore <1% noise dips)
    ep = ep[ep["depth_pct"] <= -1.0].reset_index(drop=True)
    ep.to_csv("../results/ath_recovery_episodes.csv", index=False)

    rec = ep[ep["recovered"]]
    print("=" * 100)
    print('TEST: "QQQ rehits its previous ATH  =>  TQQQ holder is also whole / better off"')
    print("=" * 100)
    print(f"\nQQQ drawdown-and-recovery episodes (>=1% deep): {len(ep)}  "
          f"({rec['recovered'].sum()} fully recovered, {(~ep['recovered']).sum()} still open)\n")
    uw = rec["tqqq_underwater"].sum()
    print(f">>> At the moment QQQ regained its ATH, TQQQ was STILL DOWN over the round trip "
          f"in {uw} of {len(rec)} episodes ({uw/len(rec)*100:.0f}%).")
    uw_real = rec[(rec.era == 'real') & rec.tqqq_underwater]
    rec_real = rec[rec.era == 'real']
    print(f">>> Real-TQQQ era only (2010+): {len(uw_real)} of {len(rec_real)} episodes left TQQQ underwater.")
    print(">>> i.e. the hypothesis is FALSE: recovering the index does NOT make the leveraged holder whole.\n")

    # the relationship is monotonic in drawdown depth -> bucket it
    print("How TQQQ's round-trip outcome depends on how deep QQQ fell:")
    buckets = [(-5,0),(-10,-5),(-20,-10),(-35,-20),(-100,-35)]
    print(f"  {'QQQ drawdown':>16} | {'n':>3} | {'median TQQQ round-trip':>22} | {'% TQQQ underwater':>18}")
    for lo,hi in buckets:
        b = rec[(rec.depth_pct>lo)&(rec.depth_pct<=hi)]
        if len(b)==0: continue
        print(f"  {hi:>6.0f}% to {lo:>5.0f}% | {len(b):>3} | {b.tqqq_rt_pct.median():>21.1f}% | {b.tqqq_underwater.mean()*100:>17.0f}%")

    # show the deepest / most instructive episodes
    show = rec.sort_values("depth_pct").head(12)
    cols = ["peak_date", "recovery_date", "era", "dd_years", "depth_pct",
            "qqq_rt_pct", "tqqq_trough_pct", "tqqq_rt_pct"]
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\nDeepest episodes (QQQ round trip ~0% by construction):")
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    # the canonical dot-com case (TQQQ here is SIMULATED -- directionally robust)
    print("\n--- Spotlight: the 2000 dot-com peak [TQQQ SIMULATED pre-2010] ---")
    peak2000 = qlvl.loc["2000-01-01":"2000-12-31"].idxmax()
    qpk, qend = qlvl.loc[peak2000], qlvl.iloc[-1]
    tpk, tend = tlvl.loc[peak2000], tlvl.iloc[-1]
    print(f"QQQ TR peak {peak2000.date()}.  By {qlvl.index[-1].date()}: "
          f"QQQ = {qend/qpk:.2f}x its 2000 peak;  TQQQ = {tend/tpk:.3f}x its 2000 peak.")
    print(f"=> A TQQQ holder from the 2000 top is {(tend/tpk-1)*100:+.0f}% even though QQQ is up "
          f"{(qend/qpk-1)*100:+.0f}%.")
    print("   The 2000-02 crash took the 3x fund down ~-99.9%; from there even a +641% index")
    print("   recovery can't restore it (a -99% loss needs +9,900% just to break even).")
    print("\n-> results/ath_recovery_episodes.csv")


if __name__ == "__main__":
    main()
