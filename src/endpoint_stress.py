"""End-point stress test: same horizons (3/5/7/10/15yr) but ENDING at bad
moments instead of today's high.  If the conclusion only holds when measured at
a peak, it's fragile.

End dates:
  2002-10-09  dot-com trough
  2009-03-09  GFC trough
  2022-12-28  2022 bear trough
Horizons that start before QQQ's 1999-03-10 inception are skipped.
$1,000/month DCA.  (Pre-2010 TQQQ / pre-2006 QLD are the validated simulation.)
"""
from __future__ import annotations
import pandas as pd, qqqlib as q, run_study as rs

ENDS = {"dot-com trough 2002-10": "2002-10-09",
        "GFC trough 2009-03":     "2009-03-09",
        "2022 bear trough 2022-12": "2022-12-28"}
HORIZONS = [3, 5, 7, 10, 15]


def main():
    qqq = q.load_qqq(); returns = rs.build_returns(qqq)
    incep = qqq.index[0]
    rows = []
    for ename, edate in ENDS.items():
        end = pd.Timestamp(edate)
        end = qqq.index[qqq.index <= end][-1]      # snap to trading day
        print(f"\n################  ENDING AT {ename}  (data ends {end.date()})  ################")
        for yrs in HORIZONS:
            start = end - pd.DateOffset(years=yrs)
            if start < incep:
                print(f"  [{yrs:>2}yr]  start {start.date()} < QQQ inception {incep.date()} -> skipped")
                continue
            dates = qqq.index[(qqq.index >= start) & (qqq.index <= end)]
            invested = None
            print(f"\n  === {yrs}yr DCA: {dates[0].date()} -> {dates[-1].date()} ===")
            for label, kw in rs.STRATS:
                res = rs.run_one(returns, qqq, dates, **kw)
                invested = res["invested"]
                rows.append(dict(endpoint=ename, end=end.date(), years=yrs,
                                 start=dates[0].date(), strategy=label,
                                 invested=res["invested"], final=res["final"],
                                 multiple=res["final"]/res["invested"], irr_pct=res["irr"]*100,
                                 max_dd_pct=res["max_drawdown"]*100, switches=res["n_switches"]))
                print(f"     {label:28s}  ${res['final']:>11,.0f}  {res['final']/res['invested']:5.2f}x  "
                      f"{res['irr']*100:7.1f}%/yr  DD {res['max_drawdown']*100:6.1f}%  sw {res['n_switches']:>2d}")
            print(f"     (invested ${invested:,.0f})")
    pd.DataFrame(rows).to_csv("../results/endpoint_stress.csv", index=False)
    print("\n-> results/endpoint_stress.csv")


if __name__ == "__main__":
    main()
