"""DCA starting N years ago (3/5/7/10/15), all ending at the latest date.
$1,000/month. Real data throughout (TQQQ exists since 2010)."""
from __future__ import annotations
import pandas as pd, qqqlib as q, run_study as rs

def main():
    qqq = q.load_qqq(); returns = rs.build_returns(qqq)
    end = qqq.index[-1]
    print(f"All windows END {end.date()} (DCA $1,000/mo). format: multiple / IRR%/yr / maxDD% / switches\n")
    rows=[]
    for yrs in [3,5,7,10,15]:
        start = end - pd.DateOffset(years=yrs)
        dates = qqq.index[(qqq.index>=start)&(qqq.index<=end)]
        for label,kw in rs.STRATS:
            res = rs.run_one(returns, qqq, dates, **kw)
            rows.append(dict(years=yrs, start=dates[0].date(), strategy=label,
                invested=res["invested"], final=res["final"],
                multiple=res["final"]/res["invested"], irr_pct=res["irr"]*100,
                max_dd_pct=res["max_drawdown"]*100, switches=res["n_switches"]))
    df=pd.DataFrame(rows)
    df.to_csv("../results/lookback_windows.csv", index=False)
    for yrs in [3,5,7,10,15]:
        sub=df[df.years==yrs]
        print(f"=== {yrs} years ago  (from {sub.iloc[0]['start']}, invested ${sub.iloc[0]['invested']:,.0f}) ===")
        for _,r in sub.iterrows():
            print(f"  {r['strategy']:28s}  ${r['final']:>12,.0f}  {r['multiple']:5.2f}x  "
                  f"{r['irr_pct']:6.1f}%/yr  DD {r['max_dd_pct']:6.1f}%  sw {int(r['switches']):>2d}")
        print()

if __name__=="__main__": main()
