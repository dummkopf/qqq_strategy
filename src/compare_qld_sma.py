"""
回答：QLD 还需要 SMA 吗？SMA 是否已解决"过度杠杆"？
对比 QQQ / QLD / QLD+SMA→QQQ / TQQQ / TQQQ+SMA→QQQ，$1000/月 10年DCA，历史滚动 + 蒙特卡洛。
"""
import numpy as np, pandas as pd
import qqqlib as q, strategy as st, run_study as rs
from monte_carlo import stationary_bootstrap_idx, build_path, L as NLEN, MEAN_BLOCK, N_SIMS

STRAT=[("QQQ","QQQ",None),("QLD","QLD",None),("QLD+SMA","QLD","QQQ"),
       ("TQQQ","TQQQ",None),("TQQQ+SMA","TQQQ","QQQ")]
NAMES=[s[0] for s in STRAT]

def run(returns,dates,price,cash,agg,defen):
    return st.run_dca(returns,dates,contribution=1000.0,aggressive=agg,defensive=defen,
                      band=0.02,qqq_price=price,cash_rate=cash)

def show(title,recs):
    print(f"\n{title}")
    print(f"  {'策略':>10} | {'倍数中位':>8} | {'IRR中位':>7} | {'最差5%(p5)倍数':>13} | {'亏损概率':>7} | {'中位回撤':>7}")
    base=np.array([r["QQQ"][0] for r in recs])
    for nm in NAMES:
        m=np.array([r[nm][0] for r in recs]); irr=np.array([r[nm][1] for r in recs]); dd=np.array([r[nm][2] for r in recs])
        print(f"  {nm:>10} | {np.median(m):>7.2f}x | {np.median(irr):>6.1f}% | {np.percentile(m,5):>12.2f}x | "
              f"{(m<1).mean()*100:>6.0f}% | {np.median(dd):>6.0f}%")

def main():
    qqq=q.load_qqq(); returns=rs.build_returns(qqq); idx=qqq.index
    last=idx[-1]-pd.DateOffset(years=10)
    mf=pd.Series(idx,index=idx).groupby([idx.year,idx.month]).first()
    starts=[d for d in mf.values if pd.Timestamp(d)<=last]
    hist=[]
    for s in starts:
        d=idx[(idx>=s)&(idx<=pd.Timestamp(s)+pd.DateOffset(years=10))]
        if len(d)<252*9: continue
        cash=q.short_rate_annual(d); rec={}
        for nm,a,de in STRAT:
            r=run(returns,d,qqq["price_level"],cash,a,de)
            rec[nm]=(r["final"]/r["invested"],r["irr"]*100,r["max_drawdown"]*100)
        hist.append(rec)
    show(f"A. 历史滚动10年（{len(hist)}起点, $1000/月）",hist)

    pr=qqq["price_ret"].values[1:]; tr=qqq["tr_ret"].values[1:]
    rate=q.short_rate_annual(qqq.index).values[1:]; nh=len(pr)
    md=pd.bdate_range("2000-01-03",periods=NLEN); mc=[]
    for i in range(N_SIMS):
        ix=stationary_bootstrap_idx(nh,NLEN,MEAN_BLOCK)
        rr,price,cash=build_path(pr,tr,rate,ix,md); rec={}
        for nm,a,de in STRAT:
            r=run(rr,md,price,cash,a,de)
            rec[nm]=(r["final"]/r["invested"],r["irr"]*100,r["max_drawdown"]*100)
        mc.append(rec)
        if (i+1)%500==0: print(f"  ...MC {i+1}/{N_SIMS}")
    show(f"B. 蒙特卡洛{N_SIMS}条（更客观）",mc)
    print("\nDONE_MARKER")

if __name__=="__main__": main()
