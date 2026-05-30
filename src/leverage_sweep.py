"""
QLD(2x) vs TQQQ(3x) 严格对比 + 最优杠杆分析。
A. 真实数据：QQQ/QLD/TQQQ 近5/10/15年 年化、最大回撤、总倍数。
B. 最优杠杆曲线：g_L = L·μ - L²σ²/2 - (L-1)f - er，扫 L=1..4，在"过去20年(强劲)"
   与"温和10%"两组参数下找峰值。
C. 杠杆扫描(模拟+验证)：L=1/1.5/2/2.5/3 的 DCA（历史滚动 + 蒙特卡洛），
   看中位、p5、亏损概率、回撤、收益/风险。
"""
from __future__ import annotations
import numpy as np, pandas as pd
import qqqlib as q, strategy as st, run_study as rs
from monte_carlo import stationary_bootstrap_idx, build_path, L as NLEN, MEAN_BLOCK, N_SIMS

ER, SPREAD = 0.0085, 0.0080   # 统一成本(介于QLD/TQQQ实测之间)


def realA():
    qqq=q.load_qqq(); end=qqq.index[-1]
    funds={"QQQ":qqq["tr_level"],
           "QLD":q.levels_from_returns(q.load_real("QLD")["ret"],100),
           "TQQQ":q.levels_from_returns(q.load_real("TQQQ")["ret"],100)}
    # 用真实基金
    qld=q.load_real("QLD"); tqq=q.load_real("TQQQ")
    print("A. 真实数据：年化收益 / 最大回撤 / 总倍数")
    print(f"{'期限':>5} | {'QQQ年化/回撤':>16} | {'QLD(2x)年化/回撤/倍数':>22} | {'TQQQ(3x)年化/回撤/倍数':>24}")
    for yrs in [5,10,15]:
        d0=end-pd.DateOffset(years=yrs)
        def stat(lvl):
            s=lvl[lvl.index>=d0]; cagr=(s.iloc[-1]/s.iloc[0])**(252/len(s))-1
            p=s.values; dd=(p/np.maximum.accumulate(p)-1).min(); return cagr*100,dd*100,s.iloc[-1]/s.iloc[0]
        qc,qd,_=stat(qqq["tr_level"])
        lc,ld,lm=stat(q.levels_from_returns(qld["ret"],100))
        tc,td,tm=stat(q.levels_from_returns(tqq["ret"],100))
        print(f"{yrs:>4}年 | {qc:>6.1f}% /{qd:>5.0f}%      | {lc:>6.1f}% /{ld:>5.0f}% /{lm:>6.1f}x      | {tc:>6.1f}% /{td:>5.0f}% /{tm:>7.1f}x")


def optimalL():
    print("\nB. 最优杠杆曲线（几何年化增长 g_L, 单位%）")
    for label,g1,sigma,f in [("过去20年(强劲)",0.155,0.25,0.045),
                             ("温和10%+当前利率",0.10,0.25,0.045),
                             ("温和10%+零利率",0.10,0.25,0.005)]:
        mu=g1+sigma**2/2   # 算术均值
        print(f"  情景 {label}: μ算术≈{mu*100:.0f}%, σ={sigma*100:.0f}%, 利率={f*100:.1f}%")
        row=[]
        best=(None,-9)
        for Lv in [1,1.5,2,2.5,3,3.5]:
            g=Lv*mu - (Lv*sigma)**2/2 - (Lv-1)*f - (ER if Lv>1 else 0)
            row.append(f"L={Lv}:{g*100:5.1f}")
            if g>best[1]: best=(Lv,g)
        print("    " + " | ".join(row) + f"   → 最优≈{best[0]}x")


def lev_sweep():
    qqq=q.load_qqq(); idx=qqq.index
    levs=[1.0,1.5,2.0,2.5,3.0]
    rets={f"L{Lv}": (qqq["tr_ret"] if Lv==1 else q.simulate_letf(qqq["tr_ret"],Lv,ER,SPREAD)) for Lv in levs}
    def run(returns,dates,price,cash,Lv):
        agg=f"L{Lv}"
        return st.run_dca({**returns,"AGG":returns[agg]},dates,contribution=1000.0,
                          aggressive="AGG",defensive=None,qqq_price=price,cash_rate=cash)
    # 历史滚动
    last=idx[-1]-pd.DateOffset(years=10)
    mf=pd.Series(idx,index=idx).groupby([idx.year,idx.month]).first()
    starts=[d for d in mf.values if pd.Timestamp(d)<=last]
    hist={Lv:dict(irr=[],dd=[],mult=[]) for Lv in levs}
    for s in starts:
        d=idx[(idx>=s)&(idx<=pd.Timestamp(s)+pd.DateOffset(years=10))]
        if len(d)<252*9: continue
        cash=q.short_rate_annual(d)
        for Lv in levs:
            r=run(rets,d,qqq["price_level"],cash,Lv)
            hist[Lv]["irr"].append(r["irr"]*100); hist[Lv]["dd"].append(r["max_drawdown"]*100)
            hist[Lv]["mult"].append(r["final"]/r["invested"])
    print(f"\nC. 杠杆扫描 — 历史滚动10年DCA（{len(hist[1.0]['irr'])}起点, $1000/月）")
    print(f"  {'杠杆':>5} | {'IRR中位':>7} | {'倍数中位':>8} | {'最差IRR(p5)':>11} | {'亏损概率':>7} | {'中位回撤':>7} | {'IRR/|回撤|':>9}")
    for Lv in levs:
        irr=np.array(hist[Lv]["irr"]); dd=np.array(hist[Lv]["dd"]); m=np.array(hist[Lv]["mult"])
        print(f"  {Lv:>4}x | {np.median(irr):>6.1f}% | {np.median(m):>7.2f}x | {np.percentile(irr,5):>10.1f}% | "
              f"{(m<1).mean()*100:>6.0f}% | {np.median(dd):>6.0f}% | {np.median(irr)/abs(np.median(dd)):>8.2f}")
    # 蒙特卡洛
    pr=qqq["price_ret"].values[1:]; tr=qqq["tr_ret"].values[1:]
    rate=q.short_rate_annual(qqq.index).values[1:]; nh=len(pr)
    md=pd.bdate_range("2000-01-03",periods=NLEN)
    mc={Lv:dict(irr=[],dd=[],mult=[]) for Lv in levs}
    for i in range(N_SIMS):
        ix=stationary_bootstrap_idx(nh,NLEN,MEAN_BLOCK)
        rr,price,cash=build_path(pr,tr,rate,ix,md)
        lret={f"L{Lv}":(rr["QQQ"] if Lv==1 else q.simulate_letf(rr["QQQ"],Lv,ER,SPREAD,rate_annual=cash)) for Lv in levs}
        for Lv in levs:
            r=st.run_dca({"AGG":lret[f"L{Lv}"]},md,contribution=1000.0,aggressive="AGG",
                         defensive=None,qqq_price=price,cash_rate=cash)
            mc[Lv]["irr"].append(r["irr"]*100); mc[Lv]["dd"].append(r["max_drawdown"]*100)
            mc[Lv]["mult"].append(r["final"]/r["invested"])
        if (i+1)%500==0: print(f"    ...MC {i+1}/{N_SIMS}")
    print(f"\nC2. 杠杆扫描 — 蒙特卡洛{N_SIMS}条（更客观）")
    print(f"  {'杠杆':>5} | {'IRR中位':>7} | {'倍数中位':>8} | {'最差IRR(p5)':>11} | {'亏损概率':>7} | {'中位回撤':>7} | {'IRR/|回撤|':>9}")
    for Lv in levs:
        irr=np.array(mc[Lv]["irr"]); dd=np.array(mc[Lv]["dd"]); m=np.array(mc[Lv]["mult"])
        print(f"  {Lv:>4}x | {np.median(irr):>6.1f}% | {np.median(m):>7.2f}x | {np.percentile(irr,5):>10.1f}% | "
              f"{(m<1).mean()*100:>6.0f}% | {np.median(dd):>6.0f}% | {np.median(irr)/abs(np.median(dd)):>8.2f}")
    print("\nDONE_MARKER")


def main():
    realA(); optimalL(); lev_sweep()


if __name__=="__main__":
    main()
