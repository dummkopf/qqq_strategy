"""
蒙特卡洛稳健性：用多个不同随机种子各跑一次 (N=2000)，比较跑次间的波动，
判断 QLD vs TQQQ 等差异是否具统计显著性（差异 >> 跨种子噪声）。
"""
import numpy as np, pandas as pd
import qqqlib as q, strategy as st, run_study as rs
import monte_carlo as mc

SEEDS=[1,2,3,4,5]
N=mc.N_SIMS; NLEN=mc.L; MB=mc.MEAN_BLOCK
STRAT=[("QQQ","QQQ",None),("QLD","QLD",None),("QLD+SMA","QLD","QQQ"),
       ("TQQQ","TQQQ",None),("TQQQ+SMA","TQQQ","QQQ")]
NAMES=[s[0] for s in STRAT]

def one_seed(seed,qqq,pr,tr,rate,nh,md):
    mc.RNG=np.random.default_rng(seed)                 # 重设种子
    acc={nm:{"mult":[],"irr":[],"dd":[]} for nm in NAMES}
    for _ in range(N):
        idx=mc.stationary_bootstrap_idx(nh,NLEN,MB)
        rr,price,cash=mc.build_path(pr,tr,rate,idx,md)
        for nm,a,de in STRAT:
            r=st.run_dca(rr,md,contribution=1000.0,aggressive=a,defensive=de,
                         band=0.02,qqq_price=price,cash_rate=cash)
            acc[nm]["mult"].append(r["final"]/r["invested"])
            acc[nm]["irr"].append(r["irr"]*100); acc[nm]["dd"].append(r["max_drawdown"]*100)
    out={}
    for nm in NAMES:
        m=np.array(acc[nm]["mult"]); irr=np.array(acc[nm]["irr"]); dd=np.array(acc[nm]["dd"])
        out[nm]=dict(med_mult=np.median(m),med_irr=np.median(irr),
                     p5=np.percentile(m,5),loss=(m<1).mean()*100,med_dd=np.median(dd))
    return out

def main():
    qqq=q.load_qqq()
    pr=qqq["price_ret"].values[1:]; tr=qqq["tr_ret"].values[1:]
    rate=q.short_rate_annual(qqq.index).values[1:]; nh=len(pr)
    md=pd.bdate_range("2000-01-03",periods=NLEN)
    runs=[]
    for s in SEEDS:
        print(f"  running seed {s} ...")
        runs.append(one_seed(s,qqq,pr,tr,rate,nh,md))
    # 每种子每策略 中位倍数
    print("\n各随机种子下的【中位倍数】(N=2000/次):")
    print("  "+"策略".ljust(9)+" | "+" | ".join(f"seed{ s}" for s in SEEDS)+" |  均值±标准差  | 极差")
    for nm in NAMES:
        vals=np.array([r[nm]["med_mult"] for r in runs])
        cells=" | ".join(f"{v:5.2f}" for v in vals)
        print(f"  {nm:9} | {cells} | {vals.mean():4.2f}±{vals.std():.3f} | {vals.max()-vals.min():.2f}")
    print("\n各随机种子下的【亏损概率 %】:")
    for nm in NAMES:
        vals=np.array([r[nm]["loss"] for r in runs])
        cells=" | ".join(f"{v:5.1f}" for v in vals)
        print(f"  {nm:9} | {cells} | {vals.mean():4.1f}±{vals.std():.2f}")
    print("\n各随机种子下的【中位IRR %】:")
    for nm in NAMES:
        vals=np.array([r[nm]["med_irr"] for r in runs])
        cells=" | ".join(f"{v:5.1f}" for v in vals)
        print(f"  {nm:9} | {cells} | {vals.mean():4.1f}±{vals.std():.2f}")
    # 显著性：QLD vs TQQQ 中位倍数差 vs 噪声
    qld=np.array([r["QLD"]["med_mult"] for r in runs]); tqqq=np.array([r["TQQQ"]["med_mult"] for r in runs])
    diff=qld-tqqq
    print(f"\n显著性检验 QLD−TQQQ 中位倍数差: 均值 {diff.mean():.3f}, 跨种子标准差 {diff.std():.3f} "
          f"→ {'差异 >> 噪声, 显著' if abs(diff.mean())>3*diff.std() else '需更多样本'}")
    print("DONE_MARKER")

if __name__=="__main__": main()
