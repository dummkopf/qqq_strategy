"""
从 1999-03 开始 DCA（$1000/月）投 QQQ/QLD/TQQQ，看 3/5/7/10/12/15/17/20 年后的结果。
QLD(2006前)/TQQQ(2010前) 为验证过的模拟，之后为真实基金。
产出：equity曲线 + 各horizon的倍数/IRR柱状图 + 表格。
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qqqlib as q, strategy as st, run_study as rs

qqq=q.load_qqq(); returns=rs.build_returns(qqq)
start=qqq.index[0]                       # 1999-03-10
HOR=[3,5,7,10,12,15,17,20]
# 策略: (名称, 进攻资产, 防守资产None=不风控)
STRAT=[("QQQ","QQQ",None),
       ("QLD","QLD",None),
       ("TQQQ","TQQQ",None),
       ("TQQQ+SMA->QQQ","TQQQ","QQQ"),
       ("QLD+SMA->QQQ","QLD","QQQ")]
COL={"QQQ":"tab:blue","QLD":"tab:orange","TQQQ":"tab:red",
     "TQQQ+SMA->QQQ":"tab:green","QLD+SMA->QQQ":"tab:purple"}
NAMES=[s[0] for s in STRAT]

def run(dates,agg,defen):
    return st.run_dca(returns,dates,contribution=1000.0,aggressive=agg,defensive=defen,
                      band=0.02,qqq_price=qqq["price_level"])

# 全程 equity 曲线 (到 start+20yr)
end20=start+pd.DateOffset(years=20)
dfull=qqq.index[(qqq.index>=start)&(qqq.index<=end20)]
curves={}; contrib=None
for nm,agg,defen in STRAT:
    r=run(dfull,agg,defen); curves[nm]=r["value"]; contrib=r["contrib_cum"]

# 各 horizon 统计
rows=[]
for H in HOR:
    e=start+pd.DateOffset(years=H)
    d=qqq.index[(qqq.index>=start)&(qqq.index<=e)]
    rec={"H":H,"invested":None}
    for nm,agg,defen in STRAT:
        r=run(d,agg,defen); rec["invested"]=r["invested"]
        rec[nm+"_mult"]=r["final"]/r["invested"]; rec[nm+"_irr"]=r["irr"]*100
        rec[nm+"_dd"]=r["max_drawdown"]*100
    rows.append(rec)
T=pd.DataFrame(rows)
T.to_csv("../results/dca_from1999.csv",index=False)
print("从1999-03定投 $1000/月，各期末 倍数(终值/投入) / 最大回撤%")
hdr="年数 |   投入$ | " + " | ".join(f"{nm:>12}" for nm in NAMES)
print(hdr)
for _,r in T.iterrows():
    cells=" | ".join(f"{r[nm+'_mult']:>5.2f}x/{r[nm+'_dd']:>4.0f}%" for nm in NAMES)
    print(f"{int(r.H):>4} | {r.invested:>8,.0f} | {cells}")

# ---- 图1: equity 曲线 (log) ----
plt.figure(figsize=(12,6.5))
for nm in NAMES:
    plt.plot(curves[nm].index,curves[nm].values,color=COL[nm],lw=1.6,label=f"DCA {nm}")
plt.plot(contrib.index,contrib.values,"k--",lw=1,label="Total contributed")
for H in HOR:
    e=start+pd.DateOffset(years=H)
    plt.axvline(e,color="gray",ls=":",lw=0.6)
    plt.text(e,plt.ylim()[1]*0.5,f"{H}y",rotation=90,va="center",fontsize=7,color="gray")
plt.yscale("log"); plt.ylabel("Portfolio value $ (log)"); plt.xlabel("Year")
plt.title("DCA $1,000/mo from 1999-03 (worst start: dot-com top)\n"
          "+SMA = sell to QQQ when QQQ<200dMA (2% band).  QLD/TQQQ pre-inception=validated sim")
plt.legend(fontsize=8); plt.grid(alpha=.3,which="both")
plt.tight_layout(); plt.savefig("../results/dca1999_equity.png",dpi=120); plt.close()

# ---- 图2: 各horizon 倍数柱状 ----
x=np.arange(len(HOR)); w=0.16
plt.figure(figsize=(13,6.5))
for i,nm in enumerate(NAMES):
    vals=[T[T.H==H][nm+"_mult"].iloc[0] for H in HOR]
    bars=plt.bar(x+(i-2)*w,vals,w,color=COL[nm],label=nm)
    for b,v in zip(bars,vals):
        plt.text(b.get_x()+b.get_width()/2,v,f"{v:.1f}",ha="center",va="bottom",fontsize=6)
plt.axhline(1,color="k",lw=0.8,ls="--")
plt.xticks(x,[f"{H}y" for H in HOR]); plt.ylabel("Terminal value / invested (x)")
plt.title("DCA from 1999-03: terminal multiple at each horizon")
plt.legend(fontsize=8); plt.grid(alpha=.3,axis="y")
plt.tight_layout(); plt.savefig("../results/dca1999_multiple.png",dpi=120); plt.close()

# ---- 图3: 各horizon 最大回撤柱状 ----
plt.figure(figsize=(13,6.5))
for i,nm in enumerate(NAMES):
    vals=[T[T.H==H][nm+"_dd"].iloc[0] for H in HOR]
    bars=plt.bar(x+(i-2)*w,vals,w,color=COL[nm],label=nm)
plt.axhline(0,color="k",lw=0.8)
plt.xticks(x,[f"{H}y" for H in HOR]); plt.ylabel("Max portfolio drawdown (%)")
plt.title("DCA from 1999-03: max drawdown at each horizon (less negative = safer)")
plt.legend(fontsize=8); plt.grid(alpha=.3,axis="y")
plt.tight_layout(); plt.savefig("../results/dca1999_drawdown.png",dpi=120); plt.close()
print("\n-> results/dca1999_equity.png, dca1999_multiple.png, dca1999_drawdown.png")
