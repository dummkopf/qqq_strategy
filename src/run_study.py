"""
Main study: does DCA-ing into TQQQ over a ~10-year horizon make sense, and does
an SMA-200 risk-off overlay improve it?

Outputs (results/):
  * fixed_windows.csv  -- strategies over 10yr windows from QQQ/QLD/TQQQ inceptions
  * band_sensitivity.csv
  * rolling_summary.csv -- distribution of outcomes over ALL rolling 10yr windows
  * equity_2010.png, rolling_irr.png, drawdown_2010.png
And a printed report.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qqqlib as q
import strategy as st

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
CONTRIB = 1000.0  # $ per month


def build_returns(qqq):
    """Dict of daily total-return series over full QQQ history: real funds where
    they exist, simulated before that."""
    return {
        "QQQ":  qqq["tr_ret"],
        "QLD":  q.build_fund_total_return(qqq, "QLD"),
        "TQQQ": q.build_fund_total_return(qqq, "TQQQ"),
    }


def window_dates(qqq, start, years=10):
    start = pd.Timestamp(start)
    end = start + pd.DateOffset(years=years)
    idx = qqq.index
    return idx[(idx >= start) & (idx <= end)]


def run_one(returns, qqq, dates, aggressive="TQQQ", defensive=None, band=0.0):
    return st.run_dca(returns, dates, contribution=CONTRIB, aggressive=aggressive,
                      defensive=defensive, band=band, qqq_price=qqq["price_level"])


def summarize(res, label):
    return dict(
        strategy=label,
        invested=res["invested"],
        final=res["final"],
        multiple=res["final"] / res["invested"],
        irr_pct=res["irr"] * 100,
        max_dd_pct=res["max_drawdown"] * 100,
        switches=res["n_switches"],
    )


STRATS = [
    ("DCA QQQ (buy&hold)",            dict(aggressive="QQQ",  defensive=None)),
    ("DCA QLD (buy&hold)",            dict(aggressive="QLD",  defensive=None)),
    ("DCA TQQQ (buy&hold)",          dict(aggressive="TQQQ", defensive=None)),
    ("DCA TQQQ + SMA200->CASH",      dict(aggressive="TQQQ", defensive="CASH", band=0.02)),
    ("DCA TQQQ + SMA200->QQQ",       dict(aggressive="TQQQ", defensive="QQQ",  band=0.02)),
    ("DCA TQQQ + SMA200->QLD",       dict(aggressive="TQQQ", defensive="QLD",  band=0.02)),
]


def fixed_windows(returns, qqq):
    starts = {
        "QQQ inception 1999-03": "1999-03-10",
        "QLD inception 2006-06": "2006-06-21",
        "TQQQ inception 2010-02": "2010-02-11",
        "pre-2022-bear 2013-01": "2013-01-02",
        "recent 2015-01": "2015-01-02",
    }
    rows = []
    for wname, sdate in starts.items():
        dates = window_dates(qqq, sdate, 10)
        if len(dates) < 252 * 9:   # need ~full 10yr
            continue
        for label, kw in STRATS:
            res = run_one(returns, qqq, dates, **kw)
            r = summarize(res, label)
            r["window"] = wname
            r["start"] = dates[0].date()
            r["end"] = dates[-1].date()
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "fixed_windows.csv"), index=False)
    return df


def band_sensitivity(returns, qqq):
    dates = window_dates(qqq, "2010-02-11", 10)
    rows = []
    for band in [0.0, 0.01, 0.02, 0.03, 0.05]:
        for defen in ["CASH", "QQQ", "QLD"]:
            res = run_one(returns, qqq, dates, aggressive="TQQQ", defensive=defen, band=band)
            r = summarize(res, f"->{defen}")
            r["band_pct"] = band * 100
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "band_sensitivity.csv"), index=False)
    return df


def rolling_study(returns, qqq, years=10):
    """Every monthly start whose 10yr window fits in the data."""
    idx = qqq.index
    last_start = idx[-1] - pd.DateOffset(years=years)
    month_firsts = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first()
    starts = [d for d in month_firsts.values if pd.Timestamp(d) <= last_start]

    labels = ["DCA QQQ (buy&hold)", "DCA TQQQ (buy&hold)",
              "DCA TQQQ + SMA200->QQQ", "DCA TQQQ + SMA200->CASH"]
    kwmap = {
        "DCA QQQ (buy&hold)":      dict(aggressive="QQQ",  defensive=None),
        "DCA TQQQ (buy&hold)":     dict(aggressive="TQQQ", defensive=None),
        "DCA TQQQ + SMA200->QQQ":  dict(aggressive="TQQQ", defensive="QQQ",  band=0.02),
        "DCA TQQQ + SMA200->CASH": dict(aggressive="TQQQ", defensive="CASH", band=0.02),
    }
    recs = []
    for s in starts:
        dates = window_dates(qqq, s, years)
        if len(dates) < 252 * 9:
            continue
        row = {"start": pd.Timestamp(s).date()}
        for lab in labels:
            res = run_one(returns, qqq, dates, **kwmap[lab])
            row[lab + "|irr"] = res["irr"] * 100
            row[lab + "|mult"] = res["final"] / res["invested"]
            row[lab + "|dd"] = res["max_drawdown"] * 100
        recs.append(row)
    roll = pd.DataFrame(recs)
    roll.to_csv(os.path.join(RESULTS, "rolling_windows_raw.csv"), index=False)

    # summary stats per strategy
    out = []
    for lab in labels:
        irr = roll[lab + "|irr"]
        mult = roll[lab + "|mult"]
        dd = roll[lab + "|dd"]
        out.append(dict(
            strategy=lab,
            n_windows=len(roll),
            irr_min=irr.min(), irr_p25=irr.quantile(.25), irr_median=irr.median(),
            irr_p75=irr.quantile(.75), irr_max=irr.max(),
            mult_min=mult.min(), mult_median=mult.median(), mult_max=mult.max(),
            worst_dd=dd.min(), median_dd=dd.median(),
            pct_irr_negative=(irr < 0).mean() * 100,
        ))
    summ = pd.DataFrame(out)
    summ.to_csv(os.path.join(RESULTS, "rolling_summary.csv"), index=False)

    # how often does each TQQQ strategy beat plain QQQ DCA?
    base = roll["DCA QQQ (buy&hold)|irr"]
    beat = {lab: (roll[lab + "|irr"] > base).mean() * 100
            for lab in labels if lab != "DCA QQQ (buy&hold)"}
    return roll, summ, beat


def make_charts(returns, qqq, roll):
    dates = window_dates(qqq, "2010-02-11", 10)
    plt.figure(figsize=(11, 6))
    for label, kw in STRATS:
        res = run_one(returns, qqq, dates, **kw)
        plt.plot(res["value"].index, res["value"].values, label=f"{label} (×{res['final']/res['invested']:.1f})")
    # contributed line
    res0 = run_one(returns, qqq, dates, aggressive="QQQ", defensive=None)
    plt.plot(res0["contrib_cum"].index, res0["contrib_cum"].values, "k--", lw=1, label="Total contributed")
    plt.title("DCA $1,000/mo, 10yr from TQQQ inception (2010-02-11)")
    plt.ylabel("Portfolio value ($)"); plt.legend(fontsize=8); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "equity_2010.png"), dpi=110); plt.close()

    # rolling IRR distribution
    plt.figure(figsize=(11, 6))
    labels = ["DCA QQQ (buy&hold)", "DCA TQQQ (buy&hold)",
              "DCA TQQQ + SMA200->QQQ", "DCA TQQQ + SMA200->CASH"]
    for lab in labels:
        s = roll.set_index("start")[lab + "|irr"]
        plt.plot(pd.to_datetime(s.index), s.values, label=lab, lw=1.3)
    plt.axhline(0, color="k", lw=.6)
    plt.title("10-year DCA money-weighted return (IRR) by start month")
    plt.ylabel("IRR (%/yr)"); plt.xlabel("Window start"); plt.legend(fontsize=8); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "rolling_irr.png"), dpi=110); plt.close()


def main():
    qqq = q.load_qqq()
    returns = build_returns(qqq)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    fmt = lambda d: d.to_string(index=False, float_format=lambda x: f"{x:,.2f}")

    print("\n" + "=" * 90)
    print("FIXED 10-YEAR WINDOWS  ($1,000/month DCA)")
    print("=" * 90)
    fw = fixed_windows(returns, qqq)
    for w in fw["window"].unique():
        sub = fw[fw["window"] == w][["strategy", "invested", "final", "multiple", "irr_pct", "max_dd_pct", "switches"]]
        print(f"\n--- {w}  ({fw[fw.window==w].iloc[0]['start']} .. {fw[fw.window==w].iloc[0]['end']}) ---")
        print(fmt(sub))

    print("\n" + "=" * 90)
    print("BAND SENSITIVITY (TQQQ + SMA200, 2010 10yr window)")
    print("=" * 90)
    bs = band_sensitivity(returns, qqq)
    print(fmt(bs[["strategy", "band_pct", "final", "multiple", "irr_pct", "max_dd_pct", "switches"]]))

    print("\n" + "=" * 90)
    print("ROLLING 10-YEAR WINDOWS — distribution across ALL monthly start dates")
    print("=" * 90)
    roll, summ, beat = rolling_study(returns, qqq)
    print(fmt(summ[["strategy", "n_windows", "irr_min", "irr_median", "irr_max",
                    "mult_median", "worst_dd", "pct_irr_negative"]]))
    print("\n% of rolling windows each TQQQ strategy beats plain DCA-QQQ (by IRR):")
    for k, v in beat.items():
        print(f"   {k:32s}: {v:5.1f}%")

    make_charts(returns, qqq, roll)
    print(f"\nCharts + CSVs written to {os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
