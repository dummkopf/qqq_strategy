# qqq_strategy

Independent backtest: **does DCA-ing into TQQQ over ~10 years make sense, and
does a QQQ 200-day-SMA risk-off overlay help?**

**Read [`REPORT.md`](REPORT.md) for the findings and recommendation.**

## Layout
```
data/raw/        QQQ/QLD/TQQQ daily Yahoo CSVs (real, dividend-adjusted)
src/qqqlib.py    data loading + validated leveraged-ETF simulation
src/validate_letf.py   simulation vs real TQQQ/QLD (corr ~0.998, CAGR error <0.05%/yr)
src/strategy.py  DCA engine + SMA-200 risk-off overlay (band, switch costs, no look-ahead)
src/run_study.py main study -> results/*.csv + charts
results/         output tables and charts
```

## Run
```
pip install -r requirements.txt
python src/validate_letf.py
python src/run_study.py
```

## Key numbers (207 rolling 10-yr windows, $1k/mo DCA, money-weighted IRR)
| Strategy | Median IRR | Worst IRR | Worst drawdown | Beats QQQ-DCA |
|---|---:|---:|---:|---:|
| DCA QQQ | 15.9% | −7.6% | −52% | — |
| DCA TQQQ (buy&hold) | 33.6% | **−52.7%** | **−96.8%** | 82.1% |
| DCA TQQQ + SMA→QQQ (2–3% band) | 30.3% | −15.1% | −83.6% | 82.6% |
| DCA TQQQ + SMA→CASH | 23.6% | −5.6% | −81.2% | 74.4% |

The SMA-200 overlay is **tail insurance**: it gives up a little median return to
remove most of the catastrophic left tail. See the report for caveats (taxes,
behaviour, regime dependence).
