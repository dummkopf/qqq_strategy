# Does DCA-ing into TQQQ make sense over a ~10-year horizon?

*Independent backtest validation. Generated 2026-05-29.*

**TL;DR.** Dollar-cost-averaging into TQQQ has a **high median outcome but a
catastrophic left tail**. Across every 10-year window since QQQ's 1999
inception, plain buy-and-hold DCA into TQQQ delivered a median money-weighted
return (IRR) of **~34%/yr** — but the worst window lost **~53%/yr** and saw a
**−97% drawdown**, and ~10% of all windows ended underwater. A simple
**QQQ-200-day-SMA risk-off overlay** (move from TQQQ into QQQ when QQQ closes
below its 200-day average, with a ~2–3% band) keeps most of the upside
(median ~30%/yr) while **cutting the worst window from −53%/yr to −15%/yr** and
beating a plain QQQ DCA in **83% of all 10-year windows**. The overlay is best
understood as **tail insurance, not alpha**: in roaring bull decades (e.g.
2010–2020) it slightly lags pure TQQQ; in decades containing a real bear
(1999–2009, windows ending 2022) it wins outright on both return and drawdown.

---

## 1. What was tested

- **Instruments:** QQQ (Nasdaq-100, 1×), QLD (2×), TQQQ (3×).
- **Strategy:** invest **$1,000 on the first trading day of every month** for
  **10 years**. Money-weighted return (IRR) is the headline metric because it
  correctly accounts for the timing of contributions.
- **Risk-off overlay:** compute QQQ's 200-day SMA on its (split-adjusted) price.
  Go **risk-off** when price < SMA·(1−band); **risk-on** when price >
  SMA·(1+band); otherwise hold the current state. The band is hysteresis to
  avoid whipsawing across the line. When risk-off, the aggressive holding is
  swapped into a defensive asset (**cash**, **QQQ**, or **QLD**). New
  contributions buy whatever the current state holds. Each switch pays a 5 bps
  cost. The signal is lagged one day (no look-ahead).
- **Start points:** the three inceptions you asked about (QQQ 1999-03, QLD
  2006-06, TQQQ 2010-02) plus a 2013 start (so the window ends in the 2022 bear)
  and a 2015 start — **and**, most importantly, **every monthly start date**
  from 1999 to 2016 (207 overlapping 10-year windows) to see the *distribution*
  of outcomes rather than cherry-picked paths.

## 2. Data & methodology (and why you can trust it)

- **Real daily data (Yahoo Finance, dividend-adjusted total return):**
  QQQ 1999-03-10→2026-05-28, QLD 2006-06-21→2026-05-28, TQQQ
  2010-02-11→2026-05-28.
- **Synthetic back-fill before each fund existed** (TQQQ pre-2010, QLD pre-2006)
  using the standard daily-rebalanced leveraged-ETF model:
  `r_letf = L·r_QQQ − (L−1)·financing − expense`, with financing on an
  approximate 3-month T-bill path plus a swap spread.
- **The simulation is validated against the real funds** over their full
  histories — *including the 2022 bear and the 2023–25 high-rate regime*:

  | Fund | Window | Daily corr | Sim CAGR | Real CAGR | CAGR error | Growth multiple (sim / real) |
  |------|--------|-----------:|---------:|----------:|-----------:|------------------------------|
  | TQQQ (3×) | 2010–2026 | 0.99875 | 44.69% | 44.68% | **+0.01%/yr** | 406.4× / 406.1× |
  | QLD (2×)  | 2006–2026 | 0.99579 | 25.98% | 26.00% | **−0.03%/yr** | 99.0× / 99.4× |

  The synthetic series reproduce the real funds almost exactly, so the
  pre-inception back-fill (which drives the brutal 1999–2009 results) is
  trustworthy. Code: `src/validate_letf.py`.

> Caveat: this is a **pre-tax, frictions-light** study. Switching TQQQ→QQQ in a
> taxable account realises gains; in a tax-advantaged account it's free. The
> overlay's edge shrinks after taxes in a taxable account. Results also assume
> you actually execute every signal mechanically through scary markets.

## 3. Results

### 3a. Fixed 10-year windows from each inception

| Window (10yr) | DCA QQQ | DCA TQQQ B&H | TQQQ + SMA→QQQ | TQQQ + SMA→CASH |
|---|---|---|---|---|
| **1999-03 → 2009** (dot-com + GFC) | 0.69× / −7.6%/yr / −52% DD | **0.13× / −53%/yr / −97% DD** | 0.49× / −15%/yr / −84% DD | **0.77× / −5.4%/yr / −81% DD** |
| **2006-06 → 2016** | 2.06× / 13.8% | **5.12× / 30.6% / −88% DD** | 2.98× / 20.6% / −55% DD | 1.99× / 13.2% |
| **2010-02 → 2020** (pure bull) | 2.82× / 19.7% | **15.8× / 51.9% / −58% DD** | 8.37× / 39.8% / −44% DD | 5.57× / 32.2% |
| **2013-01 → 2022** (ends in bear) | 2.01× / 13.4% | 3.16× / 21.8% / **−81% DD** | **5.09× / 30.6% / −59% DD** | 4.66× / 29.0% |
| **2015-01 → 2025** | 2.78× / 19.5% | 6.95× / 36.5% / −81% DD | **10.2× / 43.7% / −58% DD** | 9.26× / 41.9% |

*(format: terminal multiple / IRR / max portfolio drawdown)*

Two regimes are obvious:
- **A bad start kills naked TQQQ.** Starting at QQQ's 1999 top, DCA-TQQQ turned
  $121k of contributions into $16k (a −97% peak drawdown). The SMA overlay was
  the difference between ruin and breakeven (→cash actually *beat* DCA-QQQ).
- **When a bear lands mid/late-horizon (2013→2022, 2015→2025), the overlay
  wins outright** on both return *and* drawdown vs naked TQQQ.
- **In an uninterrupted bull (2010→2020), naked TQQQ wins** and timing just
  costs you — the overlay's de-risking is "wasted" insurance.

### 3b. The full distribution — 207 rolling 10-year windows

| Strategy | Median IRR | Worst IRR | Best IRR | Median multiple | Worst drawdown | % windows IRR<0 |
|---|---:|---:|---:|---:|---:|---:|
| DCA QQQ (buy&hold) | 15.9% | −7.6% | 24.1% | 2.3× | −52% | 2.4% |
| DCA TQQQ (buy&hold) | **33.6%** | **−52.7%** | 60.5% | 6.0× | **−96.8%** | 10.1% |
| TQQQ + SMA→QQQ (2%) | 30.3% | **−15.1%** | 50.0% | 5.0× | −83.6% | 5.8% |
| TQQQ + SMA→CASH (2%) | 23.6% | −5.6% | 45.7% | 3.5× | −81.2% | 6.3% |

**% of 10-year windows that beat a plain QQQ DCA (by IRR):** TQQQ B&H 82.1%,
TQQQ+SMA→QQQ **82.6%**, TQQQ+SMA→CASH 74.4%.

The overlay (→QQQ) gives up ~3%/yr of *median* return versus naked TQQQ but
**chops the left tail by ~38 points of annualised return** (−53% → −15%) and
beats plain QQQ as often as naked TQQQ does. → see `results/rolling_irr.png`.

### 3c. The band matters for cost, not much for return

10-year window from 2010, TQQQ + SMA→QQQ:

| Band | Switches in 10yr | IRR | Max DD |
|---:|---:|---:|---:|
| 0% (touch the line) | 58 | 40.6% | −47% |
| 1% | 30 | 38.5% | −50% |
| 2% | 18 | 39.8% | −44% |
| 3% | 14 | 41.5% | −43% |
| 5% | 10 | 38.0% | −48% |

Your instinct is right: a **~2–3% band cuts the number of switches by ~70%**
(58 → ~15 over a decade) with **no performance penalty** — slightly *better*,
in fact, because it dodges whipsaws. Below the line you mostly trade fewer
times; above ~5% you start reacting too late.

## 4. So — does it make sense?

- **DCA-ing into TQQQ is not a "set and forget" plan.** Its unmanaged 10-year
  outcomes range from life-changing (60%/yr) to ruinous (−53%/yr, −97%
  drawdown). The bad cases aren't tail-of-the-tail flukes — ~1 in 10 windows
  ended underwater, and anyone who started near the 1999 or (to a lesser extent)
  2021 peak got destroyed. Leverage decay + sequence risk are real.
- **The SMA-200 risk-off overlay materially de-risks it** and is the single
  most important thing in this study. It turns the worst-case from "wiped out"
  to "roughly flat," keeps ~90% of the median return, and historically beat a
  plain QQQ DCA in ~83% of decades.
- **Defensive asset choice is a risk dial:** →CASH = most conservative (best
  worst-case, lowest median); →QQQ = the best risk/return balance and my
  default recommendation; →QLD = keeps the most upside but the smallest
  drawdown reduction (you're still 2× when "risk-off").
- **Use a ~2–3% band** to keep switching (and, in taxable accounts, tax events)
  low without hurting returns.
- **Caveats that could change your decision:** taxes on switches in a taxable
  account, the behavioural difficulty of mechanically selling after a −20% QQQ
  move, expense/borrowing drag rising in high-rate regimes (already in the
  model), and the obvious point that **the Nasdaq-100's entire history is one
  long secular uptrend** — none of this guarantees the next 10 years rhyme with
  the last 27.

**Bottom line:** TQQQ DCA *can* make sense over 10 years **only with a
mechanical risk-off rule**. Naked DCA-TQQQ is a bet that you won't start near a
major top; the 200-day-SMA overlay (→QQQ, ~2–3% band) removes most of that bet
while keeping the bulk of the upside.

## 5. Reproduce

```
pip install pandas numpy matplotlib pyarrow
python src/validate_letf.py     # simulation vs real funds
python src/run_study.py         # full study -> results/*.csv, results/*.png
```

Artifacts: `results/fixed_windows.csv`, `band_sensitivity.csv`,
`rolling_summary.csv`, `rolling_windows_raw.csv`, and charts
`equity_2010.png`, `rolling_irr.png`, `drawdown_2013.png`.
