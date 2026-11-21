"""Compute indicators for BYD and draw the price / volume / MACD panel.

    python src/build_features.py     # data/002594_features.csv, figures/indicators_2022.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import DATA_DIR, ROOT, load_daily  # noqa: E402
from indicators import add_indicators  # noqa: E402

FEATURES = DATA_DIR / "002594_features.csv"
START, END = "2022-03-01", "2023-03-01"


def plot_panel(df, path):
    d = df[(df["date"] >= START) & (df["date"] <= END)]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1, 1.2]})
    ax1.plot(d["date"], d["close"], label="Close", lw=1.5)
    for col, lab in [("ma5", "MA5"), ("ma60", "MA60"), ("ma120", "MA120")]:
        ax1.plot(d["date"], d[col], label=lab, lw=1)
    ax1.fill_between(d["date"], d["bbl"], d["bbu"], alpha=0.12, label="Bollinger (20, 2 sd)")
    ax1.set_ylabel("Price (CNY)")
    ax1.legend(loc="upper left")
    ax1.set_title(f"BYD 002594.SZ, {START} to {END}")
    ax2.bar(d["date"], d["volume"], color="lightgray", label="Volume (lots)")
    ax2.plot(d["date"], d["ma_5_v"], label="MA5 volume", lw=1)
    ax2.plot(d["date"], d["ma_60_v"], label="MA60 volume", lw=1)
    ax2.legend(loc="upper left")
    colors = ["tab:red" if x > 0 else "tab:green" for x in d["MACD"]]
    ax3.bar(d["date"], d["MACD"], color=colors, label="MACD histogram 2(DIF - DEA)")
    ax3.plot(d["date"], d["DIF"], label="DIF", lw=1)
    ax3.plot(d["date"], d["DEA"], label="DEA", lw=1)
    ax3.axhline(0, color="black", lw=0.5)
    ax3.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    raw = load_daily()
    print(f"raw data: {len(raw)} rows, {raw['date'].min().date()} to {raw['date'].max().date()}")
    df = add_indicators(raw)
    df.to_csv(FEATURES, index=False)
    (ROOT / "figures").mkdir(exist_ok=True)
    plot_panel(df, ROOT / "figures" / "indicators_2022.png")
    print(f"saved {FEATURES} and figures/indicators_2022.png")


if __name__ == "__main__":
    main()
