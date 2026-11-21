"""Daily bars for BYD (002594.SZ) from BaoStock (free, no account or token needed).

The 2024 project used the tsanghi.com API, which needs a paid token. BaoStock serves the
same unadjusted daily bars for free, so the pipeline can be reproduced by anyone.

    python src/data.py               # writes data/002594_daily.csv
"""
import argparse
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "002594_daily.csv"
START, END = "2011-06-30", "2024-06-25"   # same period as the original project


def fetch_daily(code="sz.002594", start=START, end=END, path=RAW_FILE):
    import baostock as bs
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
    try:
        rs = bs.query_history_k_data_plus(code, "date,open,high,low,close,volume,tradestatus",
                                          start_date=start, end_date=end, frequency="d", adjustflag="3")
        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock query failed: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df[df["tradestatus"] == "1"].drop(columns="tradestatus")       # drop suspended days
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"] / 100            # shares -> lots of 100, the unit used in the original data
    df.insert(0, "ticker", code.split(".")[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved {len(df)} rows to {path}")
    return path


def load_daily(path=RAW_FILE):
    """Read daily bars: ticker kept as a string (002594, not 2594), dates parsed, sorted, de-duplicated."""
    if not pathlib.Path(path).exists():
        fetch_daily(path=path)
    df = pd.read_csv(path, dtype={"ticker": str})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()
    fetch_daily(start=args.start, end=args.end)
