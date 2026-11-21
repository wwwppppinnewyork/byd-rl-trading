"""Train the DQN on BYD and backtest it out of sample, over several random seeds.

    python src/build_features.py
    python src/train_dqn.py --seeds 10      # results/seeds.csv, results/summary.md, figures/*.png

Split by time: the first 1500 trading days (after the 21-day warm-up) train the agent, the
rest is the backtest. Each seed trains a fresh agent for 30 passes over the training data
and is then run greedily on the test period, with and without the MA21 trend filter.
One seed is not evidence, so the summary reports the spread across seeds next to buy and hold.
"""
import argparse

import numpy as np
import pandas as pd

from build_features import FEATURES
from data import ROOT
from dqn_agent import DQNAgent
from stock_env import StockEnv

SPLIT = 1500


def run_episode(agent, env, train, use_filter=False, step=0):
    obs = env.reset()
    done = False
    while not done:
        action = agent.choose_action(obs, train)
        obs_, reward, done = env.step(action, use_filter=use_filter)
        agent.store_transition(obs, action, reward, obs_)
        if train and step > 200 and step % 5 == 0:
            agent.learn()
        obs = obs_
        step += 1
    return step


def stats(curve):
    nav = 1 + np.asarray(curve)
    rets = np.diff(nav) / nav[:-1]
    years = len(nav) / 252
    return {"total": nav[-1] - 1, "ann": nav[-1] ** (1 / years) - 1,
            "sharpe": rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan,
            "max_dd": (nav / np.maximum.accumulate(nav) - 1).min()}


def main(n_seeds, epochs):
    df = pd.read_csv(FEATURES, parse_dates=["date"]).dropna(subset=["close", "ma21"]).reset_index(drop=True)
    train_df, test_df = df.iloc[:SPLIT], df.iloc[SPLIT:].reset_index(drop=True)
    print(f"train {train_df['date'].iloc[0].date()} to {train_df['date'].iloc[-1].date()}, "
          f"test {test_df['date'].iloc[0].date()} to {test_df['date'].iloc[-1].date()}")
    out, figs = ROOT / "results", ROOT / "figures"
    out.mkdir(exist_ok=True)
    rows = []
    for seed in range(n_seeds):
        env = StockEnv(train_df)
        agent = DQNAgent(env.n_actions, env.n_features, seed=seed)
        step = 0
        for _ in range(epochs):
            step = run_episode(agent, env, train=True, step=step)
        test_env = StockEnv(test_df)
        row = {"seed": seed}
        for name, use_filter in [("dqn", False), ("dqn_ma21", True)]:
            run_episode(agent, test_env, train=False, use_filter=use_filter)
            st = stats(test_env.profit_rate_account)
            row.update({f"{name}_{k}": v for k, v in st.items()})
            row[f"{name}_trades"] = len(test_env.states_buy) + len(test_env.states_sell)
            if seed == 0:
                label = "DQN" if not use_filter else "DQN + MA21 filter"
                test_env.draw(test_df["date"].values, figs / f"{name}_trades.png", figs / f"{name}_profit.png",
                              title=f"{label}, seed 0,")
        bh = stats(test_env.profit_rate_stock)
        row.update({f"buy_hold_{k}": v for k, v in bh.items()})
        rows.append(row)
        print(f"seed {seed}: DQN {row['dqn_total']:.1%}, DQN+MA21 {row['dqn_ma21_total']:.1%}, "
              f"buy and hold {row['buy_hold_total']:.1%}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "seeds.csv", index=False)

    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    lines = [f"# DQN backtest on BYD (002594.SZ), {n_seeds} seeds", "",
             f"Train {train_df['date'].iloc[0].date()} to {train_df['date'].iloc[-1].date()}; "
             f"test {test_df['date'].iloc[0].date()} to {test_df['date'].iloc[-1].date()} "
             f"({len(test_df)} trading days). Historical backtest, net of commission and stamp duty.", "",
             "| Strategy | Median total return | Worst seed | Best seed | Seeds beating buy and hold | Median Sharpe | Median max drawdown | Median trades |",
             "|---|---|---|---|---|---|---|---|"]
    for name, label in [("dqn", "DQN"), ("dqn_ma21", "DQN + MA21 trend filter")]:
        beat = (res[f"{name}_total"] > res["buy_hold_total"]).sum()
        lines.append(f"| {label} | {pct(res[f'{name}_total'].median())} | {pct(res[f'{name}_total'].min())} | "
                     f"{pct(res[f'{name}_total'].max())} | {beat} / {n_seeds} | {res[f'{name}_sharpe'].median():.2f} | "
                     f"{pct(res[f'{name}_max_dd'].median())} | {res[f'{name}_trades'].median():.0f} |")
    b = res.iloc[0]
    lines.append(f"| Buy and hold | {pct(b['buy_hold_total'])} | | | | {b['buy_hold_sharpe']:.2f} | {pct(b['buy_hold_max_dd'])} | 1 |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    main(args.seeds, args.epochs)
