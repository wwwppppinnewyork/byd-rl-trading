"""Single-stock trading environment for a DQN agent.

Adapted from an open-source DQN stock-trading example by "hcb" (2020); see the README.

State: the last ``window_size`` daily returns. Actions: 0 hold / wait, 1 buy with all cash
(in lots of 100 shares), 2 sell everything. Costs: 0.03% commission each way with a 5 CNY
minimum, 0.1% stamp duty on sales. Trades happen at the close of day t; the reward comes
from the move between t and t + 1.

Fixes against the first version: the reward-shaping branch for 0.03 < |r| < 0.05 was
missing (the previous step's reward leaked through), shaped rewards are now used in every
state, and a buy that cannot cover its commission no longer buys 0 shares for a 5 CNY fee.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def shaped_return(r):
    """Compress a daily return into a reward: damp small moves, cap the part beyond 5%."""
    a = abs(r)
    if a <= 0.015:
        return r * 0.2
    if a <= 0.03:
        return r * 0.7
    if a < 0.05:
        return r
    return (r - 0.05) * 0.1 + 0.05 if r > 0 else (r + 0.05) * 0.1 - 0.05


class StockEnv:
    n_actions = 3

    def __init__(self, df, init_money=10000, window_size=6):
        self.df = df.reset_index(drop=True)
        self.trend = self.df["close"].values
        self.init_money = init_money
        self.window_size = window_size
        self.n_features = window_size
        self.half_window = window_size // 2
        self.buy_rate, self.buy_min = 0.0003, 5
        self.sell_rate, self.sell_min = 0.0003, 5
        self.stamp_duty = 0.001

    def reset(self):
        self.hold_money = self.init_money
        self.buy_num = self.hold_num = 0
        self.market_value = self.init_money
        self.total_profit = 0
        self.t = self.half_window
        self.reward = 0
        self.states_sell, self.states_buy = [], []
        self.profit_rate_account, self.profit_rate_stock = [], []
        return self.get_state(self.t)

    def get_state(self, t):
        w = self.window_size + 1
        d = t - w + 1
        block = [self.trend[0]] * (-d) + list(self.trend[0:t + 1]) if d < 0 else self.trend[d:t + 1]
        return np.array([(block[i + 1] - block[i]) / (block[i] + 1e-4) for i in range(w - 1)])

    @staticmethod
    def _fee(money, rate, minimum):
        return max(money * rate, minimum)

    def buy_stock(self):
        price = self.trend[self.t]
        lots = int(self.hold_money // price // 100)
        while lots > 0 and price * lots * 100 + self._fee(price * lots * 100, self.buy_rate, self.buy_min) > self.hold_money:
            lots -= 1
        if lots <= 0:
            return False
        self.buy_num = lots * 100
        cost = price * self.buy_num
        self.hold_num += self.buy_num
        self.hold_money -= cost + self._fee(cost, self.buy_rate, self.buy_min)
        self.states_buy.append(self.t)
        return True

    def sell_stock(self):
        money = self.hold_num * self.trend[self.t]
        self.hold_money += money - self._fee(money, self.sell_rate, self.sell_min) - self.stamp_duty * money
        self.hold_num = 0
        self.states_sell.append(self.t)

    def trend_filter(self):
        """Rule overlay: only hold the stock while the close is at or above its 21-day average."""
        return self.df["close"][self.t] >= self.df["ma21"][self.t]

    def step(self, action, use_filter=False):
        price = self.trend[self.t]
        can_afford = self.hold_money >= price * 100 + max(self.buy_min, price * 100 * self.buy_rate)
        if action == 1 and can_afford and self.t < len(self.trend) - self.half_window:
            if not (use_filter and not self.trend_filter()):
                self.buy_stock()
        elif action == 2 and self.hold_num > 0:
            self.sell_stock()
        elif use_filter and self.hold_num > 0 and not self.trend_filter():
            self.sell_stock()

        self.market_value = price * self.hold_num + self.hold_money
        self.total_profit = self.market_value - self.init_money
        shaped = shaped_return((self.trend[self.t + 1] - price) / price)
        if self.hold_num > 0:
            self.reward = shaped            # holding: earn the next day's move
        elif action == 2:
            self.reward = -shaped           # just sold: a fall afterwards is good
        else:
            self.reward = -shaped * 0.1     # flat: small penalty for missing a rise
        self.profit_rate_account.append(self.total_profit / self.init_money)
        self.profit_rate_stock.append((price - self.trend[0]) / self.trend[0])
        self.t += 1
        done = self.t == len(self.trend) - 2
        return self.get_state(self.t), self.reward, done

    def draw(self, dates, path_trades, path_profit, title=""):
        x = dates[self.half_window:self.half_window + len(self.profit_rate_account)]
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(dates, self.trend, color="tab:red", lw=1.5)
        ax.plot(dates[self.states_sell], self.trend[self.states_sell], "v", ms=7, color="k", label="sell")
        ax.plot(dates[self.states_buy], self.trend[self.states_buy], "^", ms=7, color="m", label="buy")
        ax.set_title(f"{title} trades, total return {self.total_profit / self.init_money:.1%}")
        ax.set_ylabel("Close (CNY)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path_trades, dpi=110)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(x, self.profit_rate_account, label="agent")
        ax.plot(x, self.profit_rate_stock, label="buy and hold")
        ax.set_ylabel("Cumulative return")
        ax.set_title(f"{title} cumulative return")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path_profit, dpi=110)
        plt.close(fig)
