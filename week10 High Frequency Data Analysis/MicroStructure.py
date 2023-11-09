import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tick_rule(tick_prices : pd.Series):
    price_change = tick_prices.diff()
    aggressor = pd.Series(index=tick_prices.index, data=np.nan)
    aggressor.iloc[0] = 1.
    aggressor[price_change < 0] = -1.
    aggressor[price_change > 0] = 1.
    aggressor = aggressor.fillna(method = 'ffill')
    return aggressor

def volume_weighted_average_price(dollar_volume: list, volume: list) -> float:
    return sum(dollar_volume) / sum(volume)

def get_avg_tick_size(tick_size_arr: list) -> float:
    return np.mean(tick_size_arr)

class RollModel:
    def __init__(self, close_prices : pd.Series, window : int = 20) -> None:
        self.close_prices = close_prices
        self.window = window
    def roll_measure(self) -> pd.Series :
        price_diff = self.close_prices.diff()
        price_diff_lag = price_diff.shift(1)
        return 2 * np.sqrt(abs(price_diff.rolling(window = self.window).cov(price_diff_lag)))

    def roll_impact(self, dollar_volume : pd.Series) -> pd.Series :
        roll_measure = self.roll_measure()
        return roll_measure / dollar_volume

class CorwinSchultz :
    def __init__(self, high : pd.Series, low : pd.Series) -> None:
        self.high = high
        self.low = low
    def beta(self, window : int) -> pd.Series:
        ret = np.log(self.high / self.low)
        high_low_ret = ret ** 2
        beta = high_low_ret.rolling(window=2).sum()
        beta = beta.rolling(window=window).mean()
        return beta
    def gamma(self) -> pd.Series:
        high_max = self.high.rolling(window = 2).max()
        low_min = self.low.rolling(window = 2).min()
        gamma = np.log(high_max / low_min) ** 2
        return gamma
    def alpha(self, window : int) -> pd.Series:
        den = 3 - 2 * 2 ** .5
        alpha = (2 ** .5 - 1) * (self.beta(window = window) ** .5) / den
        alpha -= (self.gamma() / den) ** .5
        alpha[alpha < 0] = 0
        return alpha
    def corwin_schultz_estimator(self, window : int = 20) -> pd.Series :
        alpha_ = self.alpha(window = window)
        spread = 2 * (np.exp(alpha_) - 1) / (1 + np.exp(alpha_))
        start_time = pd.Series(self.high.index[0:spread.shape[0]], index=spread.index)
        spread = pd.concat([spread, start_time], axis=1)
        spread.columns = ['Spread', 'Start_Time']
        return spread.Spread
    def becker_parkinson_vol(self, window: int = 20) -> pd.Series:
        Beta = self.beta(window = window)
        Gamma = self.gamma()
        k2 = (8 / np.pi) ** 0.5
        den = 3 - 2 * 2 ** .5
        sigma = (2 ** -0.5 - 1) * Beta ** 0.5 / (k2 * den)
        sigma += (Gamma / (k2 ** 2 * den)) ** 0.5
        sigma[sigma < 0] = 0
        return sigma

def BarSampling(df, column, threshold, tick = False) :
    t = df[column]
    ts = 0
    idx = []
    if tick:
        for i, x in enumerate(t):
            ts += 1
            if ts >= threshold:
                idx.append(i)
                ts = 0
    else:
        for i, x in enumerate(t):
            ts += x
            if ts >= threshold:
                idx.append(i)
                ts = 0
    return df.iloc[idx].drop_duplicates()

def plot_bar_counts(tick, volume, dollar):
    f, ax = plt.subplots(figsize = (8, 4))
    tick.plot(ax=ax, ls='-', label='tick count')
    volume.plot(ax=ax, ls='--', label='volume count')
    dollar.plot(ax=ax, ls='-.', label='dollar count')
    ax.set_title('Scaled Bar Counts')
    ax.grid(False)
    ax.legend()
    return

def plotSampleData(ref, sub, bar_type, *args, **kwds) :
    f, axes = plt.subplots(3, sharex = True, sharey = True, figsize = (7, 5))
    ref.plot(*args, **kwds, ax = axes[0], label = 'price')
    sub.plot(*args, **kwds, ax = axes[0], marker = 'X', ls = '', label = bar_type)
    axes[0].legend()
    axes[0].grid(False)
    ref.plot(*args, **kwds, ax = axes[1], marker = 'o', label = 'price')
    sub.plot(*args, **kwds, ax = axes[2], marker = 'X', ls = '',
             color = 'r', label = bar_type)
    plt.grid(False)
    for ax in axes[1:]:
        ax.legend()
        ax.grid(False)
    plt.tight_layout()
    plt.grid(False)
    return

def select_sample_data(ref, sub, price_col, date):
    xdf = ref[price_col].loc[date]
    xtdf = sub[price_col].loc[date]
    return xdf, xtdf