import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getWeights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[:: -1]).reshape(-1, 1)
    return w

def getWeights_FFD(d, thres):
    w = [1.]
    k = 1
    while abs(w[-1]) >= thres:
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
        k += 1
    w = np.array(w[:: -1]).reshape(-1, 1)[1:]
    return w

def fracDiff_FFD(series, d, thres = 1e-5):
    w = getWeights_FFD(d, thres)
    # w = getWeights(d, series.shape[0])
    # w=getWeights_FFD(d,thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method = 'ffill').dropna()
        df_ = pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0 = seriesF.index[iloc1 - width]
            loc1 = seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_.loc[loc1] = np.dot(w.T, seriesF.loc[loc0: loc1])[0, 0]

        df[name] = df_.copy(deep = True)
    df = pd.concat(df, axis = 1)
    return df

def fracDiff(series, d, thres = .01):
    w = getWeights(d, series.shape[0])
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    df = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method = 'ffill').dropna()
        df_ = pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]

            test_val = series.loc[loc, name]
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample('1m').mean()
            if not np.isfinite(test_val).any():
                continue
            try:
                df_.loc[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
            except:
                continue
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

def get_bSADF(logP, minSL, constant, lags):
    y, x= getYX(logP, constant = constant, lags=lags)
    startPoints, bsadf, allADF = range(0, y.shape[0] + lags - minSL + 1), 0, []
    for start in startPoints:
        y_, x_ = y[start:], x[start:]
        bMean_, bStd_ = getBetas(y_, x_)
        bMean_, bStd_ = bMean_[0], bStd_[0, 0] ** .5
        allADF.append(bMean_ / bStd_)
        if allADF[-1] > bsadf : bsadf = allADF[-1]
    return bsadf

def lagDF(series, lags):
    df1 = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        df_ = series.shift(lag).copy(deep = True)
        df_.name = str(series.name) + '_' + str(lag)
        df1 = df1.join(df_, how = 'outer')
    return df1

def getYX(series, constant, lags):
    series_ = series.diff().dropna()
    x = lagDF(series_, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0] -1: -1]
    y = series_.iloc[-x.shape[0]:].values
    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis = 1)
        if constant[:2] == 'ct':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend, axis = 1)
        if constant == 'ctt':
            x = np.append(x, trend ** 2, axis = 1)
    return y, x

def getBetas(y, x):
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    if np.linalg.matrix_rank(xx) < x.shape[1]:
        pass
    else:
        xxinv = np.linalg.inv(xx)
        bMean = np.dot(xxinv, xy)
        err = y - np.dot(x, bMean)
        bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
        return bMean, bVar

def get_bSADF_test_statistics(logP, minSL, constant, lags):
    test_statistics = []
    for i in range(len(logP)):
        logP_ = logP.iloc[:i+1]
        bsadf = get_bSADF(logP_, minSL, constant, lags)
        test_statistics.append(bsadf)
    test_statistics = pd.Series(test_statistics)
    test_statistics.index = logP.index
    test_statistics.name = 'GSADF'
    return test_statistics

def pmf1(msg, w: int):
    lib = {}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    for i in range(w, len(msg)):
        msg_ = msg[i - w: i]
        if msg_ not in lib:
            lib[msg_] = [i - w]
        else:
            lib[msg_] = lib[msg_] + [i - w]
    length = float(len(msg) - w)
    pmf = {i: len(lib[i]) / length for i in lib}
    return pmf

def plug_in(msg, w: int):
    pmf = pmf1(msg, w)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / w
    return out, pmf

def lempel_ziv_lib(msg: str) -> list:
    i, lib = 1, [msg[0]]
    while i < len(msg):
        for j in range(i, len(msg)):
            msg_ = msg[i: j + 1]
            if msg_ not in lib:
                lib.append(msg_)
                break
        i = j + 1
    return lib

def match_length(msg: str, i: int, n: int):
    subS = ''
    for l in range(n):
        msg1 = msg[i: i + 1 + l]
        for j in range(i - n, i):
            msg0 = msg[j: j + 1 + l]
            if msg1 == msg0:
                subS = msg1
                break
    return len(subS) + 1, subS

def konto(msg, window = None) -> dict:
    out = {'num': 0, 'sum': 0, 'subS': []}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    if window is None:
        points = range(1, len(msg) // 2 + 1)
    else:
        window = min(window, len(msg) // 2)
        points = range(window, len(msg) - window + 1)
    for i in points:
        if window is None:
            l, msg_ = match_length(msg, i, i)
            out['sum'] += np.log2(i + 1) / l  # to avoid Doeblin condition
        else:
            l, msg_ = match_length(msg, i, window)
            out['sum'] += np.log2(window + 1) / l  # to avoid Doeblin condition
        out['subS'].append(msg_)
        out['num'] += 1
    out['h'] = out['sum'] / out['num']
    out['r'] = 1 - out['h'] / np.log2(len(msg))  # redundancy, 0 <= r <= 1
    return out

def generate_buckets(series: pd.Series, sigma: float) -> list:
    segments = []
    cnt = 1
    while series.min() + (cnt - 1) * sigma < series.max():
        segments.append([series.min() + (cnt - 1) * sigma, series.min() + cnt * sigma])
        cnt += 1
    return segments

def encode_single_obs(obs: float, segments: list) -> str:
    for i in range(len(segments)):
        if segments[i][0] <= obs < segments[i][1]:
            code = chr(48 + i)
            return code

def OptimizeBins(nObs, corr = None) :
    if corr is None :
        z = (8 + 324 * nObs + 12 * (36 * nObs + 729 * nObs ** 2)** 0.5)**(1/3)
        b = round(z/6 + 2/(3*z) + 1/3)
    else : b = round(2**(-0.5) * (1 + (1+24*nObs/(1-corr**2))**0.5)**0.5)
    return int(b)

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
    f, ax = plt.subplots(figsize=(15, 5))
    tick.plot(ax=ax, ls='-', label='tick count')
    volume.plot(ax=ax, ls='--', label='volume count')
    dollar.plot(ax=ax, ls='-.', label='dollar count')
    ax.set_title('Scaled Bar Counts')
    ax.legend()
    return

def plotSampleData(ref, sub, bar_type, *args, **kwds) :
    f, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 7))
    ref.plot(*args, **kwds, ax=axes[0], label='price')
    sub.plot(*args, **kwds, ax=axes[0], marker='X', ls='', label=bar_type)
    axes[0].legend()
    ref.plot(*args, **kwds, ax=axes[1], marker='o', label='price')
    sub.plot(*args, **kwds, ax=axes[2], marker='X', ls='',
             color='r', label=bar_type)
    for ax in axes[1:]: ax.legend()
    plt.tight_layout()
    return

def select_sample_data(ref, sub, price_col, date):
    xdf = ref[price_col].loc[date]
    xtdf = sub[price_col].loc[date]
    return xdf, xtdf