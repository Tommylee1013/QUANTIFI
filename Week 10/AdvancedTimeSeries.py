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
    return test_statistics