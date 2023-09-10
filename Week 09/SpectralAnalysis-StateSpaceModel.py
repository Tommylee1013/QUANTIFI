import numpy as np
import pandas as pd
from numpy import pi
import matplotlib.pyplot as plt
from cycler import cycler
from filterpy.kalman import FixedLagSmoother, KalmanFilter
from filterpy.common import Q_discrete_white_noise
import statsmodels.api as sm

cols = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle'] = cycler(color = cols)

def plot_2d(m, title = '') :
    plt.imshow(m)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
class SingularSpectrum(object):
    __supported_types = (pd.Series, np.ndarray, list)

    def __init__(self, tseries, L, save_mem=True):
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        self.N = len(tseries)
        if not 2 <= L <= self.N / 2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1
        self.X = np.array([self.orig_TS.values[i:L + i] for i in range(0, self.K)]).T
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)

        self.TS_comps = np.zeros((self.N, self.d))

        if not save_mem:
            self.X_elem = np.array([self.Sigma[i] * np.outer(self.U[:, i], VT[i, :]) for i in range(self.d)])
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

            self.V = VT.T
        else:
            for i in range(self.d):
                X_elem = self.Sigma[i] * np.outer(self.U[:, i], VT[i, :])
                X_rev = X_elem[::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            self.V = "Re-run with save_mem=False to retain the V matrix."
        self.calc_wcorr()

    def components_to_df(self, n = 0):
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)

    def reconstruct(self, indices):
        if isinstance(indices, int): indices = [indices]

        ts_vals = self.TS_comps[:, indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)

    def calc_wcorr(self):
        w = np.array(list(np.arange(self.L) + 1) + [self.L] * (self.K - self.L - 1) + list(np.arange(self.L) + 1)[::-1])
        def w_inner(F_i, F_j):
            return w.dot(F_i * F_j)
        F_wnorms = np.array([w_inner(self.TS_comps[:, i], self.TS_comps[:, i]) for i in range(self.d)])
        F_wnorms = F_wnorms ** -0.5
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                self.Wcorr[i, j] = abs(w_inner(self.TS_comps[:, i], self.TS_comps[:, j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j, i] = self.Wcorr[i, j]

    def plot_wcorr(self, min=None, max=None):
        if min is None:
            min = 0
        if max is None:
            max = self.d
        if self.Wcorr is None:
            self.calc_wcorr()
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0, 1)
        if max == self.d:
            max_rnge = self.d - 1
        else:
            max_rnge = max

        plt.xlim(min - 0.5, max_rnge + 0.5)
        plt.ylim(max_rnge + 0.5, min - 0.5)
class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        k_states = k_posdef = 2
        super(LocalLinearTrend, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states
        )
        self.ssm['design'] = np.array([1, 0])
        self.ssm['transition'] = np.array([[1, 1],
                                           [0, 1]])
        self.ssm['selection'] = np.eye(k_states)
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ["sigma2.measurement", "sigma2.level", "sigma2.trend"]

    @property
    def start_params(self):
        return [np.std(self.endog)]*3

    def transform_params(self, unconstrained):
        return unconstrained ** 2

    def untransform_params(self, constrained):
        return constrained ** 0.5

    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)
        # Observation covariance
        self.ssm['obs_cov',0,0] = params[0]
        # State covariance
        self.ssm[self._state_cov_idx] = params[1:]
def calculate_rts(data, noise : int = 1 , Q : float = 0.001) -> pd.DataFrame :

    fk = KalmanFilter(dim_x=2, dim_z=1)

    fk.x = np.array([0., 1.])

    fk.F = np.array([[1., 1.],
                     [0., 1.]])

    fk.H = np.array([[1., 0.]])
    fk.P*= 10.
    fk.R = noise
    fk.Q = Q_discrete_white_noise(dim=2, dt=1., var=Q)
    zs = data["TAVG"]
    mu, cov, _, _ = fk.batch_filter(zs)
    M, P, C, _ = fk.rts_smoother(mu, cov)

    result = pd.DataFrame({"Measurement": zs,
                           "RTS Smoother": M[:, 0],
                           "Kalman Filter": mu[:, 0]})
    return result
def calculate_fl(data, N = 4) -> pd.DataFrame :
    fls = FixedLagSmoother(dim_x=2, dim_z=1, N=N)
    fls.x = np.array([0., .5])
    fls.F = np.array([[1., 1.],
                      [0., 1.]])

    fls.H = np.array([[1., 0.]])
    fls.P *= 200
    fls.R *= 5.
    fls.Q *= 0.001

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0., .5])
    kf.F = np.array([[1., 1.],
                     [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 200
    kf.R *= 5.
    kf.Q = Q_discrete_white_noise(dim=2, dt=1., var=0.001)

    zs = data["TAVG"]
    nom = np.array([t / 2. for t in range(len(zs))])

    for z in zs:
        fls.smooth(z)

    kf_x, _, _, _ = kf.batch_filter(zs)
    x_smooth = np.array(fls.xSmooth)[:, 0]

    fls_res = abs(x_smooth - nom)
    kf_res = abs(kf_x[:, 0] - nom)

    result = pd.DataFrame({"Measurement": zs,
                           "FL Smoother": x_smooth,
                           "Kalman Filter": kf_x[:, 0]},
                          index=data.index)
    return result