"""
SPOT (Streaming Peaks-Over-Threshold) Algorithm

Automatic threshold selection for anomaly detection using Extreme Value Theory.
Only the upper-bound SPOT variant is included (sufficient for TranAD anomaly scores).

Original author: Alban Siffer (Amossys)
License: GNU GPLv3
Adapted from: imperial-qore/TranAD reference implementation.
"""

from math import floor, log

import numpy as np
from scipy.optimize import minimize


class SPOT:
    """Upper-bound Streaming Peaks-Over-Threshold for univariate data.

    Fits a Generalized Pareto Distribution to excesses above an initial threshold,
    then uses the estimated GPD parameters to compute an extreme quantile that
    serves as the anomaly detection threshold.

    Attributes:
        proba: Detection level (risk), set by constructor.
        extreme_quantile: Current threshold (computed after initialize).
        data: Stream data (set by fit).
        init_data: Calibration data (set by fit).
        init_threshold: Initial threshold from calibration.
        peaks: Excesses above initial threshold.
        n: Number of observed values.
        Nt: Number of observed peaks.
    """

    def __init__(self, q: float = 1e-4):
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self) -> str:
        s = ""
        s += "Streaming Peaks-Over-Threshold Object\n"
        s += "Detection level q = %s\n" % self.proba
        if self.data is not None:
            s += "Data imported : Yes\n"
            s += "\t initialization  : %s values\n" % self.init_data.size
            s += "\t stream : %s values\n" % self.data.size
        else:
            s += "Data imported : No\n"
            return s

        if self.n == 0:
            s += "Algorithm initialized : No\n"
        else:
            s += "Algorithm initialized : Yes\n"
            s += "\t initial threshold : %s\n" % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += "Algorithm run : Yes\n"
                s += "\t number of observations : %s (%.2f %%)\n" % (
                    r,
                    100 * r / self.n,
                )
            else:
                s += "\t number of peaks  : %s\n" % self.Nt
                s += "\t extreme quantile : %s\n" % self.extreme_quantile
                s += "Algorithm run : No\n"
        return s

    def fit(self, init_data, data) -> None:
        """Import data to SPOT object.

        Args:
            init_data: Initial batch to calibrate the algorithm
                       (list, np.ndarray, int, or float).
            data: Data for the run (list or np.ndarray).
        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError("Unsupported data format: %s" % type(data))

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) and (0 < init_data < 1):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            raise TypeError("Unsupported init_data format: %s" % type(init_data))

    def initialize(
        self,
        level: float = 0.98,
        min_extrema: bool = False,
        verbose: bool = True,
    ) -> None:
        """Run the calibration (initialization) step.

        Args:
            level: Probability associated with the initial threshold t.
            min_extrema: If True, find min extrema instead of max extrema.
            verbose: If True, prints log.
        """
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level

        level = level - floor(level)

        n_init = self.init_data.size

        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]

        # initial peaks
        self.peaks = (
            self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        )
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print("Initial threshold : %s" % self.init_threshold)
            print("Number of peaks : %s" % self.Nt)
            print("Grimshaw maximum log-likelihood estimation ... ", end="")

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print("[done]")
            print("\t" + chr(0x03B3) + " = " + str(g))
            print("\t" + chr(0x03C3) + " = " + str(s))
            print("\tL = " + str(l))
            print(
                "Extreme quantile (probability = %s): %s"
                % (self.proba, self.extreme_quantile)
            )

    @staticmethod
    def _rootsFinder(fun, jac, bounds, npoints, method):
        """Find possible roots of a scalar function."""
        if method == "regular":
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            # Bug fix - Shreshth Tuli
            if step == 0:
                bounds, step = (0, 1e-4), 1e-5
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == "random":
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx**2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(
            lambda X: objFun(X, fun, jac),
            X0,
            method="L-BFGS-B",
            jac=True,
            bounds=[bounds] * len(X0),
        )

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    @staticmethod
    def _log_likelihood(Y, gamma, sigma):
        """Compute log-likelihood for the Generalized Pareto Distribution (mu=0)."""
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """Compute GPD parameters estimation with Grimshaw's trick."""

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s**2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym**2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(
            lambda t: w(self.peaks, t),
            lambda t: jac_w(self.peaks, t),
            (a + epsilon, -epsilon),
            n_points,
            "regular",
        )

        right_zeros = SPOT._rootsFinder(
            lambda t: w(self.peaks, t),
            lambda t: jac_w(self.peaks, t),
            (b, c),
            n_points,
            "regular",
        )

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """Compute the quantile at level 1-q for the GPD."""
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True, dynamic=True):
        """Run SPOT on the stream.

        Args:
            with_alarm: If False, SPOT will adapt the threshold assuming
                        there are no abnormal values.
            dynamic: If False, use init_threshold for alarm detection
                     instead of updating the extreme quantile.

        Returns:
            dict with 'thresholds' (list of extreme quantiles per step)
            and 'alarms' (indices of values that triggered alarms).
        """
        if self.n > self.init_data.size:
            print(
                "Warning: the algorithm seems to have already been run, "
                "you should initialize before running again"
            )
            return {}

        th = []
        alarm = []

        for i in range(self.data.size):
            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                if self.data[i] > self.extreme_quantile:
                    if with_alarm:
                        alarm.append(i)
                    else:
                        self.peaks = np.append(
                            self.peaks, self.data[i] - self.init_threshold
                        )
                        self.Nt += 1
                        self.n += 1
                        g, s, l = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)
                elif self.data[i] > self.init_threshold:
                    self.peaks = np.append(
                        self.peaks, self.data[i] - self.init_threshold
                    )
                    self.Nt += 1
                    self.n += 1
                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    self.n += 1

            th.append(self.extreme_quantile)

        return {"thresholds": th, "alarms": alarm}
