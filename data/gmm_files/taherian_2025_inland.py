# -*- coding: utf-8 -*-
"""
Taherian et al. (2025) ground-motion model for inland SW Iberia (Portugal), rock sites
(reference Vs30 = 760 m/s, no site term). Developed from EXSIM stochastic simulations.

Inputs: mag, rjb (km), hypo_depth (km) and rake.
Intensity measures: PGA, PGV and SA (periods 0.02-4 s).
Range of application: Mw 3.0-8.5, Rjb 0-600 km.
Output: median (g for PGA and SA, cm/s for PGV) and standard deviations (total, inter-event, intra-event).
"""
import numpy as np

from openquake.hazardlib.gsim.base import GMPE
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA

MH, H0, H1 = (5.61634653, 15.21176912, 1.82601032)
DEPTH_RANGE = (2.024, 29.729)
SA_PERIODS = np.array([0.02      , 0.04      , 0.05      , 0.06666667, 0.1       , 0.125     , 0.2       , 0.25      , 0.4       ,
 0.5       , 0.625     , 0.76923077, 1.        , 1.25      , 1.49253731, 2.        , 3.125     , 4.        ])
BETA_SA = np.array([[-9.63772867e-02,  3.68097297e-01, -8.96779215e-01, -1.43317298e+00,  2.64061879e-01, -6.55211534e-03,
  -4.80967223e-02, -2.49182068e-02],
 [ 3.99818497e-01,  3.39975529e-01, -9.07961163e-01, -1.43362535e+00,  2.51175511e-01, -7.15510119e-03,
  -4.61154840e-02, -2.40903014e-02],
 [ 5.44520735e-01,  3.16590553e-01, -9.45133056e-01, -1.41309133e+00,  2.59647335e-01, -7.35890740e-03,
  -4.44005130e-02, -2.38263313e-02],
 [ 4.51725462e-01,  3.28451248e-01, -9.59726489e-01, -1.38148285e+00,  2.66698204e-01, -7.56802449e-03,
  -4.22921157e-02, -2.34526071e-02],
 [-8.04418014e-02,  4.09207708e-01, -9.57642329e-01, -1.34917081e+00,  2.73237806e-01, -7.49623905e-03,
  -4.08128069e-02, -2.29809665e-02],
 [-1.27895365e+00,  6.18047116e-01, -9.68050289e-01, -1.31810505e+00,  2.36341465e-01, -7.73800828e-03,
  -3.99597442e-02, -2.26679314e-02],
 [-4.84140706e+00,  1.20359924e+00, -1.03686095e+00, -1.18473983e+00,  1.26841572e-01, -7.77838866e-03,
  -3.75912484e-02, -2.11331537e-02],
 [-6.51719460e+00,  1.47379644e+00, -1.08105325e+00, -1.13584426e+00,  8.09074416e-02, -7.32752561e-03,
  -3.65870271e-02, -2.02944612e-02],
 [-9.57491035e+00,  1.94913390e+00, -1.29042632e+00, -1.08004505e+00,  2.06323781e-02, -5.95288288e-03,
  -3.59368382e-02, -1.90716352e-02],
 [-1.09962951e+01,  2.17204899e+00, -1.38742307e+00, -1.07631240e+00, -5.79615337e-03, -5.41148570e-03,
  -3.54831757e-02, -1.87734328e-02],
 [-1.21715881e+01,  2.34274773e+00, -1.41681183e+00, -1.08194621e+00, -2.05443744e-02, -4.80536376e-03,
  -3.59082440e-02, -1.86245663e-02],
 [-1.31738475e+01,  2.47879618e+00, -1.47555822e+00, -1.08566154e+00, -2.95363341e-02, -4.40480144e-03,
  -3.64996801e-02, -1.85840957e-02],
 [-1.41873815e+01,  2.58391714e+00, -1.50350191e+00, -1.07389664e+00, -2.77233415e-02, -4.13078075e-03,
  -3.63944240e-02, -1.86970308e-02],
 [-1.45917883e+01,  2.58005465e+00, -1.48363278e+00, -1.07014450e+00, -5.74642521e-03, -3.83383275e-03,
  -3.64421999e-02, -1.88858080e-02],
 [-1.49278651e+01,  2.58828400e+00, -1.47173450e+00, -1.08272410e+00,  5.28219624e-03, -3.57482335e-03,
  -3.65877701e-02, -1.90607431e-02],
 [-1.50901865e+01,  2.53151412e+00, -1.42924122e+00, -1.12395815e+00,  3.70068707e-02, -2.93513117e-03,
  -3.72449923e-02, -1.93375560e-02],
 [-1.52403916e+01,  2.36999804e+00, -1.22244696e+00, -1.10844740e+00,  9.16357728e-02, -2.88200020e-03,
  -3.67469422e-02, -1.97496103e-02],
 [-1.50575872e+01,  2.23538901e+00, -1.10212092e+00, -1.11594061e+00,  1.30213104e-01, -2.69045303e-03,
  -3.66163484e-02, -1.99894450e-02]])          # (n_periods, 8): c0..c7
BETA_PGA = np.array([-0.33477974,  0.3728995 , -0.91586721, -1.42158554,  0.27986772, -0.00621012, -0.04760371, -0.02514206])
BETA_PGV = np.array([-1.41461603,  1.23499228, -1.16868123, -1.33209984,  0.18400228, -0.00405178, -0.04382787, -0.01995023])
TAU_SA = np.array([0.58679710, 0.59419353, 0.58154677, 0.57152625, 0.55299632, 0.54728642, 0.54002549, 0.53969797, 0.53974868, 0.53643741, 0.53110028, 0.53296488, 0.53004941, 0.53250630, 0.52968879, 0.52975678, 0.52447035, 0.51929067])
PHI_SA = np.array([0.25042097, 0.25785168, 0.25699143, 0.25531389, 0.25187068, 0.24949612, 0.24016093, 0.23873602, 0.23471087, 0.23273434, 0.23292624, 0.23417217, 0.23703762, 0.24107611, 0.24484518, 0.25121847, 0.26028397, 0.26404791])
TAU_PGA, PHI_PGA = 0.56053496, 0.24366911
TAU_PGV, PHI_PGV = 0.53643741, 0.24920382


def _design(mag, rjb, dep, fm):
    h = H0 + H1 * np.clip(mag - 5.0, 0, None)
    lr = 0.5 * np.log(rjb ** 2 + h ** 2)
    return np.column_stack([np.ones_like(mag), mag, np.clip(mag - MH, 0, None), lr,
                            (mag - 5.0) * lr, rjb, fm, dep])


def _interp(period, table):
    """table (n_periods,) or (n_periods, k), linear in log period."""
    if period < SA_PERIODS[0] - 1e-9 or period > SA_PERIODS[-1] + 1e-9:
        raise ValueError("SA(%s) outside the fitted period range %.3g-%.3g s"
                         % (period, SA_PERIODS[0], SA_PERIODS[-1]))
    lt = np.log(SA_PERIODS)
    x = np.log(period)
    if table.ndim == 1:
        return float(np.interp(x, lt, table))
    return np.array([np.interp(x, lt, table[:, j]) for j in range(table.shape[1])])


class Taherian2025Inland(GMPE):
    """inland regression GMM (see module docstring)."""
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.STABLE_CONTINENTAL
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, PGV, SA}
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.MEDIAN_HORIZONTAL
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}
    REQUIRES_SITES_PARAMETERS = set()
    DEFINED_FOR_REFERENCE_VELOCITY = 760.0
    REQUIRES_RUPTURE_PARAMETERS = {"mag", "hypo_depth", "rake"}
    REQUIRES_DISTANCES = {"rjb"}

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        mag = np.atleast_1d(ctx.mag).astype(float)
        rjb = np.maximum(np.atleast_1d(ctx.rjb).astype(float), 0.0)
        dep = np.clip(np.atleast_1d(ctx.hypo_depth).astype(float), *DEPTH_RANGE)
        r = ((np.atleast_1d(ctx.rake).astype(float) + 180.0) % 360.0) - 180.0
        fm = ((np.abs(r) <= 30.0) | (np.abs(r) >= 150.0)).astype(float)
        X = _design(mag, rjb, dep, fm)
        for m, imt in enumerate(imts):
            if imt.string == 'PGA':
                b, ta, ph = BETA_PGA, TAU_PGA, PHI_PGA
            elif imt.string == 'PGV':
                b, ta, ph = BETA_PGV, TAU_PGV, PHI_PGV
            else:
                b = _interp(imt.period, BETA_SA)
                ta = _interp(imt.period, TAU_SA)
                ph = _interp(imt.period, PHI_SA)
            mean[m] = X @ b
            tau[m] = ta
            phi[m] = ph
            sig[m] = np.sqrt(ta ** 2 + ph ** 2)
