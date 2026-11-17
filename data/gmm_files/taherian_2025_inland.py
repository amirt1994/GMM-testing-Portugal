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

MH, H0, H1 = (5.74667941, 18.55102331, 0.00000002)
DEPTH_RANGE = (2.024, 29.729)
SA_PERIODS = np.array([0.02      , 0.04      , 0.05      , 0.06666667, 0.1       , 0.125     , 0.2       , 0.25      , 0.4       ,
 0.5       , 0.625     , 0.76923077, 1.        , 1.25      , 1.49253731, 2.        , 3.125     , 4.        ])
BETA_SA = np.array([[ 4.42172494e+00, -2.69843562e-01, -9.01087142e-01, -1.83757353e+00,  4.29303097e-01, -1.85027605e-03,
  -4.96387064e-02, -2.49577103e-02],
 [ 4.88690559e+00, -2.75778390e-01, -8.41776238e-01, -1.85925917e+00,  4.11492794e-01, -2.40754920e-03,
  -4.71157542e-02, -2.41747717e-02],
 [ 4.93821776e+00, -2.75637208e-01, -8.96653711e-01, -1.83876331e+00,  4.18735812e-01, -2.60555535e-03,
  -4.57127417e-02, -2.39176472e-02],
 [ 4.59547766e+00, -2.08433111e-01, -9.86114644e-01, -1.79633834e+00,  4.24430105e-01, -2.84675103e-03,
  -4.43526394e-02, -2.35647429e-02],
 [ 3.54813567e+00, -1.65654264e-02, -1.16263031e+00, -1.73735216e+00,  4.29426353e-01, -2.80948387e-03,
  -4.42328350e-02, -2.31139576e-02],
 [ 2.14374630e+00,  2.27338617e-01, -1.23669761e+00, -1.68466065e+00,  3.91620573e-01, -3.26434064e-03,
  -4.38589130e-02, -2.28391781e-02],
 [-1.72282620e+00,  8.75595278e-01, -1.40046389e+00, -1.53734005e+00,  2.78170462e-01, -3.52547092e-03,
  -4.25223807e-02, -2.16500266e-02],
 [-3.61902541e+00,  1.17804557e+00, -1.43695449e+00, -1.47089721e+00,  2.21774890e-01, -3.39654678e-03,
  -4.17690187e-02, -2.10062222e-02],
 [-7.09148277e+00,  1.69633061e+00, -1.58502793e+00, -1.36381318e+00,  1.38709103e-01, -2.82793844e-03,
  -4.14184964e-02, -2.01325206e-02],
 [-8.60379926e+00,  1.92526399e+00, -1.65007390e+00, -1.34602258e+00,  1.05115177e-01, -2.59873186e-03,
  -4.10849098e-02, -1.99481286e-02],
 [-9.85410207e+00,  2.10200272e+00, -1.65279109e+00, -1.34137606e+00,  8.45132541e-02, -2.21182783e-03,
  -4.13301970e-02, -1.99074981e-02],
 [-1.09007546e+01,  2.23868944e+00, -1.68583230e+00, -1.33657502e+00,  7.10825192e-02, -1.98739209e-03,
  -4.17020179e-02, -1.99532991e-02],
 [-1.19954803e+01,  2.34552153e+00, -1.68612144e+00, -1.30582338e+00,  6.83763164e-02, -1.98826425e-03,
  -4.22362769e-02, -2.01828954e-02],
 [-1.24437308e+01,  2.34299940e+00, -1.64624995e+00, -1.29303614e+00,  8.72987636e-02, -1.80308703e-03,
  -4.25606918e-02, -2.04581756e-02],
 [-1.27740989e+01,  2.34912048e+00, -1.61975346e+00, -1.30552793e+00,  9.67605969e-02, -1.59410874e-03,
  -4.30258984e-02, -2.06980532e-02],
 [-1.29278464e+01,  2.29798862e+00, -1.58123348e+00, -1.35585228e+00,  1.27041107e-01, -8.80925281e-04,
  -4.39287339e-02, -2.10672636e-02],
 [-1.32767468e+01,  2.17134846e+00, -1.42032320e+00, -1.31937109e+00,  1.79982234e-01, -1.01726277e-03,
  -4.43782633e-02, -2.15409874e-02],
 [-1.32299137e+01,  2.07086431e+00, -1.36567305e+00, -1.32292294e+00,  2.18788182e-01, -8.14220281e-04,
  -4.46928676e-02, -2.17703624e-02]])          # (n_periods, 8): c0..c7
BETA_PGA = np.array([ 4.02621887e+00, -2.31597546e-01, -9.94013482e-01, -1.81605280e+00,  4.45631911e-01, -1.50784493e-03,
 -4.97970958e-02, -2.51387816e-02])
BETA_PGV = np.array([ 1.36133982,  0.84048623, -1.23824663, -1.56147759,  0.28464012, -0.00158641, -0.04470698, -0.01994713])
TAU_SA = np.array([0.586797, 0.594194, 0.581547, 0.571526, 0.552996, 0.547286, 0.540025, 0.539698, 0.539749,
    0.536437, 0.5311, 0.532965, 0.530049, 0.532506, 0.529689, 0.529757, 0.52447, 0.519291])
PHI_SA = np.array([0.385028, 0.398904, 0.360417, 0.366348, 0.379416, 0.370299, 0.36004, 0.347921, 0.362387,
    0.358152, 0.350158, 0.364456, 0.361098, 0.351256, 0.371782, 0.364783, 0.401858, 0.395413])
TAU_PGA, PHI_PGA = 0.560535, 0.369216
TAU_PGV, PHI_PGV = 0.536437, 0.360848


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
