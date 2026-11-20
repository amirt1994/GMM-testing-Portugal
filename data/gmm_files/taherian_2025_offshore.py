# -*- coding: utf-8 -*-
"""
Taherian et al. (2025) ground-motion model for offshore SW Iberia (Portugal), rock sites
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

MH, H0, H1 = (6.75799896, 17.83325176, 7.99999960)
DEPTH_RANGE = (2.051, 29.951)
SA_PERIODS = np.array([0.02      , 0.04      , 0.05      , 0.06666667, 0.1       , 0.125     , 0.2       , 0.25      , 0.4       ,
 0.5       , 0.625     , 0.76923077, 1.        , 1.25      , 1.49253731, 2.        , 3.125     , 4.        ])
BETA_SA = np.array([[-4.73176960e-01,  7.82786071e-01, -1.15301581e+00, -1.61490620e+00,  1.29060270e-01, -3.83658090e-03,
  -4.53941198e-02, -4.72727456e-02],
 [-5.54974767e-01,  8.69971776e-01, -1.24127147e+00, -1.60124458e+00,  9.05367562e-02, -4.19466605e-03,
  -4.27631591e-02, -4.56689939e-02],
 [-5.03814773e-01,  8.59420981e-01, -1.35220436e+00, -1.57668958e+00,  9.73020682e-02, -4.32655895e-03,
  -4.12348444e-02, -4.46461222e-02],
 [-7.34406170e-01,  8.87189134e-01, -1.49272881e+00, -1.53254573e+00,  1.07613434e-01, -4.56469630e-03,
  -3.95627291e-02, -4.33802696e-02],
 [-1.43369219e+00,  9.81609812e-01, -1.71186016e+00, -1.47710019e+00,  1.27382523e-01, -4.71150958e-03,
  -3.86466510e-02, -4.19105844e-02],
 [-2.74366579e+00,  1.21142502e+00, -1.75723828e+00, -1.45262439e+00,  9.24744103e-02, -5.02418299e-03,
  -3.83025145e-02, -4.12299385e-02],
 [-6.23974670e+00,  1.82841149e+00, -1.86858423e+00, -1.41835458e+00, -8.06786423e-03, -4.71088497e-03,
  -3.86517319e-02, -3.96937247e-02],
 [-7.76622227e+00,  2.09179714e+00, -1.89530968e+00, -1.42062899e+00, -4.69977120e-02, -4.15890121e-03,
  -3.89430633e-02, -3.90069860e-02],
 [-1.02049824e+01,  2.46558837e+00, -2.04129536e+00, -1.43844365e+00, -8.46533748e-02, -2.94111111e-03,
  -3.96549359e-02, -3.81590604e-02],
 [-1.13826058e+01,  2.63984333e+00, -2.10064908e+00, -1.45022962e+00, -1.02109722e-01, -2.59540676e-03,
  -3.96636789e-02, -3.80682828e-02],
 [-1.23688835e+01,  2.76964215e+00, -2.09395324e+00, -1.45979192e+00, -1.06811818e-01, -2.23297661e-03,
  -4.05410158e-02, -3.81420743e-02],
 [-1.31666857e+01,  2.85704401e+00, -2.12786071e+00, -1.46179082e+00, -1.06767287e-01, -2.05359714e-03,
  -4.08477180e-02, -3.82651217e-02],
 [-1.39542827e+01,  2.90418625e+00, -2.13491079e+00, -1.43976274e+00, -9.28894097e-02, -2.04991070e-03,
  -4.08434872e-02, -3.85571380e-02],
 [-1.41798355e+01,  2.84916323e+00, -2.11003955e+00, -1.41703145e+00, -5.86313775e-02, -1.99892610e-03,
  -4.05782297e-02, -3.88525396e-02],
 [-1.44188649e+01,  2.82670124e+00, -2.09093172e+00, -1.41437936e+00, -4.02969489e-02, -1.93694203e-03,
  -4.07467503e-02, -3.90982102e-02],
 [-1.44441200e+01,  2.72073034e+00, -2.06144770e+00, -1.42225923e+00,  4.96612487e-03, -1.64615474e-03,
  -4.11744994e-02, -3.94427541e-02],
 [-1.45817300e+01,  2.53051544e+00, -1.89111245e+00, -1.35750837e+00,  7.70383368e-02, -1.95860969e-03,
  -4.01296910e-02, -3.99231265e-02],
 [-1.44056090e+01,  2.38287449e+00, -1.81828883e+00, -1.33533912e+00,  1.26159539e-01, -1.96471148e-03,
  -4.00191623e-02, -4.01644421e-02]])          # (n_periods, 8): c0..c7
BETA_PGA = np.array([-0.53494377,  0.73994327, -1.2488091 , -1.60092658,  0.16168636, -0.00369256, -0.04515666, -0.04562686])
BETA_PGV = np.array([-1.01229387,  1.51328424, -1.52495583, -1.63681637,  0.08138426, -0.00231271, -0.04651104, -0.03959644])
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


class Taherian2025Offshore(GMPE):
    """offshore regression GMM (see module docstring)."""
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
