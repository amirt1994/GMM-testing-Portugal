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

MH, H0, H1 = (6.23424090, 13.47490034, 2.83245078)
DEPTH_RANGE = (2.024, 29.729)
SA_PERIODS = np.array([0.02      , 0.04      , 0.05      , 0.06666667, 0.1       , 0.125     , 0.2       , 0.25      , 0.4       ,
 0.5       , 0.625     , 0.76923077, 1.        , 1.25      , 1.49253731, 2.        , 3.125     , 4.        ])
BETA_SA = np.array([[-1.09721332e+00,  4.83199284e-01, -1.21919928e+00, -1.35816518e+00,  2.26983959e-01, -5.70381153e-03,
  -5.11478560e-02, -2.49485940e-02],
 [-7.31655605e-01,  5.05195067e-01, -1.26279761e+00, -1.38674728e+00,  2.06747861e-01, -6.17884468e-03,
  -4.84956681e-02, -2.41656268e-02],
 [-6.57002500e-01,  5.02540034e-01, -1.35923889e+00, -1.36950295e+00,  2.14438776e-01, -6.34518714e-03,
  -4.69847307e-02, -2.39079105e-02],
 [-9.36073594e-01,  5.59457356e-01, -1.50521832e+00, -1.33116520e+00,  2.21976669e-01, -6.55726012e-03,
  -4.55028536e-02, -2.35540459e-02],
 [-1.86110279e+00,  7.26689095e-01, -1.76553888e+00, -1.27381555e+00,  2.31000370e-01, -6.51765590e-03,
  -4.52612924e-02, -2.31013815e-02],
 [-3.09278945e+00,  9.43567010e-01, -1.82971937e+00, -1.23380647e+00,  1.96654372e-01, -6.90680463e-03,
  -4.49740332e-02, -2.28257148e-02],
 [-6.50434157e+00,  1.52646376e+00, -1.95635469e+00, -1.12779574e+00,  9.06683447e-02, -6.94641976e-03,
  -4.38809250e-02, -2.16344444e-02],
 [-7.98376158e+00,  1.76842412e+00, -1.95131508e+00, -1.09898756e+00,  4.62157447e-02, -6.50233744e-03,
  -4.31701302e-02, -2.09901287e-02],
 [-1.05776736e+01,  2.15186628e+00, -2.02236876e+00, -1.06457064e+00, -1.05562939e-02, -5.27785569e-03,
  -4.28142831e-02, -2.01148918e-02],
 [-1.18044264e+01,  2.33585986e+00, -2.05948369e+00, -1.06989020e+00, -3.58979239e-02, -4.83451214e-03,
  -4.25197943e-02, -1.99298122e-02],
 [-1.28692963e+01,  2.48451976e+00, -2.04194028e+00, -1.08084427e+00, -5.00890891e-02, -4.29339063e-03,
  -4.27886982e-02, -1.98891248e-02],
 [-1.37629351e+01,  2.59609156e+00, -2.06466363e+00, -1.08673477e+00, -5.82330758e-02, -3.96096425e-03,
  -4.31807378e-02, -1.99346283e-02],
 [-1.47019922e+01,  2.68086756e+00, -2.06403672e+00, -1.06988360e+00, -5.48187451e-02, -3.83342539e-03,
  -4.36740853e-02, -2.01643817e-02],
 [-1.51029414e+01,  2.67256061e+00, -2.03249975e+00, -1.06130645e+00, -3.20022294e-02, -3.58829603e-03,
  -4.39080354e-02, -2.04402788e-02],
 [-1.54300671e+01,  2.67811312e+00, -2.01117334e+00, -1.07343509e+00, -2.06297380e-02, -3.35718891e-03,
  -4.43475964e-02, -2.06805598e-02],
 [-1.56531464e+01,  2.63382065e+00, -1.98926985e+00, -1.11290614e+00,  1.14892688e-02, -2.67281498e-03,
  -4.51988744e-02, -2.10504510e-02],
 [-1.60532524e+01,  2.52296862e+00, -1.86454819e+00, -1.07700033e+00,  6.87615421e-02, -2.76947933e-03,
  -4.54244343e-02, -2.15263360e-02],
 [-1.60723845e+01,  2.43304208e+00, -1.83316782e+00, -1.07417067e+00,  1.08820133e-01, -2.58447375e-03,
  -4.56081024e-02, -2.17566067e-02]])          # (n_periods, 8): c0..c7
BETA_PGA = np.array([-1.43638768,  0.50702178, -1.32565448, -1.3345561 ,  0.24503853, -0.00537864, -0.05124922, -0.0251289 ])
BETA_PGV = np.array([-1.84930705,  1.22337329, -1.44093114, -1.24312174,  0.16615952, -0.00385465, -0.04607036, -0.0199355 ])
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
