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

MH, H0, H1 = (6.65802206, 21.30705440, 6.97230753)
DEPTH_RANGE = (2.051, 29.951)
SA_PERIODS = np.array([0.02      , 0.04      , 0.05      , 0.06666667, 0.1       , 0.125     , 0.2       , 0.25      , 0.4       ,
 0.5       , 0.625     , 0.76923077, 1.        , 1.25      , 1.49253731, 2.        , 3.125     , 4.        ])
BETA_SA = np.array([[ 3.10256562e+00,  3.09147296e-01, -9.58346974e-01, -1.94149331e+00,  2.49546871e-01, -1.44857384e-03,
  -4.42737115e-02, -4.72742903e-02],
 [ 3.06858631e+00,  3.88850973e-01, -9.84874719e-01, -1.93277085e+00,  2.12725087e-01, -1.74724324e-03,
  -4.16390893e-02, -4.56711804e-02],
 [ 3.13818296e+00,  3.73439953e-01, -1.07041945e+00, -1.90716744e+00,  2.20656493e-01, -1.87693090e-03,
  -4.01281304e-02, -4.46485877e-02],
 [ 2.92368980e+00,  3.96345473e-01, -1.17464334e+00, -1.86188437e+00,  2.31980894e-01, -2.10654171e-03,
  -3.84762410e-02, -4.33831412e-02],
 [ 2.25433130e+00,  4.84585702e-01, -1.33610693e+00, -1.80781951e+00,  2.52809902e-01, -2.22244254e-03,
  -3.75839014e-02, -4.19140980e-02],
 [ 9.03497465e-01,  7.20321481e-01, -1.38041687e+00, -1.78096715e+00,  2.17048450e-01, -2.53175910e-03,
  -3.72120830e-02, -4.12335130e-02],
 [-2.69902551e+00,  1.35317279e+00, -1.49285831e+00, -1.74043838e+00,  1.14306580e-01, -2.21861461e-03,
  -3.74835324e-02, -3.96973964e-02],
 [-4.35904082e+00,  1.63700338e+00, -1.52859438e+00, -1.73310587e+00,  7.08212367e-02, -1.74805104e-03,
  -3.77833655e-02, -3.90105751e-02],
 [-7.24773356e+00,  2.07850240e+00, -1.70237195e+00, -1.71481101e+00,  1.69092032e-02, -8.81106045e-04,
  -3.86232733e-02, -3.81623290e-02],
 [-8.57947777e+00,  2.27695141e+00, -1.77050165e+00, -1.71561347e+00, -6.29442401e-03, -6.45098163e-04,
  -3.86677883e-02, -3.80714517e-02],
 [-9.67697604e+00,  2.42512391e+00, -1.76990811e+00, -1.71864898e+00, -1.57306439e-02, -3.56065170e-04,
  -3.95743574e-02, -3.81451595e-02],
 [-1.05623554e+01,  2.52674406e+00, -1.80665567e+00, -1.71514332e+00, -1.92634714e-02, -2.34895513e-04,
  -3.99050353e-02, -3.82681696e-02],
 [-1.14411903e+01,  2.58735226e+00, -1.81454210e+00, -1.68604980e+00, -9.09735358e-03, -2.95118746e-04,
  -3.99321129e-02, -3.85601681e-02],
 [-1.17038172e+01,  2.53739966e+00, -1.78751871e+00, -1.65996016e+00,  2.31663836e-02, -2.80421581e-04,
  -3.97027755e-02, -3.88555730e-02],
 [-1.19619282e+01,  2.51880471e+00, -1.76696187e+00, -1.65719473e+00,  4.00744245e-02, -2.30435543e-04,
  -3.98894139e-02, -3.91012444e-02],
 [-1.19787329e+01,  2.41424890e+00, -1.73181309e+00, -1.66898486e+00,  8.39676331e-02,  6.43415352e-05,
  -4.03511099e-02, -3.94458176e-02],
 [-1.21295149e+01,  2.22417294e+00, -1.55292697e+00, -1.60212395e+00,  1.53982091e-01, -2.68721840e-04,
  -3.93568037e-02, -3.99262452e-02],
 [-1.19311443e+01,  2.07290592e+00, -1.47296275e+00, -1.58159045e+00,  2.02795513e-01, -2.71805383e-04,
  -3.92776849e-02, -4.01676148e-02]])          # (n_periods, 8): c0..c7
BETA_PGA = np.array([ 3.05854675e+00,  2.62955168e-01, -1.04571441e+00, -1.92760941e+00,  2.82772115e-01, -1.30697585e-03,
 -4.40570468e-02, -4.56285032e-02])
BETA_PGV = np.array([ 1.76613016e+00,  1.16840734e+00, -1.36015536e+00, -1.91230823e+00,  1.70456592e-01, -4.58907160e-04,
 -4.56097540e-02, -3.95976814e-02])
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
