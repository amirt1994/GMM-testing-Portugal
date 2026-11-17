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

MH, H0, H1 = (5.69133311, 18.96040705, 3.51626732)
DEPTH_RANGE = (2.051, 29.951)
SA_PERIODS = np.array([0.02      , 0.04      , 0.05      , 0.06666667, 0.1       , 0.125     , 0.2       , 0.25      , 0.4       ,
 0.5       , 0.625     , 0.76923077, 1.        , 1.25      , 1.49253731, 2.        , 3.125     , 4.        ])
BETA_SA = np.array([[ 9.60767885e-02,  6.54919020e-01, -8.99589743e-01, -1.54748586e+00,  1.89254271e-01, -4.48669104e-03,
  -4.32295916e-02, -4.72763204e-02],
 [ 2.89907456e-01,  6.79350964e-01, -9.14935787e-01, -1.53349969e+00,  1.59650882e-01, -4.81788847e-03,
  -4.05833024e-02, -4.56725715e-02],
 [ 3.24315555e-01,  6.70922318e-01, -9.75085015e-01, -1.50587029e+00,  1.66737794e-01, -4.97996105e-03,
  -3.91328011e-02, -4.46501785e-02],
 [ 8.74294990e-02,  7.01936491e-01, -1.02821951e+00, -1.46542604e+00,  1.74923471e-01, -5.20121424e-03,
  -3.75315763e-02, -4.33851763e-02],
 [-6.05867976e-01,  8.04533644e-01, -1.08827676e+00, -1.42711007e+00,  1.88214070e-01, -5.23566186e-03,
  -3.66599981e-02, -4.19171339e-02],
 [-1.94415573e+00,  1.03849613e+00, -1.12703859e+00, -1.39897103e+00,  1.53704716e-01, -5.55791701e-03,
  -3.61724480e-02, -4.12368747e-02],
 [-5.49412998e+00,  1.66161806e+00, -1.24287533e+00, -1.35301327e+00,  5.68832146e-02, -5.27631562e-03,
  -3.61209794e-02, -3.97014747e-02],
 [-7.06214476e+00,  1.92933392e+00, -1.29628830e+00, -1.34614611e+00,  2.05389119e-02, -4.75836148e-03,
  -3.63373217e-02, -3.90147996e-02],
 [-9.74918130e+00,  2.34700086e+00, -1.48780796e+00, -1.34092908e+00, -1.49691272e-02, -3.64839596e-03,
  -3.72067608e-02, -3.81673244e-02],
 [-1.09956288e+01,  2.53645223e+00, -1.55521379e+00, -1.35040316e+00, -3.21933591e-02, -3.29526772e-03,
  -3.72269072e-02, -3.80768420e-02],
 [-1.19689190e+01,  2.66828155e+00, -1.54634861e+00, -1.36811595e+00, -3.83858939e-02, -2.85693340e-03,
  -3.81175059e-02, -3.81507448e-02],
 [-1.27878391e+01,  2.76646925e+00, -1.56434754e+00, -1.37925371e+00, -4.07895970e-02, -2.60167878e-03,
  -3.84375825e-02, -3.82741404e-02],
 [-1.35738025e+01,  2.82075794e+00, -1.53754770e+00, -1.37085188e+00, -3.16245765e-02, -2.50127022e-03,
  -3.84981945e-02, -3.85665412e-02],
 [-1.37696424e+01,  2.76885720e+00, -1.47117566e+00, -1.36649065e+00, -3.41335147e-03, -2.33286544e-03,
  -3.83494769e-02, -3.88621798e-02],
 [-1.39630002e+01,  2.74504447e+00, -1.42153715e+00, -1.37962021e+00,  1.09759541e-02, -2.16195745e-03,
  -3.85559290e-02, -3.91080573e-02],
 [-1.38950398e+01,  2.63699258e+00, -1.32407140e+00, -1.42043588e+00,  4.80256876e-02, -1.64841152e-03,
  -3.90601621e-02, -3.94530505e-02],
 [-1.38332344e+01,  2.42026911e+00, -1.05714810e+00, -1.39569873e+00,  1.07148244e-01, -1.71050096e-03,
  -3.82077786e-02, -3.99335810e-02],
 [-1.35546470e+01,  2.26123564e+00, -9.20380250e-01, -1.39749758e+00,  1.48507485e-01, -1.56948922e-03,
  -3.82154375e-02, -4.01751094e-02]])          # (n_periods, 8): c0..c7
BETA_PGA = np.array([-0.05693291,  0.63649709, -0.94181685, -1.53972407,  0.21664131, -0.00431862, -0.04307888, -0.04563114])
BETA_PGV = np.array([-0.84579578,  1.49394332, -1.20423115, -1.59348823,  0.12826617, -0.00267032, -0.04439731, -0.03960269])
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
