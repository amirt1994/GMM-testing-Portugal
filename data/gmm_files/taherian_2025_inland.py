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

MH, H0, H1 = (5.80680484, 18.57733165, 0.00000052)
DEPTH_RANGE = (2.024, 29.729)
SA_PERIODS = np.array([0.02      , 0.04      , 0.05      , 0.06666667, 0.1       , 0.125     , 0.2       , 0.25      , 0.4       ,
 0.5       , 0.625     , 0.76923077, 1.        , 1.25      , 1.49253731, 2.        , 3.125     , 4.        ])
BETA_SA = np.array([[ 4.49037841e+00, -2.86241050e-01, -9.02252466e-01, -1.83815422e+00,  4.29994539e-01, -1.84960453e-03,
  -4.96491893e-02, -2.49566813e-02],
 [ 4.86796469e+00, -2.68785959e-01, -9.07701213e-01, -1.86391853e+00,  4.09350696e-01, -2.38609946e-03,
  -4.70100746e-02, -2.41738260e-02],
 [ 4.89805595e+00, -2.63024638e-01, -9.81726203e-01, -1.84449497e+00,  4.15819936e-01, -2.57866340e-03,
  -4.55744534e-02, -2.39166489e-02],
 [ 4.53003302e+00, -1.89167340e-01, -1.09505282e+00, -1.80333839e+00,  4.20586218e-01, -2.81337683e-03,
  -4.41752114e-02, -2.35636634e-02],
 [ 3.45215618e+00,  1.06814106e-02, -1.30367057e+00, -1.74594717e+00,  4.24401263e-01, -2.76794569e-03,
  -4.40052619e-02, -2.31127158e-02],
 [ 2.05212727e+00,  2.53338169e-01, -1.37762319e+00, -1.69313645e+00,  3.86612228e-01, -3.22339542e-03,
  -4.36323994e-02, -2.28378464e-02],
 [-1.80384914e+00,  8.98548790e-01, -1.54016315e+00, -1.54551081e+00,  2.73163459e-01, -3.48618073e-03,
  -4.22973582e-02, -2.16484737e-02],
 [-3.69631982e+00,  1.19995028e+00, -1.57515945e+00, -1.47898635e+00,  2.16724372e-01, -3.35777876e-03,
  -4.15430957e-02, -2.10045924e-02],
 [-7.15607438e+00,  1.71476957e+00, -1.72029876e+00, -1.37167557e+00,  1.33624708e-01, -2.79062514e-03,
  -4.11915337e-02, -2.01306511e-02],
 [-8.66329884e+00,  1.94232302e+00, -1.78445858e+00, -1.35380526e+00,  1.00026417e-01, -2.56193780e-03,
  -4.08579625e-02, -1.99461626e-02],
 [-9.91296903e+00,  2.11888951e+00, -1.78668315e+00, -1.34915652e+00,  7.94032062e-02, -2.17507942e-03,
  -4.11028310e-02, -1.99055175e-02],
 [-1.09578798e+01,  2.25507715e+00, -1.81983897e+00, -1.34430610e+00,  6.59843046e-02, -1.95090350e-03,
  -4.14752900e-02, -1.99512820e-02],
 [-1.20542163e+01,  2.36225649e+00, -1.82102502e+00, -1.31351014e+00,  6.32934811e-02, -1.95186141e-03,
  -4.20105037e-02, -2.01808981e-02],
 [-1.25066363e+01,  2.36079881e+00, -1.78224322e+00, -1.30073540e+00,  8.22268468e-02, -1.76646639e-03,
  -4.23354628e-02, -2.04562498e-02],
 [-1.28401907e+01,  2.36774322e+00, -1.75676757e+00, -1.31324980e+00,  9.17017262e-02, -1.55725700e-03,
  -4.28014394e-02, -2.06961812e-02],
 [-1.29993288e+01,  2.31802287e+00, -1.72044295e+00, -1.36362173e+00,  1.22027367e-01, -8.43642462e-04,
  -4.37061837e-02, -2.10654856e-02],
 [-1.33640900e+01,  2.19547385e+00, -1.56327516e+00, -1.32723633e+00,  1.74991974e-01, -9.78927289e-04,
  -4.41574751e-02, -2.15394825e-02],
 [-1.33235506e+01,  2.09661555e+00, -1.51068601e+00, -1.33081947e+00,  2.13834895e-01, -7.75488383e-04,
  -4.44734821e-02, -2.17689682e-02]])          # (n_periods, 8): c0..c7
BETA_PGA = np.array([ 4.10012228e+00, -2.49452366e-01, -9.95761660e-01, -1.81649534e+00,  4.46392256e-01, -1.50789517e-03,
 -4.98093065e-02, -2.51376575e-02])
BETA_PGV = np.array([ 1.44684502,  0.81902947, -1.24077199, -1.5613588 ,  0.28546608, -0.00158926, -0.04472446, -0.01994573])
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
