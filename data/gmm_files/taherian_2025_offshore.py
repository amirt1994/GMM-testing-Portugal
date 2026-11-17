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

MH, H0, H1 = (6.41614362, 21.15815691, 5.34390770)
DEPTH_RANGE = (2.051, 29.951)
SA_PERIODS = np.array([0.02      , 0.04      , 0.05      , 0.06666667, 0.1       , 0.125     , 0.2       , 0.25      , 0.4       ,
 0.5       , 0.625     , 0.76923077, 1.        , 1.25      , 1.49253731, 2.        , 3.125     , 4.        ])
BETA_SA = np.array([[ 3.15823743e+00,  2.48633169e-01, -9.13135801e-01, -1.87196235e+00,  2.67703067e-01, -1.80623211e-03,
  -4.41872389e-02, -4.72748948e-02],
 [ 3.22227299e+00,  3.07866314e-01, -8.55850840e-01, -1.86462816e+00,  2.34029649e-01, -2.08247386e-03,
  -4.15685068e-02, -4.56724974e-02],
 [ 3.28857256e+00,  2.92429306e-01, -9.08259413e-01, -1.83772076e+00,  2.42116108e-01, -2.22403960e-03,
  -4.00921211e-02, -4.46502251e-02],
 [ 3.06793625e+00,  3.17768676e-01, -9.65238379e-01, -1.79404943e+00,  2.52647666e-01, -2.44662593e-03,
  -3.84762107e-02, -4.33852544e-02],
 [ 2.38539834e+00,  4.13241469e-01, -1.05267670e+00, -1.74677870e+00,  2.71013516e-01, -2.51978821e-03,
  -3.76184704e-02, -4.19169913e-02],
 [ 1.03907979e+00,  6.47701575e-01, -1.09448278e+00, -1.71925494e+00,  2.35655965e-01, -2.82916467e-03,
  -3.72035294e-02, -4.12365161e-02],
 [-2.53781220e+00,  1.27383724e+00, -1.20687705e+00, -1.67617564e+00,  1.34990442e-01, -2.51768739e-03,
  -3.73514371e-02, -3.97006203e-02],
 [-4.18482051e+00,  1.55330893e+00, -1.24765745e+00, -1.66615424e+00,  9.29072382e-02, -2.05816946e-03,
  -3.76024812e-02, -3.90138206e-02],
 [-7.08897729e+00,  1.99343909e+00, -1.42583184e+00, -1.63947829e+00,  4.13200023e-02, -1.24212886e-03,
  -3.83969966e-02, -3.81656786e-02],
 [-8.42057834e+00,  2.19174784e+00, -1.49425048e+00, -1.63957102e+00,  1.88041728e-02, -1.00674714e-03,
  -3.84151665e-02, -3.80748679e-02],
 [-9.50847135e+00,  2.34025826e+00, -1.49248455e+00, -1.64616252e+00,  9.11817502e-03, -6.90228679e-04,
  -3.93028030e-02, -3.81486106e-02],
 [-1.03959089e+01,  2.44492797e+00, -1.52490338e+00, -1.64628648e+00,  5.03933244e-03, -5.43023952e-04,
  -3.96238726e-02, -3.82717030e-02],
 [-1.12810980e+01,  2.51069304e+00, -1.52495376e+00, -1.62277031e+00,  1.37910117e-02, -5.68990362e-04,
  -3.96610585e-02, -3.85637948e-02],
 [-1.15548587e+01,  2.46779757e+00, -1.48804835e+00, -1.60389258e+00,  4.38239683e-02, -5.12156973e-04,
  -3.94645606e-02, -3.88592666e-02],
 [-1.18128849e+01,  2.45306237e+00, -1.46071001e+00, -1.60714323e+00,  5.93049589e-02, -4.23094761e-04,
  -3.96609673e-02, -3.91049935e-02],
 [-1.18341356e+01,  2.35716929e+00, -1.40916822e+00, -1.63106312e+00,  1.00030479e-01, -4.93082294e-05,
  -4.01470059e-02, -3.94496988e-02],
 [-1.19875014e+01,  2.17761110e+00, -1.20742203e+00, -1.58072683e+00,  1.64704585e-01, -2.85241103e-04,
  -3.92146518e-02, -3.99302341e-02],
 [-1.17964841e+01,  2.03358100e+00, -1.11053364e+00, -1.56963734e+00,  2.10208778e-01, -2.32687587e-04,
  -3.91749589e-02, -4.01717099e-02]])          # (n_periods, 8): c0..c7
BETA_PGA = np.array([ 3.07525103e+00,  2.11976166e-01, -9.83551306e-01, -1.85971842e+00,  2.98986243e-01, -1.66055105e-03,
 -4.40040351e-02, -4.56293066e-02])
BETA_PGV = np.array([ 1.77398249e+00,  1.12602611e+00, -1.28141984e+00, -1.85281449e+00,  1.86435838e-01, -7.27798548e-04,
 -4.53821583e-02, -3.95990225e-02])
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
