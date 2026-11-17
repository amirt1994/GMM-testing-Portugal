# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2025 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:`Xana2025_inland`
"""
import numpy as np
from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA


class Morgadoetal2026_offshore(GMPE):
    """
    Xana et al. (2025) GMPE for offshore Portugal
    """

    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, SA}
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GEOMETRIC_MEAN
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT
    }
    REQUIRES_RUPTURE_PARAMETERS = {'mag'}
    REQUIRES_DISTANCES = {'rjb'}
    REQUIRES_SITES_PARAMETERS = set()

    def __init__(self):
        super().__init__()
        self.Mref = 4.0
        self.h = 3.0
        self.FD = 1.5

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]



            # Magnitude term (piecewise linear)
            mag = np.atleast_1d(ctx.mag)
            fmag = np.zeros_like(mag, dtype=float)

            mask1 = mag < 5.5
            fmag[mask1] = C['a0'] + C['a1'] * mag[mask1]

            mask2 = (mag >= 5.5) & (mag < 6.5)
            fmag[mask2] = C['a0'] + C['a1'] * mag[mask2] + C['a2'] * (mag[mask2] - 5.5)

            mask3 = (mag >= 6.5) & (mag < 7.5)
            fmag[mask3] = (C['a0'] + C['a1'] * mag[mask3] + C['a2'] * (mag[mask3] - 5.5) +
                                       C['a3'] * (mag[mask3] - 6.5))

            mask4 = (mag >= 7.5) & (mag < 8.5)
            fmag[mask4] = (C['a0'] + C['a1'] * mag[mask4] + C['a2'] * (mag[mask4] - 5.5) +
                           C['a3'] * (mag[mask4] - 6.5) + C['a33'] * (mag[mask4] - 7.5))

            mask5 = mag >= 8.5
            fmag[mask5] = (C['a0'] + C['a1'] * mag[mask5] + C['a2'] * (mag[mask5] - 5.5) +
                           C['a3'] * (mag[mask5] - 6.5) + C['a33'] * (mag[mask5] - 7.5) +
                           C['a333'] * (mag[mask5] - 8.0))

            fmag = fmag if fmag.size > 1 else float(fmag[0])


            # Distance term
            R1 = np.sqrt(ctx.rjb**2 + self.h**2)
            R2 = np.sqrt(ctx.rjb**2 + self.FD**2)

            mag_dist = np.minimum(mag, 7.3)

            #self.Mref = np.where(R1 > 400.0, 4.0, self.Mref)
            fdis = (C['a4'] + C['a5'] * (mag_dist - self.Mref)) * np.log(R1) + C['a6'] * R2

            mean[m] = fmag + fdis
            tau[m] = C['std_inter']
            phi[m] = C['std_intra']
            sig[m] = np.sqrt(tau[m]**2 + phi[m]**2)

    COEFFS = CoeffsTable(sa_damping=5, table="""\
IMT      a0         a1        a2         a3         a33      a333       a4         a5        a6          std_intra  std_inter
PGA -9.79993695 1.68341132 -0.63513324 -0.90650132 0 -0.000001 -0.73875222 0.05827204 -0.005463768 0.871 0.204 0.895
0.015 -9.41377423 1.61361532 -0.56329003 -0.84796744 -0.03572309 -0.000001 -0.76477386 0.06862852 -0.00526087 0.904 0.198 0.926
0.02 -9.86484918 1.6937993 -0.64221122 -0.91402019 0 -0.000001 -0.73242477 0.05638174 -0.005463768 0.873 0.206 0.897
0.04 -8.93565262 1.59022371 -0.62011896 -0.79262874 -0.01654236 -0.000001 -0.7654891 0.07138996 -0.00526087 0.889 0.167 0.905
0.05 -10.10675879 1.81010398 -0.84670749 -0.76646914 0 -0.000001 -0.66511127 0.03967814 -0.005463768 0.769 0.144 0.783
0.067 -10.87591143 1.95747231 -0.98250206 -0.80160758 0 -0.000001 -0.58431647 0.02053328 -0.005666667 0.732 0.171 0.752
0.1 -11.61455694 2.08493499 -1.05763995 -0.86226439 0 -0.000001 -0.50768465 0.00464047 -0.005463768 0.738 0.216 0.769
0.125 -11.88481486 2.12275204 -1.05802081 -0.89530735 0 -0.000001 -0.48104012 -0.00020724 -0.005869565 0.752 0.236 0.788
0.15 -12.14335297 2.15711054 -1.05680367 -0.92573028 0 -0.000001 -0.46599514 -0.00316847 -0.006072464 0.766 0.251 0.806
0.2 -12.49685862 2.19448102 -1.02836816 -0.98255381 0 -0.000001 -0.44921055 -0.006644 -0.006681159 0.787 0.275 0.834
0.225 -12.68001973 2.21462639 -1.02269905 -1.00308777 0 -0.000001 -0.44348139 -0.00771816 -0.006884058 0.795 0.282 0.844
0.25 -12.83853906 2.22979479 -1.00916201 -1.02377403 0 -0.000001 -0.43781133 -0.00919164 -0.007086957 0.802 0.289 0.852
0.3 -13.19650162 2.2682453 -0.99706873 -1.06216901 0 -0.000001 -0.42827521 -0.01105073 -0.007492754 0.813 0.292 0.864
0.35 -13.52954384 2.30270535 -0.98080518 -1.09821614 0 -0.000001 -0.41972039 -0.01299805 -0.007492754 0.821 0.308 0.877
0.4 -13.85780389 2.3370326 -0.96500766 -1.13455959 0 -0.000001 -0.41144069 -0.01487952 -0.007898551 0.828 0.314 0.885
0.5 -14.46668254 2.40074523 -0.94046184 -1.19281425 0 -0.000001 -0.39781393 -0.01768177 -0.008304348 0.837 0.323 0.897
0.6 -15.00403876 2.4562507 -0.91413776 -1.24684207 0 -0.000001 -0.38999752 -0.01931654 -0.008710145 0.845 0.33 0.907
0.7 -15.48658416 2.5054366 -0.89052411 -1.29243869 0 -0.000001 -0.38411671 -0.0203268 -0.008913043 0.851 0.333 0.914
0.75 -15.70240269 2.52703403 -0.87957558 -1.31215348 0 -0.000001 -0.38265942 -0.02042598 -0.009115942 0.853 0.332 0.915
0.8 -15.89888441 2.54618295 -0.87004552 -1.32962517 0 -0.000001 -0.38219627 -0.02006111 -0.008913043 0.855 0.335 0.918
0.85 -16.08153037 2.56388422 -0.86005557 -1.34613892 0 -0.000001 -0.38306868 -0.01956368 -0.009115942 0.857 0.336 0.921
0.9 -16.25463313 2.58033155 -0.85002088 -1.36175034 0 -0.000001 -0.38377126 -0.01903493 -0.009318841 0.86 0.337 0.923
0.95 -16.41914484 2.59608468 -0.84167402 -1.37537536 0 -0.000001 -0.38539266 -0.01823248 -0.009115942 0.862 0.335 0.925
1 -16.57674413 2.61139304 -0.83611244 -1.38580625 0 -0.000001 -0.3876337 -0.01724754 -0.009318841 0.864 0.337 0.927
1.05 -16.72453262 2.62575049 -0.83218843 -1.3933337 0 -0.000001 -0.39066395 -0.01614594 -0.009115942 0.866 0.337 0.929
1.25 -17.14369308 2.65629265 -0.79323504 -1.42804273 -0.00000098 -0.000001 -0.4046381 -0.01095929 -0.009318841 0.873 0.336 0.935
1.5 -17.40691739 2.65880038 -0.72312771 -1.47573181 -0.0000161 -0.000001 -0.42995769 -0.00249789 -0.009115942 0.879 0.334 0.94
2 -17.51209078 2.61374722 -0.58282035 -1.53565754 -0.00005864 -0.000001 -0.49103217 0.0156148 -0.009115942 0.888 0.332 0.948





    """)
