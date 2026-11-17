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


class Morgadoetal2026_inland(GMPE):
    """
    Xana et al. (2025) GMPE for Inland Portugal
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
        self.Mref = 5.0
        self.h = 10.0
        self.FD = 10.0
    
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
            
            mask3 = mag >= 6.5
            fmag[mask3] = (C['a0'] + C['a1'] * mag[mask3] + C['a2'] * (mag[mask3] - 5.5) + 
                           C['a3'] * (mag[mask3] - 6.5))
            
            fmag = fmag if fmag.size > 1 else float(fmag[0])
            
            # Distance term
            R1 = np.sqrt(ctx.rjb**2 + self.h**2)
            R2 = np.sqrt(ctx.rjb**2 + self.FD**2)
            fdis = (C['a4'] + C['a5'] * (ctx.mag - self.Mref)) * np.log(R1) + C['a6'] * R2
            
            mean[m] = fmag + fdis
            tau[m] = C['std_inter']
            phi[m] = C['std_intra']
            sig[m] = np.sqrt(tau[m]**2 + phi[m]**2)
    
    COEFFS = CoeffsTable(sa_damping=5, table="""\
     IMT      a0         a1        a2         a3         a4         a5        a6          std_intra  std_inter
     PGA	-6.15654	0.948437	-0.12733	-0.563995	-0.704906	0.202352	-0.006282	0.585	0.276	0.647
    0.015	-6.286118	0.968492	-0.138592	-0.544486	-0.704683	0.198911	-0.006026	0.582	0.271	0.642
    0.020	-6.152311	0.948039	-0.127111	-0.567082	-0.704861	0.202411	-0.006282	0.585	0.277	0.647
    0.040	-5.179929	0.836824	-0.057312	-0.870969	-0.717423	0.214708	-0.006282	0.595	0.266	0.652
    0.050	-4.47471	0.736466	-0.032767	-0.783149	-0.708375	0.233852	-0.006538	0.634	0.293	0.698
    0.067	-4.471328	0.761239	-0.038112	-0.830865	-0.693718	0.224785	-0.006282	0.643	0.27	0.697
    0.100	-5.69524	0.992388	-0.087949	-0.994489	-0.676303	0.17052	-0.005769	0.6	0.214	0.637
    0.125	-6.28112	1.09125	-0.113305	-0.984257	-0.661883	0.150526	-0.006026	0.591	0.21	0.627
    0.150	-6.747015	1.165269	-0.135648	-0.945921	-0.650976	0.137638	-0.006282	0.587	0.208	0.623
    0.200	-7.510935	1.280376	-0.178961	-0.858502	-0.635547	0.121722	-0.006795	0.582	0.208	0.618
    0.225	-7.85349	1.330471	-0.202834	-0.817808	-0.629225	0.116343	-0.006795	0.58	0.21	0.617
    0.250	-8.184845	1.378415	-0.223732	-0.789185	-0.623561	0.111718	-0.007051	0.579	0.212	0.616
    0.300	-8.780908	1.464231	-0.277491	-0.721276	-0.614456	0.105755	-0.007308	0.576	0.217	0.615
    0.350	-9.35953	1.54672	-0.321007	-0.682273	-0.606431	0.100281	-0.007308	0.573	0.223	0.615
    0.400	-9.898675	1.62368	-0.369378	-0.644426	-0.599603	0.09654	-0.007564	0.571	0.229	0.615
    0.500	-10.870757	1.762164	-0.462928	-0.582709	-0.589028	0.092056	-0.008077	0.567	0.239	0.615
    0.600	-11.734493	1.882714	-0.53502	-0.548242	-0.579758	0.088741	-0.008077	0.563	0.248	0.615
    0.700	-12.480378	1.984056	-0.589518	-0.526859	-0.572228	0.086895	-0.008333	0.56	0.255	0.615
    0.750	-12.821791	2.028733	-0.60472	-0.525353	-0.568327	0.085697	-0.00859	0.558	0.259	0.615
    0.800	-13.128368	2.068506	-0.622725	-0.518642	-0.565295	0.085524	-0.008333	0.557	0.262	0.615
    0.850	-13.411827	2.104561	-0.637123	-0.514088	-0.56259	0.085554	-0.008333	0.556	0.265	0.616
    0.900	-13.670946	2.136577	-0.647446	-0.511407	-0.560142	0.085801	-0.009103	0.555	0.268	0.616
    0.950	-13.905337	2.16496	-0.658994	-0.50521	-0.558197	0.08654	-0.00859	0.554	0.269	0.616
    1.000	-14.128936	2.190943	-0.664465	-0.502689	-0.555728	0.087027	-0.009103	0.553	0.272	0.616
    1.050	-14.33008	2.213574	-0.670707	-0.497564	-0.553496	0.087956	-0.008846	0.552	0.275	0.617
    1.250	-14.961097	2.277685	-0.677532	-0.484134	-0.548172	0.093821	-0.009872	0.549	0.285	0.618
    1.500	-15.478075	2.315639	-0.636763	-0.497138	-0.546747	0.101611	-0.009103	0.547	0.295	0.622
    2.000	-15.934346	2.311678	-0.518303	-0.504743	-0.560058	0.119721	-0.009872	0.547	0.309	0.628



    """)