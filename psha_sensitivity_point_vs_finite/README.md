# PSHA sensitivity test: point vs finite ruptures (Portugal)

This folder compares three versions of the Portugal PSHA model (OpenQuake, `local_regional_hip2`, based on ESHM20).
Cases 1 and 2 differ only in how large ruptures are modelled in area and multi-point sources. Case 3 also corrects the tectonic region tags of some sources.

## Cases

**Case 1, point (original):** Every area and multi-point source uses `PointMSR`. Each rupture is a point at hypocentral depth, so Rjb equals the epicentral distance. `pointsource_distance = 75 km` collapses ruptures beyond that distance to points. M8 events in the Gulf of Cadiz are therefore treated as points, and the distance to the site is overestimated.

**Case 2, finite:** All area and multi-point sources use Leonard2014 (SCR inland, Interplate offshore). Ruptures have a real length and width, with aspect ratio 1.0, the original nodal planes and the original depths. `pointsource_distance = 400 km` means no collapse, so Rjb is measured to the rupture plane. Large events now sit much closer to the coast.

**Case 3, finite with corrected tags:** As case 2, with two tag corrections.

* Nine Algarve faults tagged offshore in the original model are retagged **inland**: PTCF017, PTCF01A, PTCF01B, PTCF01E, PTCF01F, PTCF01G, PTCF01H, PTCF01I and PTCF01J. They use the inland GMMs and Leonard2014 SCR.
* The area and multi-point source ESAS129 (Gulf of Cadiz), tagged inland in the original model, is retagged **offshore**. It uses the offshore GMMs and Leonard2014 Interplate.

The scaling relation follows the tag in every case.

**Everything else is identical:** the source tree (21 branches), the GMM tree, Vs30 800 m/s, the IMTs and the sites.

**Effect:**

* Case 2 against case 1: the 475-year ground motion rises by 5–25% for PGA and 5–40% for SA(1.0 s).
* Lagos and Faro change most.
* The tails differ most at low annual frequencies.
* Case 3 against case 2: Faro rises by 22% (PGA) and 24% (SA(1.0 s)), Lagos by 10% and 12%, Beja by 2% and 4%. All other cities change by 2% or less.
* Case 3 against case 1: Faro rises by 50% (PGA) and 73% (SA(1.0 s)), Lagos by 37% and 55%.
* The two tag corrections were applied together, so their separate effects are not isolated.
* Runtime roughly doubles from case 1 to case 2 (1 min to 2 min). Case 3 takes the same time as case 2.

## Results

Nine sites (Lisbon, Porto, Faro, Sines, Lagos, Coimbra, Évora, Beja, Braga). Mean hazard curves, shading shows the 16 to 84% quantiles, the dotted line marks a 475-year return period.

![PGA hazard curves, three cases](figures/hazard_curves_PGA.png)

![SA(1.0 s) hazard curves, three cases](figures/hazard_curves_SA1p0.png)

475-year ground motion (g):

| City | PGA case 1 | PGA case 2 | PGA case 3 | SA(1.0 s) case 1 | SA(1.0 s) case 2 | SA(1.0 s) case 3 |
|---|---|---|---|---|---|---|
| Lisbon | 0.182 | 0.204 | 0.204 | 0.119 | 0.140 | 0.140 |
| Porto | 0.076 | 0.082 | 0.082 | 0.043 | 0.046 | 0.046 |
| Faro | 0.221 | 0.271 | 0.332 | 0.127 | 0.178 | 0.221 |
| Sines | 0.160 | 0.183 | 0.184 | 0.112 | 0.134 | 0.137 |
| Lagos | 0.217 | 0.270 | 0.297 | 0.149 | 0.206 | 0.231 |
| Coimbra | 0.095 | 0.108 | 0.108 | 0.058 | 0.068 | 0.068 |
| Évora | 0.119 | 0.135 | 0.136 | 0.079 | 0.094 | 0.095 |
| Beja | 0.131 | 0.149 | 0.152 | 0.088 | 0.105 | 0.109 |
| Braga | 0.069 | 0.073 | 0.073 | 0.039 | 0.041 | 0.041 |

## Files

| File | Content |
|---|---|
| `figures/hazard_curves_PGA.png` | PGA mean hazard curves, three cases, 3×3 grid of cities |
| `figures/hazard_curves_SA1p0.png` | SA(1.0 s) mean hazard curves, three cases, 3×3 grid of cities |
| `results/psha_point_vs_finite_hazard.xlsx` | `Summary`: 475-year values and ratios. `PGA_curves`, `SA1p0_curves`: mean hazard curves (probability of exceedance in 50 years) for the three cases |

## Run settings

* OpenQuake engine 3.21.0, classical PSHA, investigation time 50 years
* GMM logic tree with four branches: `KothaEtAl2020ESHM20` (high stress, slow attenuation), `Taherian2025ModifiedTest6` and `Morgadoetal2026` (inland and offshore variants)
* Site model: Vs30 800 m/s, ESHM20 region 5
* IMTs: PGA and SA(1.0 s), 30 intensity levels from 0.0005 g to 3 g
* Only the magnitude scaling relation, `pointsource_distance` and (case 3) the tectonic region tags differ between the runs
