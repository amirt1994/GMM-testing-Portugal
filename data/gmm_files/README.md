# GMM source files used in validation - final - v2
| file | class | used as | settings in the validation |
|---|---|---|---|
| `xana_2025_inland.py` | `Morgadoetal2026_inland` | "Xana et al. 2026 (inland)" | as supplied (mag, rjb only) |
| `xana_2025_offshore.py` | `Morgadoetal2026_offshore` | "Xana et al. 2026 (offshore)" | updated version (new `a6` column); mag, rjb only |
| `taherian_2025_inland.py` | `Taherian2025Inland` | "Taherian 2025 (inland)" | hypo_depth 10 km, rake -90 |
| `taherian_2025_offshore.py` | `Taherian2025Offshore` | "Taherian 2025 (offshore)" | hypo_depth 10 km, rake -90 |
| `kotha_2020.py` | `KothaEtAl2020ESHM20` (copy of the OpenQuake file, not modified) | both Kotha curves | see below |

Kotha curves (Vs30 760 m/s, hypo_depth 10 km):
* "Kotha 2020 HighStress-Slow": `KothaEtAl2020ESHM20(sigma_mu_epsilon=1.35563, c3_epsilon=1.732051)`, ESHM20 region 0 inland / 5 offshore, vs30measured True inland / False offshore;
* "Kotha 2020 (ESHM20 region 5, c3_epsilon=0)": `KothaEtAl2020ESHM20(c3_epsilon=0.0)`, region 5 for both, same vs30measured.
`kotha_2020.py` is the OpenQuake 3.23 source (`openquake/hazardlib/gsim/kotha_2020.py`); it imports other OpenQuake modules, so it runs only inside an OpenQuake install.
The Taherian 2025 files are self-contained (coefficients embedded).
