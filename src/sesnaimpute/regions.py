"""The 30 SESNA star-forming regions: name, distance, and basis.

Transcribed from sesnacomplete.constants.REGIONS: name, d_r_pc (the point
estimate), sigma_pc (its 1-sigma uncertainty), range_kpc, and basis are
carried unchanged; `source` keeps only the distance citation from the old
table's `notes` field, with revision history dropped. Region footprints
(which pixels and tiles belong to a region) come from the coverage and
granule-map products, not from this module, which carries names, distances
and citations only.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str
    d_r_pc: float
    sigma_pc: float
    range_kpc: tuple
    basis: str
    source: str


_REGIONS = (
    Region("AFGL 490", 1060, 75, (0.985, 1.135), "S+D",
           "median of seven Gaia-era determinations within 1 deg of the footprint: "
           "Prisinzano, Damiani, Sciortino et al. 2022, A&A 664, A175, table 3; "
           "Mullen, Mast, Kounkel, Stassun, Roman-Lopes & Tan 2025, ApJ (arXiv:2508.09393), table 1; "
           "Chen, Huang, Yuan et al. 2020, MNRAS 493, 351; "
           "Hunt & Reffert 2023, A&A 673, A114 / 2024, A&A 686, A42"),
    Region("Aquila", 436, 9, (0.427, 0.445), "M+S",
           "Ortiz-Leon, VLBA + Gaia parallax (Pokhrel et al. concurs)"),
    Region("Auriga-California", 470, 24, (0.446, 0.494), "D",
           "Zucker, Speagle, Schlafly, Green, Finkbeiner, Goodman & Alves 2019, ApJ 879, 125, "
           "table 3 ('California' row), Gaia DR2 3-D dust"),
    Region("BD+40o4124", 980, 196, (0.784, 1.176), "A-FLAG",
           "Shevchenko 1991, via Sandell 2012"),
    Region("Cepheus Flare", 358, 32, (0.326, 0.39), "M",
           "Dzib et al., VLBA parallaxes of 47 YSOs"),
    Region("Cepheus OB3", 830, 40, (0.790, 0.870), "M+S",
           "Reid, Menten, Brunthaler et al. 2019, ApJ 885, 131, table 1 (Cep A VLBI maser parallax); "
           "corroborated by Karnath, Prchlik, Gutermuth et al. 2019, ApJ 871, 46 and "
           "Dias et al. 2021, MNRAS 504, 356"),
    Region("Chameleon", 192, 6, (0.186, 0.198), "M",
           "source: not recorded in the old table"),
    Region("CrA", 154, 4, (0.15, 0.158), "M",
           "Dzib et al."),
    Region("Cygnus X", 1400, 80, (1.32, 1.48), "O",
           "Rygl et al., maser parallax (combined)"),
    Region("GGD4, CB34", 1356, 77, (1.279, 1.433), "O",
           "mean of three Zucker et al. dust sightlines (1349/1396/1322 pc)"),
    Region("IC 5146", 813, 106, (0.707, 0.919), "M",
           "Dzib et al., VLBA parallaxes of 62 YSOs"),
    Region("IRAS 20050+2720", 1326, 84, (1.242, 1.410), "S",
           "Prisinzano, Damiani, Sciortino et al. 2022, A&A 664, A175, table 3 group 391 "
           "('65.78-2.61'), 134 Gaia EDR3 members"),
    Region("L988", 620, 32, (0.588, 0.652), "O",
           "Zucker et al., dust map d50 (612/627 pc sightlines)"),
    Region("Lupus", 158.3, 0.6, (0.1577, 0.1589), "M",
           "Galli et al., 113 Gaia members"),
    Region("Mon OB1", 745, 37.12, (0.70788, 0.78212), "D",
           "Zucker, Speagle, Schlafly, Green, Finkbeiner, Goodman & Alves 2019, ApJ 879, 125, table 3"),
    Region("Mon R2", 860, 31, (0.829, 0.891), "S",
           "Pokhrel et al., Gaia fit of >300 SESNA sources"),
    Region("Musca", 172, 16, (0.156, 0.188), "D",
           "Zucker, Goodman, Alves, Bialy, Koch, Speagle et al. 2021, ApJ 919, 35, table 1 "
           "('Musca' row)"),
    Region("NGC 7129", 926, 163, (0.763, 1.089), "M",
           "Dzib et al., VLBA parallaxes of 40 YSOs"),
    Region("North America Nebula", 785, 16, (0.769, 0.801), "O",
           "Kuhn & Hillenbrand, Gaia EDR3"),
    Region("Ophiuchus", 137.3, 1.2, (0.1361, 0.1385), "M+S",
           "VLBA parallax (author not recorded in the old table)"),
    Region("Orion A", 418, 21, (0.397, 0.439), "S",
           "Yan, via Pokhrel et al."),
    Region("Orion B", 418, 21, (0.397, 0.439), "S",
           "Yan, via Pokhrel et al. (as Orion A)"),
    Region("Perseus", 294, 15.133, (0.2789, 0.3091), "D",
           "Zucker et al."),
    Region("Pipe", 163, 5, (0.158, 0.168), "M",
           "B59 member astrometry"),
    Region("S131", 918, 48, (0.87, 0.966), "O",
           "mean of four Zucker et al. dust sightlines (identified with IC 1396 via the Sharpless catalog)"),
    Region("S140", 908, 30, (0.878, 0.938), "S",
           "Szilagyi, Kun, Abraham & Marton 2023, MNRAS 520, 1390, table 2, group 8 "
           "('SH 2-140'), 45 Gaia EDR3 members"),
    Region("S171", 1010, 40, (0.970, 1.050), "S",
           "Wiesneth, Muzic & Almendros-Abad 2025, A&A 703, A193, table 2, column d2 "
           "('Berkeley 59'), Gaia DR3 with Lindegren et al. 2021 parallax-bias correction"),
    Region("Scorpius", 146, 6.708, (0.1393, 0.1527), "M",
           "Galli et al., kinematic distance"),
    Region("Taurus", 141, 7.28, (0.1337, 0.1483), "D",
           "Zucker et al."),
    Region("Vela D", 930, 80, (0.850, 1.010), "D",
           "Dharmawardena, Bailer-Jones, Fouesneau et al. 2023, MNRAS 519, 228, table 3, "
           "region 'Vela D', leaf L70"),
)

REGIONS = _REGIONS
REGIONS_BY_NAME = {r.name: r for r in _REGIONS}
