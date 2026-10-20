"""The 29 SESNA star-forming regions: name, distance, and basis.

Transcribed from sesnacomplete.constants.REGIONS: name, d_r_pc (the point
estimate), sigma_pc (its 1-sigma uncertainty), range_kpc, and basis are
carried unchanged; `source` keeps only the distance citation from the old
table's `notes` field, with revision history dropped. Region footprints
(which pixels and tiles belong to a region) come from the coverage and
granule-map products, not from this module, which carries names, distances
and citations only.
"""

import dataclasses as _dc
import math

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str
    d_r_pc: float
    sigma_pc: float
    range_kpc: tuple
    basis: str
    source: str
    # The young-star law's diskless Class III share (owner's ruling,
    # 2026-10-20, `bms_review/briefs/round4_2026-10-20/
    # C11_yso_class3_population.md`): a published mean age of the
    # region's young population, and the disk fraction a published
    # disk-fraction-versus-age relation gives at that age (unit C12's
    # literature table, `bms_review/briefs/round4_2026-10-20/
    # C12_ages_disk_fractions.txt`). Defaulted to the IDENTITY case (no
    # age applied; disk_fraction = 1 leaves the young-star law's level
    # and shares exactly as they were before this ruling) so every
    # region not explicitly overridden below is unaffected; a region
    # whose two values ARE measured overrides both kwargs where its
    # tuple entry is built, below.
    age_myr: float = float("nan")
    age_source: str = ("no published age applied here; disk_fraction defaults to the "
                        "identity case (1.0), not a measurement")
    disk_fraction: float = 1.0
    disk_fraction_source: str = ("identity (no measurement applied): N_LAW_TOTAL_FACTOR = "
                                  "1/disk_fraction = 1, CLASS_SHARE_DISKLESS = 0")


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
    Region("NGC 7129", 912.0, 7.6, (0.904, 0.920), "S",
           "Hunt & Reffert 2023, A&A 673, A114 (VizieR J/A+A/673/A114), cluster "
           "'NGC_7129': mean parallax Plx = 1.0965 +/- 0.0092 mas (standard error "
           "over 136 Gaia DR3 members, 91 used for the distance fit), d = 1000/Plx; "
           "their own maximum-likelihood distance (dist50) is 910.96 "
           "(908.32-913.62) pc, consistent"),
    Region("North America Nebula", 785, 16, (0.769, 0.801), "O",
           "Kuhn & Hillenbrand, Gaia EDR3"),
    Region("Ophiuchus", 137.3, 1.2, (0.1361, 0.1385), "M+S",
           "VLBA parallax (author not recorded in the old table)"),
    Region("Orion A", 418, 21, (0.397, 0.439), "S",
           "Yan, via Pokhrel et al."),
    Region("Orion B", 413.4, 9.2, (0.404, 0.423), "S",
           "Hunt & Reffert 2023, A&A 673, A114 (VizieR J/A+A/673/A114), cluster "
           "'NGC_2068' (the only one of Orion B's three named clusters -- NGC 2068, "
           "NGC 2071, NGC 2024 -- resolved by name in this download; 102 Gaia DR3 "
           "members, 77 within the tidal radius, 87 used for the distance fit): "
           "mean parallax Plx = 2.4188 +/- 0.0054 mas (standard error), d = 1000/Plx "
           "= 413.4 pc; their own maximum-likelihood distance (dist50) for the same "
           "cluster is 404.20 (403.62-404.79) pc, so sigma is the 9.2 pc gap between "
           "the inversion and dist50 (the standard error alone, 0.9 pc, understates "
           "the catalogue's own systematic); consistent with Kounkel et al. 2017's "
           "independent NGC 2068 parallax distance of 405 +/- 16 pc, within 1 sigma "
           "of either estimate"),
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

# ====================================================================
# age_myr / disk_fraction for all 29 regions (unit C12's final table,
# 2026-10-20, `bms_review/briefs/round4_2026-10-20/
# C12_ages_disk_fractions.txt`, status FINAL): the owner's ruling,
# `C11_yso_class3_population.md` -- the young-star law's diskless Class
# III share. ONE relation, applied uniformly (C12's own section 1):
# Richert, Getman, Feigelson et al. 2018, MNRAS 477, 5191, sec. 3.2,
# table 4, `f_disk(t) = exp(-t/tau)`, `t_half = 2.0 Myr`, `tau = t_half
# / ln 2`. `disk_fraction` below is this relation's own value at each
# region's own `age_myr`, rounded to 3 decimals (C12's own table
# column) -- NOT C12's own measured (R18) sums or the Ribas et al. 2014
# band, both reported in C12's table as a cross-check only: the owner
# is deciding separately whether a measured sum ever replaces the
# relation value for some region, which will be a later table edit
# here, not a code change -- no switch between the two is coded.
# `age_source` carries a leading "[P]"/"[M]"/"[L]" flag for the six
# regions C12's own table flags that way (P: no published age, a
# proposal from elsewhere; M: a mixed-model-scale mean; L: a lower
# bound only), then C12's own citation text verbatim.
# ====================================================================

DISK_LIFETIME_MYR = 2.0 / math.log(2.0)  # tau = 2.885 Myr (Richert et al. 2018 t_half/ln 2)

_DF_SOURCE = ("Richert, Getman, Feigelson et al. 2018, MNRAS 477, 5191, table 4, "
              "f = exp(-age/tau), t_half = 2.0 Myr (Siess 2000 ages, Gutermuth 2009 "
              "IRAC classes, protostars included); band: Ribas et al. 2014, A&A 561, A54, "
              "table 5")
_G14 = "Getman et al. 2014, ApJ 787, 108, table 2 (Age_JX, Siess 2000)"
_G18 = "Getman et al. 2018, MNRAS 477, 298, table 4 (Age_JX, Siess 2000)"
_PROP = ("[P] NO PUBLISHED AGE: proposal = median Age_JX of 28 cloud-associated clusters, "
         "Getman et al. 2018, MNRAS 477, 298, table 7")

#: `{region name: (age_myr, age_source)}`, C12's own final table (section 3/4).
_AGES = {
    "AFGL 490": (1.41, _PROP),
    "Aquila": (1.35, _G14 + " W 40 A,U; " + _G18 + " Serpens Main B,U, Serpens South U; "
               "member-weighted mean"),
    "Auriga-California": (1.64, _G18 + " LkHa 101 A,U; member-weighted mean"),
    "Cepheus Flare": (1.85, "[M] MIXED SCALES: mean of NGC 7023 ~1 Myr (Szilagyi et al. "
                      "2021, MNRAS 505, 5164, CMD) and LDN 1251B 2.7 Myr (" + _G18 + ")"),
    "Cepheus OB3": (2.19, _G18 + " Cep OB3b A,B,C,U, Cep A A,U, Cep C A,U; "
                    "member-weighted mean"),
    "Chameleon": (2.38, "Galli et al. 2021, A&A 646, A46, table tab_YSO, Siess 2000: "
                  "Cha I 2.4 (102), Cha II 2.3 (23); weighted"),
    "CrA": (5.0, "Galli et al. 2020, A&A 634, A98, HRD section, on-cloud median "
            "(BT-Settl/BHAC15); misses the embedded Coronet"),
    "Cygnus X": (1.87, "[L] LOWER BOUND: " + _G14 + " DR 21 A,D,E,I,U; " + _G18 +
                 " Sh 2-106 U; Cyg OB2 excluded"),
    "GGD4, CB34": (1.41, _PROP),
    "IC 5146": (1.97, _G18 + " IC 5146 B,U; member-weighted mean"),
    "IRAS 20050+2720": (2.69, _G18 + " IRAS 20050+2720 B,C,D,E,U; member-weighted mean"),
    "L988": (1.7, "Herbig & Dahm 2006, AJ 131, 1530 (Siess 2000, H-alpha population), "
             "as quoted in Allen et al. 2008, ApJ 675, 491, sec. 1"),
    "Lupus": (3.1, "Galli et al. 2020, A&A 643, A148, table tab_YSO, Siess 2000, "
              "full sample (68)"),
    "Mon OB1": (2.49, _G14 + " NGC 2264 A,D,E,G,I,J,K,M,U; member-weighted mean"),
    "Mon R2": (1.50, _G18 + " Mon R2 A,B,C,U, GGD 12-15 A,U; member-weighted mean"),
    "Musca": (2.38, "[P] NO PUBLISHED AGE: proposal = Chameleon value (adjacent complex)"),
    "NGC 7129": (1.8, "Dahm & Hillenbrand 2015, AJ (arXiv:1503.06230), Siess 2000 median "
                 "of 50 spectral-typed members"),
    "North America Nebula": (1.0, "Fang et al. 2020, ApJ 904, 146, sec. 7, spectroscopic "
                             "HRD, Feiden 2016 non-magnetic tracks"),
    "Ophiuchus": (2.0, "Esplin & Luhman 2020, AJ 159, 282, abstract: L1689 and embedded "
                  "L1688, M_K relative to Upper Sco = 10 Myr"),
    "Orion A": (1.59, _G14 + " ONC A-D,U; " + _G18 + " ONC Flanks N,S, OMC 2-3 A-D,U; "
                "member-weighted mean"),
    "Orion B": (1.15, _G14 + " Flame A,U; " + _G18 + " NGC 2068-2071 A-D,U; "
                "member-weighted mean"),
    "Perseus": (2.45, _G18 + " NGC 1333 A,B,U, IC 348 A,B,U; member-weighted mean"),
    "Pipe": (2.6, "Covey et al. 2010, ApJ 722, 971, B59 median age (D'Antona & "
             "Mazzitelli 1998 tracks)"),
    "S131": (4.0, "Sicilia-Aguilar et al. 2005, AJ 130, 188 (Tr 37, Siess 2000), as quoted "
             "in Sicilia-Aguilar et al. 2006, ApJ 645, 1162; field lies 1 deg S of Tr 37"),
    "S140": (1.41, _PROP),
    "S171": (1.81, _G18 + " SFO 2 A,U (8 Age_JX stars); member-weighted mean"),
    "Scorpius": (2.0, "[P] NO PUBLISHED AGE: proposal = Ophiuchus value (Ophiuchus North "
                 "cores, Hatchell et al. 2012, ApJ 754, 104)"),
    "Taurus": (1.51, "Krolikowski, Kraus & Rizzuto 2021, AJ 162, 110, table groupprops "
               "(BHAC15+MIST): C2-L1495 1.34 (55), C1-L1551 1.73 (44); weighted"),
    "Vela D": (1.5, "Strafella et al. 2010, ApJ 719, 9: 1-2 Myr from class statistics and "
               "the 1.5 Myr dynamical time (not an isochrone age)"),
}

_REGIONS = tuple(
    _dc.replace(r,
                age_myr=_AGES[r.name][0],
                age_source=_AGES[r.name][1],
                disk_fraction=round(math.exp(-_AGES[r.name][0] / DISK_LIFETIME_MYR), 3),
                disk_fraction_source=_DF_SOURCE)
    if r.name in _AGES else r
    for r in _REGIONS
)

REGIONS = _REGIONS
REGIONS_BY_NAME = {r.name: r for r in _REGIONS}
