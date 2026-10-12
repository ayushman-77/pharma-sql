"""Generate the synthetic pharma commercial data mart.

The data is fully seeded: the same seed always produces the same database, which the
benchmark's gold answers depend on. The generator models a few real commercial dynamics
so that analytical questions have meaningful answers:

* brand trends (a growing GLP-1-style brand, a declining competitor, an oncology launch curve),
* respiratory seasonality and regional skews,
* rep calls that lift prescribing in the following three months,
* payer mix by therapeutic area, shifted by formulary coverage,
* gross-to-net discounts by channel, so net sales < gross sales.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
START_MONTH = "2024-01-01"
END_MONTH = "2026-09-01"
N_HCPS = 1200
OWN = "Zephyr Pharma"


@dataclass(frozen=True)
class Product:
    brand: str
    generic: str
    area: str
    manufacturer: str
    launch: str
    wac: float
    trend: tuple[float, float]  # demand multiplier at first and last month
    base_share: float  # national share within its therapeutic area
    nrx_ratio: float


PRODUCTS = [
    Product("Cardiozen", "zenapixaban", "Cardiology", OWN, "2019-04-01", 520.0, (1.0, 1.18), 0.45, 0.32),
    Product("Thrombex", "trombaxaban", "Cardiology", "Helix Therapeutics", "2016-09-01", 480.0, (1.0, 0.84), 0.55, 0.28),
    Product("Glucovia", "glucatide", "Diabetes", OWN, "2021-06-01", 890.0, (1.0, 1.85), 0.30, 0.45),
    Product("Insulix", "insuglutide", "Diabetes", "Meridia Labs", "2018-02-01", 760.0, (1.0, 0.78), 0.45, 0.30),
    Product("Metaglide", "metaglinide", "Diabetes", "Corvant Bio", "2020-11-01", 640.0, (1.0, 0.95), 0.25, 0.30),
    Product("Pulmova", "fluvaterol", "Respiratory", OWN, "2018-10-01", 340.0, (1.0, 1.04), 0.50, 0.30),
    Product("Breathon", "brethanide", "Respiratory", "Helix Therapeutics", "2017-03-01", 310.0, (1.0, 0.92), 0.50, 0.28),
    Product("Oncorex", "rexatinib", "Oncology", OWN, "2025-03-01", 14500.0, (1.0, 1.0), 0.40, 0.45),
    Product("Tumorin", "tumorizumab", "Oncology", "Corvant Bio", "2020-01-01", 13200.0, (1.0, 0.72), 0.60, 0.30),
    Product("Neurolin", "neurepant", "Neurology", OWN, "2020-08-01", 760.0, (1.0, 1.32), 0.45, 0.38),
    Product("Migrastat", "migrapitant", "Neurology", "Meridia Labs", "2019-05-01", 820.0, (1.0, 1.05), 0.55, 0.33),
]
LAUNCH_BRAND = "Oncorex"

REGION_TERRITORIES = {
    "Northeast": [("Boston", "MA"), ("New York City", "NY"), ("Philadelphia", "PA"),
                  ("Pittsburgh", "PA"), ("Hartford", "CT")],
    "Southeast": [("Atlanta", "GA"), ("Miami", "FL"), ("Charlotte", "NC"),
                  ("Nashville", "TN"), ("Tampa", "FL")],
    "Midwest": [("Chicago", "IL"), ("Detroit", "MI"), ("Minneapolis", "MN"),
                ("Columbus", "OH"), ("St. Louis", "MO")],
    "Southwest": [("Dallas", "TX"), ("Houston", "TX"), ("Phoenix", "AZ"),
                  ("Albuquerque", "NM"), ("Oklahoma City", "OK")],
    "West": [("Los Angeles", "CA"), ("San Francisco", "CA"), ("Seattle", "WA"),
             ("Denver", "CO"), ("Portland", "OR")],
}

REGION_MULT = {
    ("Glucovia", "Southeast"): 1.35, ("Glucovia", "Southwest"): 1.15, ("Glucovia", "West"): 0.85,
    ("Cardiozen", "Northeast"): 1.25, ("Cardiozen", "West"): 0.90,
    ("Pulmova", "Midwest"): 1.30, ("Pulmova", "Southwest"): 0.85,
    ("Oncorex", "West"): 1.25, ("Oncorex", "Northeast"): 1.15, ("Oncorex", "Southeast"): 0.80,
    ("Neurolin", "West"): 1.20, ("Neurolin", "Midwest"): 0.90,
    ("Insulix", "Midwest"): 1.20, ("Thrombex", "Southeast"): 1.15,
}

# specialty -> (share of HCP universe, {therapeutic area: prescribing weight})
SPECIALTIES = {
    "Cardiology": (0.12, {"Cardiology": 1.0}),
    "Endocrinology": (0.10, {"Diabetes": 1.0}),
    "Pulmonology": (0.09, {"Respiratory": 1.0}),
    "Oncology": (0.06, {"Oncology": 1.0}),
    "Neurology": (0.08, {"Neurology": 1.0}),
    "Internal Medicine": (0.25, {"Cardiology": 0.35, "Diabetes": 0.45}),
    "Family Medicine": (0.30, {"Diabetes": 0.35, "Respiratory": 0.40}),
}
# Monthly TRx across a whole therapeutic area for a decile-10 specialist.
AREA_TOTAL_TRX = {"Cardiology": 60, "Diabetes": 75, "Respiratory": 50, "Oncology": 8, "Neurology": 30}

TEAMS = {"Primary Care": ["Cardiozen", "Glucovia", "Pulmova"], "Specialty": ["Oncorex", "Neurolin"]}

PAYERS = [
    ("BlueHarbor Health", "Commercial", 0.35),
    ("Pinnacle Health Plans", "Commercial", 0.25),
    ("UnityCare", "Commercial", 0.25),
    ("Crestline Insurance", "Commercial", 0.15),
    ("Silverline Part D", "Medicare", 0.60),
    ("Evergreen Medicare Advantage", "Medicare", 0.40),
    ("State Medicaid Managed Care", "Medicaid", 1.00),
    ("Cash / Self-Pay", "Cash", 1.00),
]
CHANNEL_MIX = {  # Commercial, Medicare, Medicaid, Cash
    "Cardiology": (0.40, 0.48, 0.09, 0.03),
    "Diabetes": (0.52, 0.32, 0.13, 0.03),
    "Respiratory": (0.55, 0.23, 0.18, 0.04),
    "Oncology": (0.48, 0.46, 0.06, 0.00),
    "Neurology": (0.68, 0.15, 0.13, 0.04),
}
CHANNELS = ("Commercial", "Medicare", "Medicaid", "Cash")
GROSS_TO_NET = {"Commercial": 0.38, "Medicare": 0.30, "Medicaid": 0.60, "Cash": 0.0}

# (payer, brand) -> (coverage_status, prior_auth); everything else is Preferred without PA.
FORMULARY_OVERRIDES = {
    ("Pinnacle Health Plans", "Glucovia"): ("Non-Preferred", 1),
    ("Pinnacle Health Plans", "Thrombex"): ("Not Covered", 0),
    ("Crestline Insurance", "Pulmova"): ("Not Covered", 0),
    ("Crestline Insurance", "Neurolin"): ("Non-Preferred", 1),
    ("BlueHarbor Health", "Insulix"): ("Non-Preferred", 0),
    ("UnityCare", "Metaglide"): ("Not Covered", 0),
    ("UnityCare", "Cardiozen"): ("Non-Preferred", 0),
    ("Silverline Part D", "Oncorex"): ("Non-Preferred", 1),
    ("Silverline Part D", "Migrastat"): ("Not Covered", 0),
    ("Evergreen Medicare Advantage", "Breathon"): ("Non-Preferred", 0),
    ("State Medicaid Managed Care", "Glucovia"): ("Non-Preferred", 1),
    ("State Medicaid Managed Care", "Neurolin"): ("Not Covered", 0),
}
COVERAGE_MULT = {"Preferred": 1.0, "Non-Preferred": 0.55, "Not Covered": 0.08}
ALWAYS_PRIOR_AUTH = {"Oncorex", "Tumorin"}


def npi_with_check_digit(base9: str) -> str:
    """Return a 10-digit NPI with a valid Luhn check digit (CMS uses the 80840 prefix)."""
    total = 0
    for i, ch in enumerate(reversed("80840" + base9)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return base9 + str((10 - total % 10) % 10)


def _business_days(month: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(month, month + pd.offsets.MonthEnd(0))


def generate(db_path: Path, seed: int = SEED) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    fake = Faker("en_US")
    Faker.seed(seed)

    months = pd.date_range(START_MONTH, END_MONTH, freq="MS")
    n_months = len(months)
    month_str = months.strftime("%Y-%m-%d").to_numpy()
    prod_idx = {p.brand: i for i, p in enumerate(PRODUCTS)}

    # ---- geography -------------------------------------------------------------------
    regions = list(REGION_TERRITORIES)
    territories = [
        (tid, name, regions.index(region) + 1, state)
        for tid, (region, name, state) in enumerate(
            ((r, n, s) for r, terrs in REGION_TERRITORIES.items() for n, s in terrs), start=1
        )
    ]
    terr_region = {t[0]: regions[t[2] - 1] for t in territories}

    # ---- HCP universe ----------------------------------------------------------------
    spec_names = list(SPECIALTIES)
    spec_p = np.array([SPECIALTIES[s][0] for s in spec_names])
    hcp_spec = rng.choice(spec_names, size=N_HCPS, p=spec_p / spec_p.sum())
    terr_weights = rng.uniform(0.7, 1.3, len(territories))
    hcp_terr = rng.choice([t[0] for t in territories], size=N_HCPS, p=terr_weights / terr_weights.sum())
    decile_p = np.arange(10, 0, -1, dtype=float)
    hcp_decile = rng.choice(np.arange(1, 11), size=N_HCPS, p=decile_p / decile_p.sum())
    target_p = np.where(hcp_decile >= 7, 0.9, np.where(hcp_decile >= 4, 0.4, 0.05))
    hcp_target = (rng.random(N_HCPS) < target_p).astype(int)

    npis: set[str] = set()
    hcp_rows = []
    for i in range(N_HCPS):
        while True:
            npi = npi_with_check_digit("1" + "".join(rng.choice(list("0123456789"), 8)))
            if npi not in npis:
                npis.add(npi)
                break
        t = territories[hcp_terr[i] - 1]
        hcp_rows.append((i + 1, npi, fake.first_name(), fake.last_name(), str(hcp_spec[i]),
                         int(hcp_terr[i]), t[1], t[3], int(hcp_decile[i]), int(hcp_target[i])))

    rep_rows = []
    rep_lookup: dict[tuple[int, str], int] = {}
    for t in territories:
        for team in TEAMS:
            rep_id = len(rep_rows) + 1
            hire = pd.Timestamp("2014-01-01") + pd.Timedelta(days=int(rng.integers(0, 3600)))
            rep_rows.append((rep_id, fake.first_name(), fake.last_name(), t[0], team,
                             hire.strftime("%Y-%m-%d")))
            rep_lookup[(t[0], team)] = rep_id
    brand_team = {b: team for team, brands in TEAMS.items() for b in brands}

    # ---- product demand curves (product x month) -------------------------------------
    trend = np.ones((len(PRODUCTS), n_months))
    for j, p in enumerate(PRODUCTS):
        trend[j] = np.linspace(p.trend[0], p.trend[1], n_months)
        if p.area == "Respiratory":
            trend[j] *= 1 + 0.22 * np.cos(2 * np.pi * (months.month.to_numpy() - 1) / 12)
        if p.brand == LAUNCH_BRAND:
            since = (months.year - 2025) * 12 + months.month - 3
            trend[j] = np.where(since >= 0, 1 / (1 + np.exp(-(since.to_numpy() - 6) / 2.5)), 0.0)
    launch_idx = np.array([
        max(0, (pd.Timestamp(p.launch).year - 2024) * 12 + pd.Timestamp(p.launch).month - 1)
        for p in PRODUCTS
    ])

    # ---- HCP x product pairs -----------------------------------------------------------
    pair_hcp, pair_prod, pair_base = [], [], []
    for i in range(N_HCPS):
        for area, weight in SPECIALTIES[hcp_spec[i]][1].items():
            area_prods = [j for j, p in enumerate(PRODUCTS) if p.area == area]
            alpha = np.array([PRODUCTS[j].base_share for j in area_prods]) * 8
            prefs = rng.dirichlet(alpha)
            region = terr_region[hcp_terr[i]]
            for j, pref in zip(area_prods, prefs, strict=True):
                mult = REGION_MULT.get((PRODUCTS[j].brand, region), 1.0)
                base = AREA_TOTAL_TRX[area] * weight * (hcp_decile[i] / 10) ** 1.6 * pref * mult
                pair_hcp.append(i)
                pair_prod.append(j)
                pair_base.append(base)
    pair_hcp = np.array(pair_hcp)
    pair_prod = np.array(pair_prod)
    pair_base = np.array(pair_base)
    n_pairs = len(pair_hcp)

    # ---- calls (own brands only) and their lagged lift on prescribing ------------------
    is_own = np.array([p.manufacturer == OWN for p in PRODUCTS])
    promoted = is_own[pair_prod]
    n_promoted = np.bincount(pair_hcp[promoted], minlength=N_HCPS)
    call_rate = np.where(hcp_target[pair_hcp] == 1, 0.25 + 0.12 * hcp_decile[pair_hcp], 0.03)
    call_rate = call_rate * np.where(n_promoted[pair_hcp] > 1, 0.75, 1.0) * promoted
    active = np.arange(n_months)[None, :] >= launch_idx[pair_prod][:, None]
    calls = rng.poisson(call_rate[:, None] * active)
    lagged = np.zeros_like(calls)
    for lag in (1, 2, 3):
        lagged[:, lag:] += calls[:, :-lag]
    lift = np.minimum(1 + 0.05 * lagged, 1.35)

    noise = rng.gamma(shape=8.0, scale=1 / 8.0, size=(n_pairs, n_months))
    lam = pair_base[:, None] * trend[pair_prod] * noise * lift * active
    trx = rng.poisson(lam)

    # ---- payer split -------------------------------------------------------------------
    payer_names = [p[0] for p in PAYERS]
    coverage = {}
    for prod in PRODUCTS:
        for payer, channel, _ in PAYERS:
            if channel == "Cash":
                continue
            status, pa = FORMULARY_OVERRIDES.get((payer, prod.brand), ("Preferred", 0))
            coverage[(payer, prod.brand)] = (status, 1 if prod.brand in ALWAYS_PRIOR_AUTH else pa)
    payer_w = np.zeros((len(PRODUCTS), len(PAYERS)))
    for j, prod in enumerate(PRODUCTS):
        mix = dict(zip(CHANNELS, CHANNEL_MIX[prod.area], strict=True))
        for k, (payer, channel, within) in enumerate(PAYERS):
            status = coverage.get((payer, prod.brand), ("Preferred", 0))[0]
            payer_w[j, k] = mix[channel] * within * COVERAGE_MULT[status]
    payer_w /= payer_w.sum(axis=1, keepdims=True)

    nz_pair, nz_month = np.nonzero(trx)
    remaining = trx[nz_pair, nz_month]
    w = payer_w[pair_prod[nz_pair]]
    split = np.zeros((len(nz_pair), len(PAYERS)), dtype=np.int64)
    rest_w = np.ones(len(nz_pair))
    for k in range(len(PAYERS) - 1):
        prob = np.clip(np.divide(w[:, k], rest_w, out=np.zeros_like(rest_w), where=rest_w > 0), 0, 1)
        split[:, k] = rng.binomial(remaining, prob)
        remaining = remaining - split[:, k]
        rest_w = rest_w - w[:, k]
    split[:, -1] = remaining

    nrx_ratio = np.array([p.nrx_ratio for p in PRODUCTS])[pair_prod[nz_pair]]
    launch_cells = pair_prod[nz_pair] == prod_idx[LAUNCH_BRAND]
    months_since = nz_month - launch_idx[prod_idx[LAUNCH_BRAND]]
    nrx_ratio = np.where(launch_cells, np.clip(0.85 - 0.035 * months_since, 0.45, 0.85), nrx_ratio)
    nrx = rng.binomial(split, nrx_ratio[:, None])

    cell, payer_k = np.nonzero(split)
    rx = pd.DataFrame({
        "month": month_str[nz_month[cell]],
        "hcp_id": pair_hcp[nz_pair[cell]] + 1,
        "product_id": pair_prod[nz_pair[cell]] + 1,
        "payer_id": payer_k + 1,
        "nrx": nrx[cell, payer_k],
        "trx": split[cell, payer_k],
    })

    # ---- sales (own brands, territory x month) ----------------------------------------
    hcp_df = pd.DataFrame(hcp_rows, columns=["hcp_id", "npi", "first_name", "last_name",
                                             "specialty", "territory_id", "city", "state",
                                             "decile", "is_target"])
    own_ids = [j + 1 for j, p in enumerate(PRODUCTS) if p.manufacturer == OWN]
    s = rx[rx["product_id"].isin(own_ids)].merge(hcp_df[["hcp_id", "territory_id"]], on="hcp_id")
    wac = {j + 1: p.wac for j, p in enumerate(PRODUCTS)}
    net_factor = {k + 1: 1 - GROSS_TO_NET[p[1]] for k, p in enumerate(PAYERS)}
    s["gross_sales"] = s["trx"] * s["product_id"].map(wac)
    s["net_sales"] = s["gross_sales"] * s["payer_id"].map(net_factor)
    sales = (s.groupby(["month", "territory_id", "product_id"], as_index=False)
              .agg(units=("trx", "sum"), gross_sales=("gross_sales", "sum"),
                   net_sales=("net_sales", "sum")))
    sales[["gross_sales", "net_sales"]] = sales[["gross_sales", "net_sales"]].round(2)

    # ---- quotas (own brands, territory x quarter; includes next quarter's plan) --------
    cal = pd.DataFrame({"month": month_str, "year": months.year, "quarter": months.quarter})
    cal["year_quarter"] = cal["year"].astype(str) + "-Q" + cal["quarter"].astype(str)
    cal["month_name"] = months.strftime("%B")
    q = (sales.merge(cal[["month", "year_quarter"]], on="month")
              .groupby(["year_quarter", "territory_id", "product_id"], as_index=False)["net_sales"]
              .sum())
    oncorex_id = prod_idx[LAUNCH_BRAND] + 1
    q = q[~((q["product_id"] == oncorex_id) & (q["year_quarter"] < "2025-Q2"))]
    q["target_net_sales"] = (q["net_sales"] * rng.uniform(0.90, 1.12, len(q)) / 1000).round() * 1000
    nxt = q[q["year_quarter"] == "2026-Q3"].copy()
    nxt["year_quarter"] = "2026-Q4"
    nxt["target_net_sales"] = (nxt["net_sales"] * rng.uniform(1.02, 1.10, len(nxt)) / 1000).round() * 1000
    quota = pd.concat([q, nxt])[["year_quarter", "territory_id", "product_id", "target_net_sales"]]

    # ---- call rows -----------------------------------------------------------------------
    c_pair, c_month = np.nonzero(calls)
    call_rows = []
    for pi, mi in zip(c_pair, c_month, strict=True):
        hcp = pair_hcp[pi]
        brand = PRODUCTS[pair_prod[pi]].brand
        rep = rep_lookup[(int(hcp_terr[hcp]), brand_team[brand])]
        days = _business_days(months[mi])
        year_frac = mi / (n_months - 1)
        ch_p = np.array([0.65 - 0.15 * year_frac, 0.20 + 0.10 * year_frac, 0.15 + 0.05 * year_frac])
        for _ in range(calls[pi, mi]):
            call_rows.append((days[rng.integers(len(days))].strftime("%Y-%m-%d"), rep, int(hcp) + 1,
                              int(pair_prod[pi]) + 1,
                              str(rng.choice(["In-Person", "Virtual", "Email"], p=ch_p / ch_p.sum()))))
    call_rows.sort()
    call_rows = [(i + 1, *row) for i, row in enumerate(call_rows)]

    # ---- write -------------------------------------------------------------------------
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    ddl = resources.files("pharmasql.data").joinpath("schema.sql").read_text()
    with sqlite3.connect(db_path) as con:
        con.executescript(ddl)
        con.executemany("INSERT INTO dim_region VALUES (?,?)",
                        [(i + 1, r) for i, r in enumerate(regions)])
        con.executemany("INSERT INTO dim_territory VALUES (?,?,?,?)", territories)
        con.executemany("INSERT INTO dim_product VALUES (?,?,?,?,?,?,?,?)", [
            (j + 1, p.brand, p.generic, p.area, p.manufacturer, int(p.manufacturer == OWN),
             p.launch, p.wac) for j, p in enumerate(PRODUCTS)])
        con.executemany("INSERT INTO dim_hcp VALUES (?,?,?,?,?,?,?,?,?,?)", hcp_rows)
        con.executemany("INSERT INTO dim_rep VALUES (?,?,?,?,?,?)", rep_rows)
        con.executemany("INSERT INTO dim_payer VALUES (?,?,?)",
                        [(k + 1, p[0], p[1]) for k, p in enumerate(PAYERS)])
        con.executemany("INSERT INTO dim_calendar VALUES (?,?,?,?,?)",
                        cal.itertuples(index=False, name=None))
        con.executemany("INSERT INTO formulary VALUES (?,?,?,?)", [
            (payer_names.index(payer) + 1, prod_idx[brand] + 1, status, pa)
            for (payer, brand), (status, pa) in coverage.items()])
        con.executemany("INSERT INTO fact_rx VALUES (?,?,?,?,?,?)", rx.itertuples(index=False, name=None))
        con.executemany("INSERT INTO fact_sales VALUES (?,?,?,?,?,?)",
                        sales.itertuples(index=False, name=None))
        con.executemany("INSERT INTO fact_call VALUES (?,?,?,?,?,?)", call_rows)
        con.executemany("INSERT INTO fact_quota VALUES (?,?,?,?)", quota.itertuples(index=False, name=None))
        con.execute("ANALYZE")
        counts = {
            t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
            for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                    "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        }
    con.close()
    with sqlite3.connect(db_path) as con:
        con.execute("VACUUM")
    con.close()
    return counts
