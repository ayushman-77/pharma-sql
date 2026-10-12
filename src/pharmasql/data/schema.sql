-- Synthetic pharma commercial data mart (star schema), SQLite dialect.
-- All companies, brands, payers and people are fictional.

CREATE TABLE dim_region (
    region_id    INTEGER PRIMARY KEY,
    region_name  TEXT NOT NULL UNIQUE
);

CREATE TABLE dim_territory (
    territory_id    INTEGER PRIMARY KEY,
    territory_name  TEXT NOT NULL UNIQUE,
    region_id       INTEGER NOT NULL REFERENCES dim_region(region_id),
    state           TEXT NOT NULL
);

CREATE TABLE dim_product (
    product_id        INTEGER PRIMARY KEY,
    brand_name        TEXT NOT NULL UNIQUE,
    generic_name      TEXT NOT NULL,
    therapeutic_area  TEXT NOT NULL,
    manufacturer      TEXT NOT NULL,
    is_own_brand      INTEGER NOT NULL,
    launch_date       TEXT NOT NULL,
    wac_price         REAL NOT NULL
);

CREATE TABLE dim_hcp (
    hcp_id        INTEGER PRIMARY KEY,
    npi           TEXT NOT NULL UNIQUE,
    first_name    TEXT NOT NULL,
    last_name     TEXT NOT NULL,
    specialty     TEXT NOT NULL,
    territory_id  INTEGER NOT NULL REFERENCES dim_territory(territory_id),
    city          TEXT NOT NULL,
    state         TEXT NOT NULL,
    decile        INTEGER NOT NULL,
    is_target     INTEGER NOT NULL
);

CREATE TABLE dim_rep (
    rep_id        INTEGER PRIMARY KEY,
    first_name    TEXT NOT NULL,
    last_name     TEXT NOT NULL,
    territory_id  INTEGER NOT NULL REFERENCES dim_territory(territory_id),
    sales_team    TEXT NOT NULL,
    hire_date     TEXT NOT NULL
);

CREATE TABLE dim_payer (
    payer_id    INTEGER PRIMARY KEY,
    payer_name  TEXT NOT NULL UNIQUE,
    channel     TEXT NOT NULL
);

CREATE TABLE dim_calendar (
    month         TEXT PRIMARY KEY,
    year          INTEGER NOT NULL,
    quarter       INTEGER NOT NULL,
    year_quarter  TEXT NOT NULL,
    month_name    TEXT NOT NULL
);

CREATE TABLE formulary (
    payer_id         INTEGER NOT NULL REFERENCES dim_payer(payer_id),
    product_id       INTEGER NOT NULL REFERENCES dim_product(product_id),
    coverage_status  TEXT NOT NULL,
    prior_auth       INTEGER NOT NULL,
    PRIMARY KEY (payer_id, product_id)
);

CREATE TABLE fact_rx (
    month       TEXT NOT NULL REFERENCES dim_calendar(month),
    hcp_id      INTEGER NOT NULL REFERENCES dim_hcp(hcp_id),
    product_id  INTEGER NOT NULL REFERENCES dim_product(product_id),
    payer_id    INTEGER NOT NULL REFERENCES dim_payer(payer_id),
    nrx         INTEGER NOT NULL,
    trx         INTEGER NOT NULL,
    PRIMARY KEY (month, hcp_id, product_id, payer_id)
);

CREATE TABLE fact_sales (
    month         TEXT NOT NULL REFERENCES dim_calendar(month),
    territory_id  INTEGER NOT NULL REFERENCES dim_territory(territory_id),
    product_id    INTEGER NOT NULL REFERENCES dim_product(product_id),
    units         INTEGER NOT NULL,
    gross_sales   REAL NOT NULL,
    net_sales     REAL NOT NULL,
    PRIMARY KEY (month, territory_id, product_id)
);

CREATE TABLE fact_call (
    call_id     INTEGER PRIMARY KEY,
    call_date   TEXT NOT NULL,
    rep_id      INTEGER NOT NULL REFERENCES dim_rep(rep_id),
    hcp_id      INTEGER NOT NULL REFERENCES dim_hcp(hcp_id),
    product_id  INTEGER NOT NULL REFERENCES dim_product(product_id),
    channel     TEXT NOT NULL
);

CREATE TABLE fact_quota (
    year_quarter      TEXT NOT NULL,
    territory_id      INTEGER NOT NULL REFERENCES dim_territory(territory_id),
    product_id        INTEGER NOT NULL REFERENCES dim_product(product_id),
    target_net_sales  REAL NOT NULL,
    PRIMARY KEY (year_quarter, territory_id, product_id)
);

CREATE INDEX ix_rx_product_month ON fact_rx(product_id, month);
CREATE INDEX ix_rx_hcp ON fact_rx(hcp_id);
CREATE INDEX ix_sales_product_month ON fact_sales(product_id, month);
CREATE INDEX ix_call_hcp ON fact_call(hcp_id);
CREATE INDEX ix_call_rep_date ON fact_call(rep_id, call_date);
