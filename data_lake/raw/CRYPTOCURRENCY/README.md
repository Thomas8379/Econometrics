# Cryptocurrency Data Lake — Raw Data

## Overview

This directory contains two data sources for cryptocurrency research:

1. **CoinGecko daily snapshots** (`raw_price_daily/`) — 483 coins, price/market-cap/volume
2. **CoinMetrics on-chain data** (`coinmetrics_data/`) — 518 coins, multi-measure daily series

Total raw data: ~1.2 GB. All files are gitignored; only this README and `manifest.json` are tracked.

---

## Source 1: CoinGecko Daily Prices (`raw_price_daily/`)

**Files:** 483 CSVs, named `<TICKER>-usd-max.csv`

**Columns:**
| Column | Type | Description |
|---|---|---|
| `snapped_at` | datetime (UTC) | Trading day (midnight UTC) |
| `price` | float | USD close price |
| `market_cap` | float | USD market capitalisation |
| `total_volume` | float | USD 24h trading volume |

**Coverage:** BTC from 2013-04-28; most altcoins from 2017–2021 launch dates.

**Sample (BTC-usd-max.csv):**
```
snapped_at,price,market_cap,total_volume
2013-04-28 00:00:00 UTC,135.30,1500517590,0
2013-04-29 00:00:00 UTC,141.96,1575032004.0,0.0
2013-04-30 00:00:00 UTC,135.30,1501657493.0,0.0
2025-05-08 00:00:00 UTC,97026.49,1927134557858.27,64376174032.08
2025-05-09 00:00:00 UTC,103076.28,2047883753852.69,50020047720.66
2025-05-10 00:00:00 UTC,102962.54,2045046445523.52,40800894903.23
```

---

## Source 2: CoinMetrics On-Chain Data (`coinmetrics_data/`)

**Coverage:** 518 tickers. Each ticker has its own subdirectory containing multiple measure categories.

**Directory structure per coin:**
```
coinmetrics_data/<TICKER>/
├── price/                    # USD price, BTC price, reference rates
├── volatility/               # Rolling vol (7/14/21/30/60/90d), GARCH, EWMA
├── volume_measures/          # Spot volume, real volume
├── market_cap/               # Market cap USD
├── network_value/            # NVT ratio, network value metrics
├── transactions/             # Tx count, tx value, fees
├── fees/                     # Fee per tx, total fees
├── issuance/                 # Issuance rate, newly issued supply
├── supply_metrics/           # Circulating supply, total supply
├── supply_activity/          # Active supply (1d/7d/30d/1y)
├── supply_address_balance/   # Supply held at various balance thresholds
├── supply_address_native/    # Same, in native units
├── supply_address_usd/       # Same, in USD
├── supply_concentration/     # Herfindahl, top-address concentration
├── address_balance_count/    # Count of addresses by balance threshold
├── address_balance_native/   # Addresses by native balance
├── address_balance_usd/      # Addresses by USD balance
├── difficulty/               # Mining difficulty (PoW coins)
├── all_measures/             # Combined wide table (all measures merged)
├── cross_sectional/          # Pre-merged cross-sectional panel data
├── variance_measures/        # Variance-based risk metrics
├── idiosyncratic/            # Coin-specific metrics
└── other/                    # Miscellaneous CoinMetrics metrics
```

Each directory contains `.csv` and `.parquet` versions of the data.

**Sample (BTC volatility, `coinmetrics_data/BTC/volatility/volatility_measures.csv`):**
```
date,vol_rolling_7d,vol_rolling_14d,vol_rolling_30d,vol_rolling_90d,vol_garch,vol_ewma
2013-05-26,,,,, 0.0402,
2013-05-27,,,,, 0.0389,
2020-03-13, 0.189, 0.142, 0.098, 0.072, 0.183, 0.191
2021-11-10, 0.041, 0.037, 0.038, 0.052, 0.042, 0.039
```

**Sample (BTC price, `coinmetrics_data/BTC/price/btc_PriceUSD.csv`):**
```
time,PriceUSD
2009-01-03 00:00:00+00:00, 0.0
2010-07-18 00:00:00+00:00, 0.09
2013-04-10 00:00:00+00:00, 259.34
2021-11-10 00:00:00+00:00, 68789.63
2025-05-10 00:00:00+00:00, 102962.54
```

---

## Asset Catalogue (`cm-assets.csv`)

A 1,931-row registry mapping CoinMetrics ticker symbols to full asset names.

```
Asset,Full name
btc,Bitcoin
eth,Ethereum
sol,Solana
bnb,BNB
...
```

---

## Notable Tickers Available

| Ticker | Name | BTC data from |
|---|---|---|
| BTC | Bitcoin | 2009-01-03 |
| ETH | Ethereum | 2015-08-08 |
| SOL | Solana | 2020-04-11 |
| BNB | BNB | 2017-07-25 |
| AAVE | Aave | 2017-11-18 |
| 1INCH | 1inch | 2020-12-26 |
| UNI | Uniswap | 2020-09-17 |

---

## Curated Datasets (derived, in `data_lake/curated/`)

These Parquet files are produced by scripts in `Showing what we can do/working_space/`:

| File | Obs | Description |
|---|---|---|
| `crypto_panel_v1.parquet` | ~1M | 433 coins, daily returns/vol/momentum |
| `crypto_btc_onchain_v1.parquet` | 5,428 | BTC fundamentals: NVT, MVRV, fees, tx |
| `crypto_majors_v1.parquet` | ~31K | Top 10 coins with vol measures |
| `crypto_btc_halving_v1.parquet` | — | BTC with halving event windows annotated |

Recreate with:
```bash
python "Showing what we can do/working_space/01_curate_crypto.py"
python "Showing what we can do/working_space/02_btc_fundamentals.py"
```
