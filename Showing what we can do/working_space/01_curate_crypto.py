"""
01_curate_crypto.py — Build curated crypto datasets for econometric analysis.

Produces:
  1. crypto_panel_v1.parquet  — daily panel: 401 coins, price/mcap/volume/returns
  2. crypto_btc_onchain_v1.parquet — BTC on-chain fundamentals (fees, tx, NVT, supply)
  3. crypto_majors_v1.parquet — Top 10 coins with volatility measures merged
  4. crypto_btc_halving_v1.parquet — BTC data annotated with halving events
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

RAW_DIR = "C:/Econometrics/data_lake/raw/CRYPTOCURRENCY"
CURATED_DIR = "C:/Econometrics/data_lake/curated"
os.makedirs(CURATED_DIR, exist_ok=True)


def build_price_panel():
    """Load all raw_price_daily CSVs into a returns panel."""
    price_dir = os.path.join(RAW_DIR, "raw_price_daily")
    files = sorted(f for f in os.listdir(price_dir) if f.endswith(".csv"))

    frames = []
    for f in files:
        ticker = f.replace("-usd-max.csv", "").upper()
        df = pd.read_csv(os.path.join(price_dir, f))
        df["ticker"] = ticker
        if len(df) >= 500:
            frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={"snapped_at": "date"})
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_localize(None)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Compute returns
    panel["log_price"] = np.log(panel["price"].clip(lower=1e-10))
    panel["log_return"] = panel.groupby("ticker")["log_price"].diff()
    panel["simple_return"] = panel.groupby("ticker")["price"].pct_change()

    # Compute log market cap and log volume
    panel["log_market_cap"] = np.log(panel["market_cap"].clip(lower=1))
    panel["log_volume"] = np.log(panel["total_volume"].clip(lower=1))

    # Realized volatility (rolling 30d)
    panel["vol_30d"] = panel.groupby("ticker")["log_return"].transform(
        lambda x: x.rolling(30, min_periods=20).std() * np.sqrt(365)
    )

    # Momentum (past 7d, 30d cumulative return)
    for w in [7, 30]:
        panel[f"mom_{w}d"] = panel.groupby("ticker")["log_return"].transform(
            lambda x: x.rolling(w, min_periods=max(3, w // 2)).sum()
        )

    # Amihud illiquidity: |return| / dollar volume
    panel["amihud"] = panel["log_return"].abs() / panel["total_volume"].clip(lower=1)

    print(f"Price panel: {panel.shape[0]:,} rows, {panel['ticker'].nunique()} coins")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    return panel


def build_btc_onchain():
    """Merge BTC on-chain metrics from coinmetrics CSV files."""
    btc_dir = os.path.join(RAW_DIR, "coinmetrics_data", "BTC")

    metrics = {}

    # Price
    df = pd.read_csv(os.path.join(btc_dir, "price", "btc_PriceUSD.csv"))
    df.columns = ["date", "price_usd"]
    metrics["price"] = df

    # Market cap
    df = pd.read_csv(os.path.join(btc_dir, "market_cap", "btc_CapMrktCurUSD.csv"))
    df.columns = ["date", "market_cap_usd"]
    metrics["mcap"] = df

    # MVRV ratio
    df = pd.read_csv(os.path.join(btc_dir, "market_cap", "btc_CapMVRVCur.csv"))
    df.columns = ["date", "mvrv"]
    metrics["mvrv"] = df

    # NVT ratio
    df = pd.read_csv(os.path.join(btc_dir, "network_value", "btc_NVTAdj.csv"))
    df.columns = ["date", "nvt_adj"]
    metrics["nvt"] = df

    # Transaction count
    df = pd.read_csv(os.path.join(btc_dir, "transactions", "btc_TxCnt.csv"))
    df.columns = ["date", "tx_count"]
    metrics["txcnt"] = df

    # Transfer value (USD)
    df = pd.read_csv(os.path.join(btc_dir, "transactions", "btc_TxTfrValAdjUSD.csv"))
    df.columns = ["date", "tx_transfer_val_usd"]
    metrics["txval"] = df

    # Fees (USD)
    df = pd.read_csv(os.path.join(btc_dir, "fees", "btc_FeeTotUSD.csv"))
    df.columns = ["date", "fee_total_usd"]
    metrics["fees"] = df

    # Mean fee (USD)
    df = pd.read_csv(os.path.join(btc_dir, "fees", "btc_FeeMeanUSD.csv"))
    df.columns = ["date", "fee_mean_usd"]
    metrics["feemean"] = df

    # Difficulty
    df = pd.read_csv(os.path.join(btc_dir, "difficulty", "btc_DiffMean.csv"))
    df.columns = ["date", "difficulty"]
    metrics["diff"] = df

    # Active addresses
    df = pd.read_csv(
        os.path.join(btc_dir, "address_balance_count", "btc_AdrActCnt.csv")
    )
    df.columns = ["date", "active_addresses"]
    metrics["adr"] = df

    # Supply metrics
    df = pd.read_csv(os.path.join(btc_dir, "supply_metrics", "btc_SplyCur.csv"))
    df.columns = ["date", "supply_current"]
    metrics["supply"] = df

    # Issuance
    df = pd.read_csv(os.path.join(btc_dir, "issuance", "btc_IssContNtv.csv"))
    df.columns = ["date", "issuance_native"]
    metrics["issuance"] = df

    # Merge all
    result = metrics["price"].copy()
    result["date"] = pd.to_datetime(result["date"], utc=True).dt.tz_localize(None)
    for key, df in metrics.items():
        if key == "price":
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        result = result.merge(df, on="date", how="left")

    result = result.sort_values("date").reset_index(drop=True)

    # Derived features
    result["log_price"] = np.log(result["price_usd"].clip(lower=0.01))
    result["log_return"] = result["log_price"].diff()
    result["log_market_cap"] = np.log(result["market_cap_usd"].clip(lower=1))
    result["log_tx_count"] = np.log(result["tx_count"].clip(lower=1))
    result["log_transfer_val"] = np.log(result["tx_transfer_val_usd"].clip(lower=1))
    result["log_active_addr"] = np.log(result["active_addresses"].clip(lower=1))
    result["log_fee_total"] = np.log(result["fee_total_usd"].clip(lower=0.01))
    result["log_difficulty"] = np.log(result["difficulty"].clip(lower=1))
    result["log_nvt"] = np.log(result["nvt_adj"].clip(lower=0.01))
    result["log_mvrv"] = np.log(result["mvrv"].clip(lower=0.01))

    # Volatility
    result["vol_30d"] = (
        result["log_return"].rolling(30, min_periods=20).std() * np.sqrt(365)
    )
    result["vol_90d"] = (
        result["log_return"].rolling(90, min_periods=60).std() * np.sqrt(365)
    )

    print(f"BTC on-chain: {result.shape[0]:,} rows, {result.shape[1]} columns")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    return result


def build_majors_panel():
    """Top coins with volatility measures merged."""
    top_tickers = ["BTC", "ETH", "SOL", "ADA", "DOGE", "XRP", "LTC", "LINK", "DOT", "AVAX"]
    cm_dir = os.path.join(RAW_DIR, "coinmetrics_data")

    frames = []
    for ticker in top_tickers:
        vol_path = os.path.join(cm_dir, ticker, "volatility", "volatility_measures.parquet")
        all_path = os.path.join(cm_dir, ticker, "all_measures", "all_measures.parquet")

        if not os.path.exists(all_path):
            continue

        df = pd.read_parquet(all_path)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

        # Also get price from raw_price_daily
        price_file = os.path.join(RAW_DIR, "raw_price_daily", f"{ticker.lower()}-usd-max.csv")
        if os.path.exists(price_file):
            pdf = pd.read_csv(price_file)
            pdf = pdf.rename(columns={"snapped_at": "date"})
            pdf["date"] = pd.to_datetime(pdf["date"], utc=True).dt.tz_localize(None)
            pdf = pdf.rename(columns={"price": "price_usd", "total_volume": "volume_usd"})
            df = df.merge(pdf[["date", "price_usd", "volume_usd", "market_cap"]], on="date", how="left")

        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Compute returns
    panel["log_price"] = np.log(panel["price_usd"].clip(lower=1e-10))
    panel["log_return"] = panel.groupby("ticker")["log_price"].diff()

    print(f"Majors panel: {panel.shape[0]:,} rows, {panel['ticker'].nunique()} coins")
    return panel


def build_halving_dataset(btc_onchain):
    """Annotate BTC data with halving event windows."""
    halving_dates = {
        "halving_1": pd.Timestamp("2012-11-28"),
        "halving_2": pd.Timestamp("2016-07-09"),
        "halving_3": pd.Timestamp("2020-05-11"),
        "halving_4": pd.Timestamp("2024-04-19"),
    }

    df = btc_onchain.copy()
    df["halving_event"] = ""
    df["days_from_halving"] = np.nan
    df["halving_window"] = False

    for name, hdate in halving_dates.items():
        mask = (df["date"] >= hdate - pd.Timedelta(days=180)) & (
            df["date"] <= hdate + pd.Timedelta(days=365)
        )
        df.loc[mask, "halving_event"] = name
        df.loc[mask, "days_from_halving"] = (df.loc[mask, "date"] - hdate).dt.days
        df.loc[mask, "halving_window"] = True

    # Post-halving dummy (1 if after any halving within 365 days)
    df["post_halving"] = 0
    for name, hdate in halving_dates.items():
        mask = (df["date"] >= hdate) & (df["date"] <= hdate + pd.Timedelta(days=365))
        df.loc[mask, "post_halving"] = 1

    print(f"Halving dataset: {df.shape[0]:,} rows")
    print(f"Observations in halving windows: {df['halving_window'].sum():,}")
    return df


def write_sidecar(name, df, description):
    """Write a metadata sidecar JSON."""
    meta = {
        "name": name,
        "description": description,
        "created": datetime.now().isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": [str(df["date"].min()), str(df["date"].max())] if "date" in df.columns else None,
    }
    meta_path = os.path.join(CURATED_DIR, f"{name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Sidecar: {meta_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("CURATING CRYPTOCURRENCY DATA")
    print("=" * 60)

    # 1. Price panel
    print("\n--- Building price panel ---")
    panel = build_price_panel()
    out = os.path.join(CURATED_DIR, "crypto_panel_v1.parquet")
    panel.to_parquet(out, engine="pyarrow", compression="snappy", index=True)
    write_sidecar("crypto_panel_v1", panel, "Daily price/return panel for 400+ cryptocurrencies")
    print(f"  Saved: {out}")

    # 2. BTC on-chain
    print("\n--- Building BTC on-chain dataset ---")
    btc = build_btc_onchain()
    out = os.path.join(CURATED_DIR, "crypto_btc_onchain_v1.parquet")
    btc.to_parquet(out, engine="pyarrow", compression="snappy", index=True)
    write_sidecar("crypto_btc_onchain_v1", btc, "BTC daily on-chain fundamentals from CoinMetrics")
    print(f"  Saved: {out}")

    # 3. Majors panel
    print("\n--- Building majors panel ---")
    majors = build_majors_panel()
    out = os.path.join(CURATED_DIR, "crypto_majors_v1.parquet")
    majors.to_parquet(out, engine="pyarrow", compression="snappy", index=True)
    write_sidecar("crypto_majors_v1", majors, "Top 10 crypto coins with volatility/cross-sectional measures")
    print(f"  Saved: {out}")

    # 4. Halving dataset
    print("\n--- Building halving event dataset ---")
    halving = build_halving_dataset(btc)
    out = os.path.join(CURATED_DIR, "crypto_btc_halving_v1.parquet")
    halving.to_parquet(out, engine="pyarrow", compression="snappy", index=True)
    write_sidecar("crypto_btc_halving_v1", halving, "BTC data with halving event windows annotated")
    print(f"  Saved: {out}")

    print("\n" + "=" * 60)
    print("CURATION COMPLETE")
    print("=" * 60)
