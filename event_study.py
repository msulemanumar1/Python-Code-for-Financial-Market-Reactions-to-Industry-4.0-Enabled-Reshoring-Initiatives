# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 13:55:54 2025

@author: shurl
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------------
# CONFIGURATION
# ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "outputs")
plot_dir = os.path.join(output_dir, "plots")
excel_dir = os.path.join(output_dir, "excel_files")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(excel_dir, exist_ok=True)

company_events = {
    "AAPL": ["2012-12-06"],    # Apple
    "TXN": ["2013-11-22"],     # Texas Instruments
    "VYX": ["2009-06-02"],     # NCR
    "MU": ["2018-08-29"],      # Micron Technology
    "GE": ["2012-03-01"],      # General Electric
    "GNTX": ["2017-05-30"],    # Gentex
    "HPQ": ["2017-09-07"],     # HP
    "INTC": ["2009-02-10"],    # Intel
    "TSLA": ["2014-02-26"],    # Tesla
    "SWK": ["2017-03-09"],     # Stanley Black & Decker
}

industry_map = {
    "Technology": ["AAPL", "HPQ", "INTC", "TXN", "MU", "VYX"],
    "Industrial": ["GE", "GNTX", "SWK", "TSLA"]
}

market_ticker = "^GSPC"  # S&P 500
estimation_window = 250
estimation_gap = 30
event_window = 5

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def read_prepare_data_yf(ticker, prefix, start="2007-01-01", end="2024-12-31"):
    """Download and prepare data from Yahoo Finance"""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [col.replace(f" {ticker}", "").strip() for col in df.columns]
    
    # Select price column
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise KeyError(f"No price column found for ticker {ticker}. Columns are: {df.columns}")
    
    # Prepare columns
    cols = [price_col]
    if "Volume" in df.columns:
        cols.append("Volume")
    
    df = df[cols].copy()
    df.rename(columns={price_col: f"{prefix}_Price"}, inplace=True)
    
    if "Volume" in df.columns:
        df.rename(columns={"Volume": f"{prefix}_Volume"}, inplace=True)
    
    # Calculate returns and volatility
    df[f"{prefix}_Return"] = df[f"{prefix}_Price"].pct_change()
    
    if f"{prefix}_Return" in df.columns:
        df[f"{prefix}_Volatility"] = df[f"{prefix}_Return"].rolling(window=5, min_periods=1).std()
    
    return df

def merge_market_firm(market_df, firm_df):
    """Merge market and firm data"""
    if firm_df.empty:
        return pd.DataFrame()
    
    df = pd.DataFrame(index=market_df.index.union(firm_df.index)).sort_index()
    
    # Add market data
    for col in ["M_Price", "M_Return"]:
        if col in market_df.columns:
            df[col] = market_df[col]
    
    # Add firm data
    for col in ["F_Price", "F_Return", "F_Volume", "F_Volatility"]:
        if col in firm_df.columns:
            df[col] = firm_df[col]
    
    return df

def estimate_market_model(df, event_date, est_window=250, gap=30):
    """Estimate market model parameters"""
    event_date = pd.to_datetime(event_date)
    all_dates = df.index
    
    if event_date not in all_dates:
        if (df.index <= event_date).sum() == 0:
            raise ValueError("No trading days before event date")
        event_date = all_dates[all_dates <= event_date][-1]
    
    event_loc = all_dates.get_loc(event_date)
    est_end = event_loc - gap - 1
    est_start = est_end - est_window + 1
    
    if est_start < 0:
        raise ValueError("Estimation window too big for available data.")
    
    est_index = all_dates[est_start:est_end + 1]
    est_df = df.loc[est_index].dropna(subset=["M_Return", "F_Return"])
    
    if est_df.empty:
        raise ValueError("No data in estimation window.")
    
    X = sm.add_constant(est_df["M_Return"].values)
    y = est_df["F_Return"].values
    
    model = OLS(y, X).fit()
    alpha, beta = model.params[0], model.params[1]
    resid_std = np.std(model.resid, ddof=2)
    
    return dict(alpha=alpha, beta=beta, resid_std=resid_std, model=model, est_index=est_index)

def compute_abnormal_returns(df, alpha, beta, event_date, w=5):
    """Compute abnormal returns around event date"""
    event_date = pd.to_datetime(event_date)
    all_dates = df.index
    
    if event_date not in all_dates:
        event_date = all_dates[all_dates <= event_date][-1]
    
    loc = all_dates.get_loc(event_date)
    start = max(0, loc - w)
    end = min(len(all_dates) - 1, loc + w)
    
    window_dates = all_dates[start:end + 1]
    sub = df.loc[window_dates].dropna(subset=["M_Return", "F_Return"])
    
    if sub.empty:
        return pd.DataFrame()
    
    sub = sub.copy()
    sub["Expected"] = alpha + beta * sub["M_Return"]
    sub["AR"] = sub["F_Return"] - sub["Expected"]
    sub["CAR"] = sub["AR"].cumsum()
    sub["RelDay"] = range(-w, len(sub)-w)
    
    return sub

def ar_significance(ar_series, resid_std):
    """Calculate significance statistics for abnormal returns"""
    if ar_series.empty:
        return dict(t_ar=None, car=None, t_car=None)
    
    T = len(ar_series)
    car = ar_series.cumsum().iloc[-1]
    t_car = car / (resid_std * np.sqrt(T)) if resid_std > 0 else None
    
    return dict(t_ar=ar_series / resid_std if resid_std > 0 else None, car=car, t_car=t_car)

# ---------------------------
# MAIN ANALYSIS
# ---------------------------

print("[INFO] Starting comprehensive event study analysis...")
print("[INFO] Downloading market data...")

market = read_prepare_data_yf(market_ticker, "M")

if market.empty:
    raise RuntimeError(f"Market data for {market_ticker} could not be loaded.")

all_results = []
agg_rows = []
aligned_ars = []
aligned_volumes = []
aligned_volatilities = []
industry_ar_rows = []

car_window = range(-event_window, event_window + 1)
rel_days = list(car_window)

# Process each company
for ticker, dates in company_events.items():
    print(f"\n[PROCESSING] {ticker} ...")
    
    firm = read_prepare_data_yf(ticker, "F")
    if firm.empty:
        print(f"[SKIP] No data for {ticker}")
        continue
    
    df = merge_market_firm(market, firm)
    if df.empty:
        print(f"[SKIP] Unable to merge data for {ticker}")
        continue
    
    # Save merged data
    merged_file = os.path.join(output_dir, f"merged_{ticker}.csv")
    df.to_csv(merged_file)
    
    # PLOT 1: Normalized price series
    plt.figure(figsize=(12, 6))
    m_norm = (df["M_Price"] / df["M_Price"].ffill().iloc[0]) * 100
    f_norm = (df["F_Price"] / df["F_Price"].ffill().iloc[0]) * 100
    
    plt.plot(df.index, m_norm, label="S&P 500 (norm)", linewidth=1.2)
    plt.plot(df.index, f_norm, label=f"{ticker} (norm)", linewidth=1.2)
    
    for ed in dates:
        if pd.to_datetime(ed) not in df.index:
            if (df.index <= pd.to_datetime(ed)).sum() == 0:
                continue
            ed_dt = df.index[df.index <= pd.to_datetime(ed)][-1]
        else:
            ed_dt = pd.to_datetime(ed)
        
        plt.axvline(ed_dt, linestyle='--', alpha=0.6)
        plt.text(ed_dt, plt.ylim()[1]*0.95, ed, rotation=90, va="top", fontsize=8)
    
    plt.legend()
    plt.title(f"Normalized Price Series — {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Indexed Price (100 = start)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"price_series_{ticker}.png"))
    plt.close()
    
    # Event study for each event date
    for ev in dates:
        try:
            mm = estimate_market_model(df, ev, est_window=estimation_window, gap=estimation_gap)
        except Exception as e:
            print(f"Error estimating market model for {ticker} {ev}: {e}")
            continue
        
        ar_df = compute_abnormal_returns(df, mm["alpha"], mm["beta"], ev, w=event_window)
        if ar_df.empty:
            print(f"[SKIP] No AR data for {ticker} {ev}")
            continue
        
        stats_dict = ar_significance(ar_df["AR"], mm["resid_std"])
        
        # Save AR data
        ar_path = os.path.join(output_dir, f"AR_{ticker}_{ev}.csv")
        ar_df.to_csv(ar_path)
        
        # PLOT 2: Individual AR & CAR
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(ar_df.index, ar_df["AR"], alpha=0.6, label='Abnormal Return')
        ax1.set_ylabel("Abnormal Return (AR)")
        
        ax2 = ax1.twinx()
        ax2.plot(ar_df.index, ar_df["CAR"], color="black", marker="o", linewidth=1.5, label='CAR')
        ax2.set_ylabel("Cumulative Abnormal Return (CAR)", color='black')
        
        plt.axvline(x=pd.to_datetime(ev), color='red', linestyle="--", alpha=0.6)
        plt.title(f"{ticker} — AR & CAR around {ev}")
        fig.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"AR_CAR_{ticker}_{ev}.png"))
        plt.close(fig)
        
        # PLOT 3: Individual Volume (if available)
        if "F_Volume" in ar_df.columns:
            plt.figure(figsize=(10,4))
            plt.plot(ar_df.index, ar_df["F_Volume"], marker='o', label="Volume")
            plt.axvline(x=pd.to_datetime(ev), color='red', linestyle="--", alpha=0.6, label="Event")
            plt.legend()
            plt.title(f"{ticker} — Trading Volume around {ev}")
            plt.xlabel("Date")
            plt.ylabel("Volume")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"VOLUME_{ticker}_{ev}.png"))
            plt.close()
        
        # PLOT 4: Individual Volatility (if available)
        if "F_Volatility" in ar_df.columns:
            plt.figure(figsize=(10,4))
            plt.plot(ar_df.index, ar_df["F_Volatility"], marker='o', color='orange', label="Volatility")
            plt.axvline(x=pd.to_datetime(ev), color='red', linestyle="--", alpha=0.6, label="Event")
            plt.legend()
            plt.title(f"{ticker} — Volatility around {ev}")
            plt.xlabel("Date")
            plt.ylabel("Rolling 5D Volatility")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"VOLATILITY_{ticker}_{ev}.png"))
            plt.close()
        
        # Store for aggregation
        ar_df["Ticker"] = ticker
        agg_rows.append(ar_df)
        
        # Prepare data for alignment
        event_dt = pd.to_datetime(ev)
        ar_df_copy = ar_df.copy()
        ar_df_copy['rel_day'] = (ar_df_copy.index - event_dt).days
        ar_df_copy = ar_df_copy.set_index('rel_day', drop=True)
        ar_df_copy = ar_df_copy.loc[ar_df_copy.index.isin(rel_days)]
        
        aligned_ars.append(ar_df_copy.reindex(rel_days)['AR'])
        
        # Volume alignment
        if 'F_Volume' in ar_df_copy.columns:
            aligned_volumes.append(ar_df_copy.reindex(rel_days)['F_Volume'])
        else:
            aligned_volumes.append(pd.Series(np.nan, index=rel_days))
        
        # Volatility alignment
        if 'F_Volatility' in ar_df_copy.columns:
            aligned_volatilities.append(ar_df_copy.reindex(rel_days)['F_Volatility'])
        else:
            aligned_volatilities.append(pd.Series(np.nan, index=rel_days))
        
        # Industry classification for aggregation
        industry = next((ind for ind, tickers in industry_map.items() if ticker in tickers), None)
        if industry:
            ar_df_industry = ar_df.copy()
            ar_df_industry["Industry"] = industry
            industry_ar_rows.append(ar_df_industry)
        
        # Store results
        all_results.append({
            "ticker": ticker,
            "event": ev,
            "alpha": mm["alpha"],
            "beta": mm["beta"],
            "resid_std": mm["resid_std"],
            "AR_mean": ar_df["AR"].mean(),
            "CAR": stats_dict["car"],
            "t_car": stats_dict["t_car"]
        })

# ---------------------------
# AGGREGATED ANALYSIS & PLOTS
# ---------------------------

print("\n[INFO] Creating aggregated analysis...")

# Save main results
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(os.path.join(output_dir, "event_study_all_companies.csv"), index=False)
summary_df.to_excel(os.path.join(excel_dir, "event_study_all_companies.xlsx"), index=False)

# PLOT 5: Aggregate Event Study
if len(agg_rows) > 0:
    combined_df = pd.concat(agg_rows, ignore_index=True)
    mean_ar = combined_df.groupby("RelDay")["AR"].mean()
    mean_car = combined_df.groupby("RelDay")["CAR"].mean()
    
    plt.figure(figsize=(12,6))
    plt.bar(mean_ar.index, mean_ar.values, alpha=0.6, label="Mean AR (all)")
    plt.plot(mean_car.index, mean_car.values, marker="o", color="red", label="Mean CAR (all)")
    plt.axvline(0, linestyle="--", alpha=0.6)
    plt.xlabel("Trading days relative to event")
    plt.ylabel("Abnormal Return / CAR")
    plt.title("Aggregate Event Study — All Companies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "aggregate_event_study.png"))
    plt.close()

# AAR and CAR Tables
if aligned_ars:
    ar_agg = pd.DataFrame(aligned_ars).T
    aar = ar_agg.mean(axis=1)
    aar_se = ar_agg.std(axis=1) / np.sqrt(ar_agg.shape[1])
    
    # AAR Table
    aar_table = pd.DataFrame({
        'Event Day': aar.index,
        'AAR': aar.values,
        'Std Err': aar_se.values
    })
    aar_table.to_csv(os.path.join(output_dir, "AAR_table.csv"), index=False)
    aar_table.to_excel(os.path.join(excel_dir, "AAR_table.xlsx"), index=False)
    
    # CAR Table
    car_individual = ar_agg.loc[rel_days].sum(axis=0)
    car_mean = car_individual.mean()
    car_se = car_individual.std() / np.sqrt(car_individual.size)
    
    car_table = pd.DataFrame({
        'Ticker': ar_agg.columns,
        'CAR [-5,5]': car_individual.values
    })
    car_table.loc[len(car_table)] = ['Average', car_mean]
    car_table.loc[len(car_table)] = ['Std Err', car_se]
    car_table.to_csv(os.path.join(output_dir, "CAR_table.csv"), index=False)
    car_table.to_excel(os.path.join(excel_dir, "CAR_table.xlsx"), index=False)

# PLOT 6: Aggregated Volume
if aligned_volumes:
    volume_agg = pd.DataFrame(aligned_volumes).T
    volume_mean = volume_agg.mean(axis=1)
    volume_se = volume_agg.std(axis=1) / np.sqrt(volume_agg.count(axis=1))
    
    plt.figure(figsize=(10,5))
    plt.bar(volume_mean.index, volume_mean.values, yerr=volume_se.values, alpha=0.7, color='C0', capsize=3)
    plt.axvline(0, color='red', linestyle='--', label='Announcement Day')
    plt.title('Mean Trading Volume (Aggregated) Around Announcement')
    plt.xlabel('Days relative to announcement')
    plt.ylabel('Average Volume')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "AGG_VOLUME.png"))
    plt.close()
    
    # Save volume data
    volume_table = pd.DataFrame({
        'Event Day': volume_mean.index,
        'Mean Volume': volume_mean.values,
        'Std Err': volume_se.values
    })
    volume_table.to_csv(os.path.join(output_dir, "aggregated_volume.csv"), index=False)
    volume_table.to_excel(os.path.join(excel_dir, "aggregated_volume.xlsx"), index=False)

# PLOT 7: Aggregated Volatility
if aligned_volatilities:
    volatility_agg = pd.DataFrame(aligned_volatilities).T
    volatility_mean = volatility_agg.mean(axis=1)
    volatility_se = volatility_agg.std(axis=1) / np.sqrt(volatility_agg.count(axis=1))
    
    plt.figure(figsize=(10,5))
    plt.plot(volatility_mean.index, volatility_mean.values, marker='o', color='orange', label='Mean Volatility')
    plt.fill_between(volatility_mean.index,
                     volatility_mean - volatility_se,
                     volatility_mean + volatility_se, alpha=0.2, color='orange')
    plt.axvline(0, color='red', linestyle='--', label='Announcement Day')
    plt.title('Mean Volatility (Aggregated) Around Announcement')
    plt.xlabel('Days relative to announcement')
    plt.ylabel('Average Rolling 5D Volatility')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "AGG_VOLATILITY.png"))
    plt.close()
    
    # Save volatility data
    volatility_table = pd.DataFrame({
        'Event Day': volatility_mean.index,
        'Mean Volatility': volatility_mean.values,
        'Std Err': volatility_se.values
    })
    volatility_table.to_csv(os.path.join(output_dir, "aggregated_volatility.csv"), index=False)
    volatility_table.to_excel(os.path.join(excel_dir, "aggregated_volatility.xlsx"), index=False)

# ---------------------------
# INDUSTRY-LEVEL ANALYSIS
# ---------------------------

print("\n[INFO] Creating industry-level analysis...")

if industry_ar_rows:
    industry_df = pd.concat(industry_ar_rows, ignore_index=True)
    industry_results = []
    
    # PLOTS 8-9: Industry-specific analysis
    for industry, group in industry_df.groupby("Industry"):
        mean_ar = group.groupby("RelDay")["AR"].mean()
        mean_abs_ar = group.groupby("RelDay")["AR"].apply(lambda s: s.abs().mean())
        mean_car = group.groupby("RelDay")["CAR"].mean()
        
        plt.figure(figsize=(11,6))
        ax1 = plt.gca()
        ax1.bar(mean_ar.index, mean_ar.values, alpha=0.6, label=f"{industry} Mean AR")
        ax1.plot(mean_abs_ar.index, mean_abs_ar.values, marker="o", linewidth=1.2, label=f"{industry} Mean |AR|")
        ax1.axvline(0, linestyle="--", alpha=0.6)
        ax1.set_xlabel("Trading days relative to event (0 = announcement day)")
        ax1.set_ylabel("Abnormal Return")
        
        ax2 = ax1.twinx()
        ax2.plot(mean_car.index, mean_car.values, marker="s", linewidth=1.2, color="black", label=f"{industry} Mean CAR")
        ax2.set_ylabel("Cumulative Abnormal Return")
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        plt.title(f"{industry} Abnormal Returns and CAR around Events")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{industry}_AR_CAR.png"))
        plt.close()
        
        # Store industry results
        industry_summary = pd.DataFrame({
            'Event Day': mean_ar.index,
            'Mean AR': mean_ar.values,
            'Mean |AR|': mean_abs_ar.values,
            'Mean CAR': mean_car.values
        })
        industry_summary.to_csv(os.path.join(output_dir, f"{industry}_analysis.csv"), index=False)
        industry_summary.to_excel(os.path.join(excel_dir, f"{industry}_analysis.xlsx"), index=False)
        
        industry_results.append({
            'Industry': industry,
            'Mean_CAR': mean_car.iloc[-1],
            'Max_AR': mean_ar.max(),
            'Min_AR': mean_ar.min(),
            'Companies': len(group['Ticker'].unique())
        })
    
    # Save industry comparison
    industry_comparison = pd.DataFrame(industry_results)
    industry_comparison.to_csv(os.path.join(output_dir, "industry_comparison.csv"), index=False)
    industry_comparison.to_excel(os.path.join(excel_dir, "industry_comparison.xlsx"), index=False)

# ---------------------------
# SUMMARY REPORT
# ---------------------------

print("\n[INFO] Generating summary report...")

summary_stats = {
    'Total Companies Analyzed': len(company_events),
    'Total Events': sum(len(dates) for dates in company_events.values()),
    'Technology Companies': len(industry_map["Technology"]),
    'Industrial Companies': len(industry_map["Industrial"]),
    'Estimation Window (days)': estimation_window,
    'Event Window (+/- days)': event_window,
    'Market Benchmark': market_ticker
}

summary_report = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
summary_report.to_csv(os.path.join(output_dir, "analysis_summary.csv"), index=False)
summary_report.to_excel(os.path.join(excel_dir, "analysis_summary.xlsx"), index=False)

print(f"\n[COMPLETED] Comprehensive event study analysis finished!")
print(f"Results saved in: {output_dir}")
print(f"Excel files saved in: {excel_dir}")
print(f"Plots saved in: {plot_dir}")

print(f"\nFiles generated:")
print(f"- CSV files: {len([f for f in os.listdir(output_dir) if f.endswith('.csv')])}")
print(f"- Excel files: {len([f for f in os.listdir(excel_dir) if f.endswith('.xlsx')])}")
print(f"- Plot files: {len([f for f in os.listdir(plot_dir) if f.endswith('.png')])}")

print(f"\nMain outputs:")
print(f"1. Individual company normalized price series plots")
print(f"2. Individual AR & CAR plots for each company")
print(f"3. Individual volume plots (where available)")
print(f"4. Individual volatility plots (where available)")
print(f"5. Aggregated event study plot")
print(f"6. Aggregated volume plot")
print(f"7. Aggregated volatility plot")
print(f"8. Technology industry AR & CAR plot")
print(f"9. Industrial industry AR & CAR plot")
print(f"10. Excel files with all tabular results")