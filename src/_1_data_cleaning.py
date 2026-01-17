#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 15:42:25 2025

@author: Dylan Bretz Jr.

Cyclistic Bike-Share Analysis Case Study Python Script (Part 1 of 3)
- Combine, clean, and transform data

Description:
    Combines monthly CSV files, cleans data by removing outliers
    and inconsistencies, and transforms data for analysis.

Inputs:
    - Raw CSV files located in '../data/raw/csv-files/'

Outputs:
    - '../data/processed/divvy-combined-cleaned.csv'
    - Static data visualizations located in '../viz/png/'

Dependencies:
    - config.py module for standardized file paths

Usage:
    Run this script as the first part of the Cyclistic Bike-Share
    Analysis Case Study pipeline.

Sections:
    0. CONFIG
    1. LOAD AND COMBINE RAW DATA
    2. INITIAL CLEANING AND EDA
    3. TRANSFORM DATA
    4. VERIFICATION AND ADDITIONAL CLEANING
    5. EXPORT CLEANED DATA
"""

# =============================================================================
#%% 0. CONFIG
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import config

# Set style
plt.style.use(config.STYLE_PATH)

# ---- GLOBAL CONSTANTS ----

TIMEFRAME_START = pd.Timestamp('2024-11-01')

# Daylight Savings Time (DST) ranges
DST_FALL_START = pd.Timestamp('2024-11-03 01:00:00')
DST_FALL_END = pd.Timestamp('2024-11-03 02:00:00')
DST_SPRING_START = pd.Timestamp('2025-03-09 02:00:00')
DST_SPRING_END = pd.Timestamp('2025-03-09 03:00:00')

# ---- UTILITY FUNCTIONS ----

def apply_filter(df_in, original_count, condition, description):
    """
    Filters dataframe based on condition, prints drop statistics,
    and returns filtered dataframe.
    """

    pre_filter = len(df_in)
    filtered_data = df_in[condition].copy()
    post_filter = len(filtered_data)

    step_drop = pre_filter - post_filter
    step_pct = (step_drop / pre_filter) * 100 if pre_filter else 0.0

    total_drop = original_count - post_filter
    total_pct = (total_drop / original_count) * 100 if original_count else 0.0

    print(f'\n---- {description} ----')
    print(
        'Rows dropped this step:',
        f'{pre_filter:,} - {post_filter:,} = {step_drop:,}',
        f'({step_pct:.2f}%)'
    )
    print(
        'Overall rows dropped:  ',
        f'{original_count:,} - {post_filter:,} = {total_drop:,}',
        f'({total_pct:.2f}%)'
    )

    return filtered_data

# =============================================================================
#%% 1. LOAD AND COMBINE RAW DATA
# =============================================================================

print('\nLoading and combining data...')

raw_csv = sorted(config.RAW_PATH.glob('*.csv'))
if not raw_csv:
    raise FileNotFoundError(f'No CSV files found in {config.RAW_PATH}')

df = pd.concat([pd.read_csv(f) for f in raw_csv], ignore_index=True)

print('Data loaded and combined.')

original_row_count = len(df)
print(f'\nTotal records: {original_row_count:,}')

print('\nFirst five rows:')
print(df.head())

# =============================================================================
#%% 2. INITIAL CLEANING AND EDA
# =============================================================================
# Steps:
# - Check for duplicate rides
# - Drop unnecessary columns
# - Convert incorrect datatypes
# - Manage null values
# - Drop rows containing rides starting before timeframe

# ---- Check for duplicate rides ----

print('\nChecking for duplicate rows/rides...')
print('Duplicate row count:', df.duplicated().sum())
print('Duplicate ride IDs:', df['ride_id'].duplicated().sum())

# ---- Drop unnecessary columns ----

print('\nColumns and datatypes:')
df.info()

print('\nDropping unnecessary columns...')

df = df.drop(columns=['start_station_id', 'end_station_id'])

print("Dropped 'start_station_id' and 'end_station_id' columns.")

# ---- Convert incorrect datatypes ----

print('\nConverting incorrect datatypes...')

df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])

print("Converted 'started_at' and 'ended_at' columns to datetime.")

print('\nColumns and datatypes:')
df.info()

# ---- Manage null values ----

print('\nCounting null values...')
print(f'Total null values: {df.isna().sum().sum():,}')

print('\nNull count per column:')
print(df.isna().sum())

print(
    '\nTotal rows missing both end coordinates:',
    f'{df[["end_lat", "end_lng"]].isna().all(axis=1).sum():,}'
)

df = apply_filter(
    df,
    original_row_count,
    df['end_lat'].notna() & df['end_lng'].notna(),
    'Drop Rows Missing End Coordinates'
)

print(
    '\nTotal rows missing both end coordinates:',
    f'{df[["end_lat", "end_lng"]].isna().all(axis=1).sum():,}'
)

print('\nFilling missing station names with placeholder...')

df['start_station_name'] = (
    df['start_station_name'].fillna('no_station_recorded')
)
df['end_station_name'] = (
    df['end_station_name'].fillna('no_station_recorded')
)

print('Filled missing station names with placeholder.')

print(f'\nTotal null values: {df.isna().sum().sum():,}')

# ---- Drop rows containing rides starting before timeframe ----

print('\nInspecting timestamp range...')
print(
    'Start date ranges from',
    f'{df["started_at"].min()} to {df["started_at"].max()}'
)
print(
    'End date ranges from',
    f'{df["ended_at"].min()} to {df["ended_at"].max()}'
)

print(
    f'\nTotal rides starting before {TIMEFRAME_START}:',
    len(df[df['started_at'] < TIMEFRAME_START])
)

df = apply_filter(
    df,
    original_row_count,
    df['started_at'] >= TIMEFRAME_START,
    'Drop Rides Starting Before Timeframe'
)

print(
    f'\nTotal rides starting before {TIMEFRAME_START}:',
    len(df[df['started_at'] < TIMEFRAME_START])
)

# =============================================================================
#%% 3. TRANSFORM DATA
# =============================================================================

print('\nExtracting start date, month, week, day of week, and hour...')

df['start_date'] = df['started_at'].dt.strftime('%Y-%m-%d')
df['start_month'] = df['started_at'].dt.strftime('%Y-%m')
df['start_week'] = df['started_at'].dt.strftime('%Y-w%W')
df['start_weekday'] = df['started_at'].dt.strftime('%w').astype(int)
df['start_hour'] = df['started_at'].dt.hour

print('\nCalculating ride duration in minutes...')

df['ride_duration_min'] = (
    (df['ended_at'] - df['started_at'])
    .dt.total_seconds() / 60
).round(2)

print('\nColumns and datatypes:')
df.info()

# =============================================================================
#%% 4. VERIFICATION AND ADDITIONAL CLEANING
# =============================================================================
# Steps:
# - Inspect negative-duration rides
# - Drop rows affected by DST time-shift
# - Inspect ride duration distribution
# - Visualize ride duration distribution to identify outliers
# - Drop rows containing rides 1 minute or less and 1 day or more

# ---- Inspect negative-duration rides ----

print('\nCounting negative-duration rides...')

print('Total negative-duration rides:',
      f'{len(df[df["ride_duration_min"] < 0]):,}'
)

print('\nFirst five rows containing negative-duration rides:')
print(
    df[df['ride_duration_min'] < 0]
    .sort_values('started_at')
    [['ride_duration_min', 'started_at', 'ended_at']]
    .head()
)

print('\nLast five rows containing negative-duration rides:')
print(
    df[df['ride_duration_min'] < 0]
    .sort_values('started_at')
    [['ride_duration_min', 'started_at', 'ended_at']]
    .tail()
)

# ---- Drop rows affected by DST time-shift ----

print('\nDropping rows affected by DST time-shift...')

fall_dst_mask = (
    (df['started_at'] < DST_FALL_END) &
    (df['ended_at'] >= DST_FALL_START)
)
spring_dst_mask = (
    (df['started_at'] < DST_SPRING_START) &
    (df['ended_at'] >= DST_SPRING_END)
)

combined_dst_mask = fall_dst_mask | spring_dst_mask

print(
    '\nTotal rows affected by fall time-shift:',
    len(df[fall_dst_mask])
)
print(
    'Total rows affected by spring time-shift:',
    len(df[spring_dst_mask])
)
print(
    'Total rows affected by both fall and spring time-shift:',
    len(df[combined_dst_mask])
)

df = apply_filter(
    df,
    original_row_count,
    ~combined_dst_mask,
    'Drop Rows Affected by DST'
)

print('\nTotal negative-duration rides:',
      len(df[df['ride_duration_min'] < 0]))

del fall_dst_mask, spring_dst_mask, combined_dst_mask

# ---- Inspect ride duration distribution ----

print('\nInspecting ride duration distribution...')

print('Minimum ride duration:',
      df['ride_duration_min'].min(),
      'minutes')
print('Maximum ride duration:',
      df['ride_duration_min'].max(),
      'minutes')

# ---- Visualize ride duration distribution to identify outliers ----

print('\nVisualizing ride duration distribution...')

# Ensure visualization output directory exists
viz_png = config.VIZ_PATH / 'png'
viz_png.mkdir(parents=True, exist_ok=True)

# 1. Full Distribution
fig, ax = plt.subplots(figsize=(10, 8))

n, bins, patches = ax.hist(
    df['ride_duration_min'] / 60,
    bins=np.arange(0, (df['ride_duration_min'].max() / 60) + 1, 1),
    color='lightslategray',
    edgecolor='black',
    linewidth=1.5,
    zorder=2
)

# Highlight tallest bar
for count, patch in zip(n, patches):
    if count == n.max():
        patch.set_facecolor('limegreen')

ax.set_title('Full Distribution of Ride Duration')
ax.set_xlabel('Ride Duration (hours)')
ax.set_ylabel('Frequency (in millions)')

ax.set_xlim(0, (df['ride_duration_min'].max() // 60) + 1)
ax.set_xticks(
    np.arange(0, (df['ride_duration_min'].max() // 60) + 5, 5)
)
ax.set_xticks(
    np.arange(0, (df['ride_duration_min'].max() // 60) + 1, 1),
    minor=True
)

ax.set_yticks(
    np.arange(0, ax.get_ylim()[1], 500_000),
    minor=True
)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f'{x/1_000_000:.0f}M')
)

ax.grid(which='minor', linestyle='-', alpha=0.2, zorder=0)

# Annotation: Start spike
ax.annotate('Most rides last 1 hour or less',
            xy=(1, 5_200_000),
            xytext=(3.15, 4_200_000),
            arrowprops={
                'arrowstyle': '->',
                'color': 'black',
                'linewidth': 1.5,
                'connectionstyle': 'arc3,rad=.2'
            },
            fontsize=18,
            fontweight='bold'
            )

# Annotation: Sample size
ax.text(
    0.99, 0.95,
    f'n = {len(df["ride_duration_min"]):,} rides',
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=16, color='dimgray'
)

plt.savefig(
    viz_png / 'fig_B1_eda_ride_duration_distribution_full.png',
    bbox_inches='tight',
    dpi=300
)
plt.show()

# 2. Log Scale Distribution
fig, ax = plt.subplots(figsize=(10, 8))

n, bins, patches = ax.hist(
    df['ride_duration_min'] / 60,
    bins=np.arange(0, (df['ride_duration_min'].max() / 60) + 1, 1),
    color='lightslategray',
    edgecolor='black',
    linewidth=1.5,
    zorder=2
)

# Highlight tallest bar
for count, patch in zip(n, patches):
    if count < 300:
        patch.set_facecolor('lightskyblue')

ax.set_title('Full Distribution of Ride Duration\n(Log Scale)')
ax.set_xlabel('Ride Duration (hours)')
ax.set_ylabel('Frequency (log scale)')

ax.set_xlim(0, (df['ride_duration_min'].max() // 60) + 1)
ax.set_xticks(
    np.arange(0, (df['ride_duration_min'].max() // 60) + 5, 5)
)
ax.set_xticks(
    np.arange(0, (df['ride_duration_min'].max() // 60) + 1, 1),
    minor=True
)

ax.set_yscale('log')

ax.grid(which='minor', linestyle='-', alpha=0.2, zorder=0)

# Annotation: Threshold line
ax.axhline(
    y=300,
    color='black',
    linestyle='--',
    linewidth=1.5,
    zorder=3,
    marker='none'
)
ax.text(
    x=24.5, y=350,
    s='Freq = 300',
    color='black',
    fontsize=14,
    fontweight='bold',
    ha='right'
)

# Annotation: Long tail
tail_props = {
    'text': 'Long right tail of lengthy\nrides (frequency < 300)',
    'xytext': (10.5, 1500),
    'fontsize': 18,
    'fontweight': 'bold'
}
ax.annotate(
    xy=(10.5, 200),
    arrowprops={
        'arrowstyle': '->',
        'color': 'black',
        'linewidth': 1.5,
        'connectionstyle': 'arc3,rad=.2'
    },
    **tail_props
)
ax.annotate(
    xy=(20.5, 200),
    arrowprops={
        'arrowstyle': '->',
        'color': 'black',
        'linewidth': 1.5,
        'connectionstyle': 'arc3,rad=-.2'
    },
    **tail_props
)

# Annotation: Sample size
ax.text(
    0.99, 0.95,
    f'n = {len(df["ride_duration_min"]):,} rides',
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=16, color='dimgray'
)

plt.savefig(
    viz_png / 'fig_B2_eda_ride_duration_distribution_log.png',
    bbox_inches='tight',
    dpi=300
)
plt.show()

# 3. Distribution Under 1 Hour
fig, ax = plt.subplots(figsize=(10, 8))

n, bins, patches = ax.hist(
    df[df['ride_duration_min'] <= 60]['ride_duration_min'],
    bins=np.arange(0, 61, 1),
    color='lightslategray',
    edgecolor='black',
    linewidth=1.5,
    zorder=2
)

patches[0].set_facecolor('salmon')

ax.set_title('Distribution of Ride Duration\nUnder 1 Hour')
ax.set_xlabel('Ride Duration (minutes)')
ax.set_ylabel('Frequency (in thousands)')

ax.set_xlim(0, 60)

ax.set_xticks(np.arange(0, 70, 10))
ax.set_xticks(np.arange(0, 61, 1), minor=True)

ax.set_yticks(np.arange(0, ax.get_ylim()[1], 100_000))
ax.set_yticks(np.arange(0, ax.get_ylim()[1], 50_000), minor=True)

ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f'{x/1000:.0f}K')
)

ax.grid(which='minor', linestyle='-', alpha=0.2, zorder=0)

# Annotation: Start spike
ax.annotate('Unusually high count of\nrides under 1 minute',
            xy=(1, 130_000),
            xytext=(21, 165_000),
            arrowprops={
                'arrowstyle': '->',
                'color': 'black',
                'linewidth': 1.5,
                'connectionstyle': 'arc3,rad=.2'
            },
            fontsize=18,
            fontweight='bold'
)

# Annotation: Sample size
ax.text(
    0.99, 0.95,
    f'n = {len(df[df["ride_duration_min"] <= 60]):,} rides',
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=16, color='dimgray'
)

plt.savefig(
    viz_png / 'fig_B3_eda_ride_duration_distribution_under_1hr.png',
    bbox_inches='tight',
    dpi=300
)
plt.show()

# 4. Distribution Under 5 Minutes
fig, ax = plt.subplots(figsize=(10, 8))

n, bins, patches = ax.hist(
    df[df['ride_duration_min'] <= 5]['ride_duration_min'],
    bins=np.arange(0, 5.25, 0.25),
    color='lightslategray',
    edgecolor='black',
    linewidth=1.5,
    zorder=2
)

# Highlight bars under 30 seconds
for patch in patches:
    if patch.get_x() < 0.5:
        patch.set_facecolor('salmon')

ax.set_title('Distribution of Ride Duration\nUnder 5 Minutes')
ax.set_xlabel('Ride Duration (minutes)')
ax.set_ylabel('Frequency (in thousands)')

ax.set_xlim(0, 5)

ax.set_xticks(np.arange(0, 5.5, 0.5))
ax.set_xticks(np.arange(0, 5.25, 0.25), minor=True)

ax.set_yticks(np.arange(0, ax.get_ylim()[1], 10_000), minor=True)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f'{x/1000:.0f}K')
)

ax.grid(which='minor', linestyle='-', alpha=0.2, zorder=0)

# Annotation: Start spike
ax.annotate('Unusually high count of\nrides under 30 seconds',
            xy=(0.3, 52_000),
            xytext=(0.35, 70_500),
            arrowprops={
                'arrowstyle': '->',
                'color': 'black',
                'linewidth': 1.5,
                'connectionstyle': 'arc3,rad=.2'
            },
            fontsize=18,
            fontweight='bold'
            )

# Sample size annotation
ax.text(
    0.90, 0.96,
    f'n = {len(df[df["ride_duration_min"] <= 5]):,} rides',
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=16, color='dimgray'
)

plt.savefig(
    viz_png / 'fig_B4_eda_ride_duration_distribution_under_5min.png',
    bbox_inches='tight',
    dpi=300
)
plt.show()

# Close figures and free resources
plt.close('all')

# ---- Drop rows containing rides 1 minute or less and 1 day or more ----

print('\nDropping rides 1 min or less and 1 day or more...')

print(
    'Total rides lasting 1 minute or less:',
    f'{len(df[df["ride_duration_min"] < 1]):,}'
)
print(
    'Total rides lasting 1 day or more:',
    f'{len(df[df["ride_duration_min"] > 1440]):,}'
)

df = apply_filter(
    df,
    original_row_count,
    (df['ride_duration_min'] > 1) & (df['ride_duration_min'] < 1440),
    'Drop Rides 1 Min or Less and 1 Day or More'
)

print(
    '\nTotal rides lasting 1 minute or less:',
    f'{len(df[df["ride_duration_min"] < 1]):,}'
)
print(
    'Total rides lasting 1 day or more:',
    f'{len(df[df["ride_duration_min"] > 1440]):,}'
)

print('\nMinimum ride duration:',
      df['ride_duration_min'].min(),
      'minutes')
print('Maximum ride duration:',
      df['ride_duration_min'].max(),
      'minutes')

# =============================================================================
#%% 5. EXPORT CLEANED DATA
# =============================================================================

print('\nExporting cleaned data...')

df = df.reset_index(drop=True)

# Ensure cleaned CSV output directory exists
config.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(config.CLEAN_PATH, index=False)
print(f'Cleaned data exported to: {config.CLEAN_PATH}')

print('\nValidating data export...')
clean_df = pd.read_csv(
    config.CLEAN_PATH, parse_dates=['started_at', 'ended_at']
)

try:
    pd.testing.assert_frame_equal(
        df, clean_df, check_dtype=False, check_exact=False
    )
    print('SUCCESS: Dataframes match.')
    del clean_df
except AssertionError as e:
    print('WARNING: Dataframes do not match:')
    print(e)
