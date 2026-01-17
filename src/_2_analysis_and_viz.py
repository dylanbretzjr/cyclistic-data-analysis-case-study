#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 15:44:09 2025

@author: Dylan Bretz Jr.

Cyclistic Bike-Share Analysis Case Study Python Script (Part 2 of 3)
- Analyze and visualize cleaned data

Description:
    Imports cleaned dataset, performs statistical analysis to
    compare rider behavior, and generates static visualizations
    to illustrate key findings.

Inputs:
    - '../data/processed/divvy-combined-cleaned.csv'

Outputs:
    - Static data visualizations located in '../viz/png/'

Dependencies:
    - config.py module for standardized file paths

Usage:
    Run this script as the second part of the Cyclistic Bike-Share
    Analysis Case Study pipeline.

Sections:
    0. CONFIG
    1. LOAD CLEANED DATA
    2. ANALYZE CLEANED DATA
    3. VISUALIZE DATA INSIGHTS
"""

# =============================================================================
#%% 0. CONFIG
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns

import config

# Apply style settings
plt.style.use(config.STYLE_PATH)

# Weekday labels used throughout plots
WEEKDAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

# ---- UTILITY FUNCTIONS ----

def get_counts_and_pct(df_in, group_cols):
    """
    Aggregates data by group_cols and calculates counts and percentages.

    If multiple columns are provided, percentages are calculated relative to
    first column provided.
    """
    # 1. Group and count
    summary = df_in.groupby(group_cols).size().reset_index(name='count')

    # 2. Calculate denominator
    if len(group_cols) > 1:
        denominator = summary.groupby(group_cols[0])['count'].transform('sum')
    else:
        denominator = summary['count'].sum()

    # 3. Calculate percentage
    summary['percentage'] = (summary['count'] / denominator) * 100

    return summary

def get_ranking(df_in, group_cols, member_type=None):
    """
    Filters data by member_type (optional), groups by columns, counts rows, 
    calculates percentage, and sorts descending.
    """
    # 1. Filter by member type if provided
    df_filtered = df_in
    if member_type:
        df_filtered = df_in[df_in['member_casual'] == member_type]

    # 2. Group and count
    summary = df_filtered.groupby(group_cols).size().reset_index(name='count')

    # 3. Calculate percentage (relative to this specific slice)
    summary['percentage'] = (summary['count'] / summary['count'].sum()) * 100

    # 4. Sort descending
    return summary.sort_values('count', ascending=False).reset_index(drop=True)

# =============================================================================
#%% 1. LOAD CLEANED DATA
# =============================================================================

clean_path = config.CLEAN_PATH
if not clean_path.exists():
    raise FileNotFoundError(f'Cleaned data not found at {config.CLEAN_PATH}')
df = pd.read_csv(clean_path, parse_dates=['started_at', 'ended_at'])

# Ensure visualization output directory exists
viz_png = config.VIZ_PATH / 'png'
viz_png.mkdir(parents=True, exist_ok=True)

# =============================================================================
#%% 2. ANALYZE CLEANED DATA
# =============================================================================
# Steps:
# - Calculate proportion of rides by membership status
# - Calculate trends and statistics
# - Find top stations, routes, and round-trips
# - Prep heat map data

# ---- Calculate proportion of rides by membership status ----

sum_overall = get_counts_and_pct(df, ['member_casual'])

sum_month = get_counts_and_pct(
    df, ['start_month', 'member_casual']
)

sum_week = get_counts_and_pct(
    df.assign(start_week=df['started_at'].dt.to_period('W').dt.start_time),
    ['start_week', 'member_casual']
)

sum_weekday = get_counts_and_pct(
    df, ['start_weekday', 'member_casual']
)

sum_weekend = get_counts_and_pct(
    df.assign(
        day_type=np.where(
            df['start_weekday'].isin([0, 6]), 'Weekend', 'Weekday'
            )
    ),
    ['day_type', 'member_casual']
)

sum_hour = get_counts_and_pct(
    df, ['start_hour', 'member_casual']
)

sum_biketype = get_counts_and_pct(
    df, ['rideable_type', 'member_casual']
)

sum_roundtrip = get_counts_and_pct(
    df.assign(
        is_roundtrip=df['start_station_name'] == df['end_station_name']
    ),
    ['is_roundtrip', 'member_casual']
)

# ---- Calculate trends and statistics ----

# Percent change in rides per month
sum_month['pct_change'] = (
    sum_month
    .groupby('member_casual')['count']
    .pct_change() * 100
)

# Ride duration statistics
stats_aggs = {
    'min': 'min',
    'max': 'max',
    'mean': 'mean',
    'median': 'median',
    'mode': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    'q1': lambda x: x.quantile(0.25),
    'q3': lambda x: x.quantile(0.75),
    'iqr': lambda x: x.quantile(0.75) - x.quantile(0.25)
}

duration_overall = (
    df.groupby('member_casual')['ride_duration_min']
    .agg(**stats_aggs).reset_index()
)

duration_month = (
    df.groupby(['start_month', 'member_casual'])['ride_duration_min']
    .agg(**stats_aggs).reset_index()
)

duration_weekday = (
    df.groupby(['start_weekday', 'member_casual'])['ride_duration_min']
    .agg(**stats_aggs).reset_index()
)

duration_biketype = (
    df.groupby(['rideable_type', 'member_casual'])['ride_duration_min']
    .agg(**stats_aggs).reset_index()
)

del stats_aggs

# Mode of start day of week
mode_weekday = (
    df.pivot_table(
        index='member_casual',
        values='start_weekday',
        aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        margins=True,
        margins_name='overall'
    )
    .rename(columns={'start_weekday': 'mode_weekday'})
    .reset_index()
)

# ---- Find top stations, routes, and round-trips ----

# Top stations
top_start_m = get_ranking(df, ['start_station_name'], 'member')
top_start_c = get_ranking(df, ['start_station_name'], 'casual')

top_end_m = get_ranking(df, ['end_station_name'], 'member')
top_end_c = get_ranking(df, ['end_station_name'], 'casual')

# Top routes
top_routes_m = get_ranking(
    df, ['start_station_name', 'end_station_name'], 'member'
)
top_routes_c = get_ranking(
    df, ['start_station_name', 'end_station_name'], 'casual'
)

# Top round-trips
is_roundtrip = df['start_station_name'] == df['end_station_name']

top_roundtrips_m = get_ranking(
    df[is_roundtrip], ['start_station_name'], 'member'
)
top_roundtrips_c = get_ranking(
    df[is_roundtrip], ['start_station_name'], 'casual'
)

# ---- Prep heat map data ----

density_day_hour_m = get_ranking(
    df, ['start_weekday', 'start_hour'], 'member'
)

density_day_hour_c = get_ranking(
    df, ['start_weekday', 'start_hour'], 'casual'
)

# =============================================================================
#%% 3. VISUALIZE DATA INSIGHTS
# =============================================================================
# Steps:
# - Overall rides
# - Seasonality
# - Day of week
# - Time of day
# - Ride duration
# - Top locations
# - Round trips
# - Additional visuals

#%% ---- Overall rides ----

#%% fig_1 Total rides by membership status

fig, ax = plt.subplots(figsize=(10, 8))
WIDTH = 0.6

bars_member = ax.bar(
    ['Member'],
    sum_overall.loc[
        sum_overall['member_casual'] == 'member', 'count'
    ],
    width=WIDTH,
    label='Member',
    zorder=2
)
bars_casual = ax.bar(
    ['Casual'],
    sum_overall.loc[
        sum_overall['member_casual'] == 'casual', 'count'
    ],
    width=WIDTH,
    label='Casual',
    zorder=2
)

ax.set_title('Total Rides by Membership Status')
ax.set_xlabel('Membership Status')
ax.set_ylabel('Total Rides (in millions)')

ymax = sum_overall['count'].max() * 1.2
ax.set_ylim(0, ymax)
ax.set_yticks(np.arange(0, ymax, 1_000_000))
ax.set_yticks(np.arange(0, ymax, 500_000), minor=True)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f'{x/1_000_000:.0f}M')
)

ax.grid(False, axis='x')
ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)

# Annotation: Bar counts and percentages
for bars, category in [(bars_member, 'member'), (bars_casual, 'casual')]:
    row = sum_overall.loc[
        sum_overall['member_casual'] == category
    ].iloc[0]
    ax.bar_label(
        bars,
        labels=[f'{row["count"]:,.0f}\n({row["percentage"]:.1f}%)'],
        padding=8,
        fontsize=16,
        fontweight='bold',
        color='black'
    )

# Annotation: Sample size
ax.text(
    0.99, 0.95,
    f'n = {sum_overall["count"].sum():,} rides',
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=16, color='dimgray'
)

plt.savefig(
    viz_png / 'fig_1_Total_rides.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% ---- Seasonality ----

#%% fig_2.1 Total monthly rides by membership status

pivot_monthly_counts = sum_month.pivot(
    index='start_month',
    columns='member_casual',
    values='count'
)

x = np.arange(len(pivot_monthly_counts.index))
WIDTH = 0.38

fig, ax = plt.subplots(figsize=(10, 8))

bars_member = ax.bar(
    x - WIDTH/2,
    pivot_monthly_counts['member'],
    width=WIDTH,
    label='Member',
    zorder=2
)
bars_casual = ax.bar(
    x + WIDTH/2,
    pivot_monthly_counts['casual'],
    width=WIDTH,
    label='Casual',
    zorder=2
)

ax.set_title('Total Monthly Rides\nby Membership Status')
ax.set_xlabel('Month')
ax.set_ylabel('Total Rides (in thousands)')

ax.set_xticks(x)
ax.set_xticklabels(
    pd.to_datetime(
        pivot_monthly_counts.index, format='%Y-%m'
    ).strftime('%b %Y'),
    rotation=45, ha='right'
)

ymax = pivot_monthly_counts.max().max() * 1.2
ax.set_ylim(0, ymax)
ax.set_yticks(np.arange(0, ymax, 100_000))
ax.set_yticks(np.arange(0, ymax, 50_000), minor=True)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f'{x/1000:.0f}K')
)

ax.grid(False, axis='x')
ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

# Annotation: Bar counts
ax.bar_label(
    bars_member, fmt=lambda x: f'{x/1000:.0f}K',
    rotation=90, padding=5, fontsize=12, fontweight='bold'
)
ax.bar_label(
    bars_casual, fmt=lambda x: f'{x/1000:.0f}K',
    rotation=90, padding=5, fontsize=12, fontweight='bold'
)

plt.savefig(
    viz_png / 'fig_2_1_Total_monthly_rides.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% fig_2.2 Proportion of rides per week by membership status

pivot_weekly_counts = (
    sum_week
        .pivot(index='start_week', columns='member_casual', values='count')
        .fillna(0)
        .sort_index()
)
weekly_total = pivot_weekly_counts.sum(axis=1)
weekly_prop = pivot_weekly_counts.div(weekly_total, axis=0)

member_prop = weekly_prop['member'].values
casual_prop = weekly_prop['casual'].values

x = np.arange(len(weekly_prop.index))

fig, ax = plt.subplots(figsize=(10, 8))

ax.stackplot(
    x,
    member_prop,
    casual_prop,
    labels=['Member', 'Casual'],
    zorder=2
)

ax.set_title('Weekly Proportion of Rides\nby Membership Status')
ax.set_xlabel('Week')
ax.set_ylabel('Proportion of Rides')

tick_frequency = max(1, len(x) // 12)
ax.set_xlim([0, len(x) - 1])
ax.set_xticks(x[::tick_frequency])
ax.set_xticklabels(
    pd.to_datetime(weekly_prop.index).strftime('%b %d %Y')[::tick_frequency],
    rotation=45,
    ha='right'
)

ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f'{val*100:.0f}%')
)

ax.legend()

plt.savefig(
    viz_png / 'fig_2_2_Proportion_of_rides_per_week.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% fig_2.3 Percent change in rides per month by membership status

pivot_monthly_pct_change = (
    sum_month
    .pivot(
        index='start_month',
        columns='member_casual',
        values='pct_change'
    )
    .sort_index()
)

x = np.arange(len(pivot_monthly_pct_change.index))

fig, ax = plt.subplots(figsize=(10, 8))

line_member = ax.plot(
    x,
    pivot_monthly_pct_change['member'],
    label='Member',
    zorder=3
)
line_casual = ax.plot(
    x,
    pivot_monthly_pct_change['casual'],
    label='Casual',
    zorder=3
)

ax.axhline(
    y=0,
    color='black',
    linestyle='--',
    linewidth=1.5,
    zorder=2,
    marker='none'
)

ax.set_title('Monthly Percent Change in Rides\nby Membership Status')
ax.set_xlabel('Month')
ax.set_ylabel('Percent Change (%)')

ax.set_xticks(x)
ax.set_xticklabels(
    pd.to_datetime(
        pivot_monthly_pct_change.index, format='%Y-%m'
    ).strftime('%b %Y'),
    rotation=45, ha='right'
)

ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(25))
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f'{val:.0f}%')
)

ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

plt.savefig(
    viz_png / 'fig_2_3_Percent_change_in_rides_per_month.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% --- Day of week ----

#%% fig_3.1 Total rides per day of week by membership status

pivot_weekday_counts = sum_weekday.pivot(
    index='start_weekday',
    columns='member_casual',
    values='count'
)

x = np.arange(len(pivot_weekday_counts.index))
WIDTH = 0.38

fig, ax = plt.subplots(figsize=(10, 8))

bars_member = ax.bar(
    x - WIDTH/2,
    pivot_weekday_counts['member'],
    width=WIDTH,
    label='Member',
    zorder=2
)
bars_casual = ax.bar(
    x + WIDTH/2,
    pivot_weekday_counts['casual'],
    width=WIDTH,
    label='Casual',
    zorder=2
)

ax.set_title('Total Daily Rides\nby Membership Status')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Total Rides (in thousands)')

ax.set_xticks(x)
ax.set_xticklabels(WEEKDAY_LABELS)

ymax = pivot_weekday_counts.max().max() * 1.2
ax.set_ylim(0, ymax)
ax.set_yticks(np.arange(0, ymax, 50_000), minor=True)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f'{x/1000:.0f}K')
)

ax.grid(False, axis='x')
ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

# Annotation: Bar counts
ax.bar_label(
    bars_member, fmt=lambda x: f'{x/1000:.0f}K',
    rotation=90, padding=5, fontsize=14, fontweight='bold'
)
ax.bar_label(
    bars_casual, fmt=lambda x: f'{x/1000:.0f}K',
    rotation=90, padding=5, fontsize=14, fontweight='bold'
)

plt.savefig(
    viz_png / 'fig_3_1_Total_rides_per_day_of_week.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% fig_A1 Proportion of rides per day of week by membership status

pivot_weekday_prop = (
    sum_weekday
    .pivot(index='start_weekday', columns='member_casual', values='percentage')
    .reindex(range(7), fill_value=0)  # ensure Sun–Sat present
    .sort_index()
)

member_prop = pivot_weekday_prop['member'].values / 100
casual_prop = pivot_weekday_prop['casual'].values / 100

x = np.arange(7)
WIDTH = 0.6

fig, ax = plt.subplots(figsize=(10, 8))

bars_member = ax.bar(
    x,
    member_prop,
    width=WIDTH,
    label='Member',
    zorder=2
)
bars_casual = ax.bar(
    x,
    casual_prop,
    width=WIDTH,
    bottom=member_prop,
    label='Casual',
    zorder=2
)

ax.set_title('Daily Proportion of Rides\nby Membership Status')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Proportion of Rides')

ax.set_xlim(-0.5, 6.5)
ax.set_xticks(x)
ax.set_xticklabels(WEEKDAY_LABELS)

ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f'{val*100:.0f}%')
)

ax.grid(False, axis='x')
ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

# Annotation: Bar percentages
for i, (m, c) in enumerate(zip(member_prop, casual_prop)):
    ax.text(x[i], m/2, f'{m:.0%}', ha='center', va='center',
            fontweight='bold', color='white', fontsize=14)
    ax.text(x[i], m + c/2, f'{c:.0%}', ha='center', va='center',
            fontweight='bold', color='black', fontsize=14)

plt.savefig(
    viz_png / 'fig_A1_Proportion_of_rides_per_day_of_week.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% fig_A2 Total weekday vs weekend rides by membership status

pivot_weekend_counts = (
    sum_weekend
    .pivot(index='day_type', columns='member_casual', values='count')
    .reindex(['Weekday', 'Weekend'])
)

x = np.arange(len(pivot_weekend_counts.index))
WIDTH = 0.38

fig, ax = plt.subplots(figsize=(10, 8))

bars_member = ax.bar(
    x - WIDTH/2,
    pivot_weekend_counts['member'],
    width=WIDTH,
    label='Member',
    zorder=2
)
bars_casual = ax.bar(
    x + WIDTH/2,
    pivot_weekend_counts['casual'],
    width=WIDTH,
    label='Casual',
    zorder=2
)

ax.set_title('Total Weekday vs Weekend Rides\nby Membership Status')
ax.set_xlabel('Weekday vs Weekend')
ax.set_ylabel('Total Rides (in millions)')

ax.set_xticks(x)
ax.set_xticklabels(pivot_weekend_counts.index)

ymax = pivot_weekend_counts.max().max() * 1.2
ax.set_ylim(0, ymax)
ax.set_yticks(np.arange(0, ymax, 1_000_000))
ax.set_yticks(np.arange(0, ymax, 500_000), minor=True)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f'{val/1_000_000:.0f}M')
)

ax.grid(False, axis='x')
ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

# Annotation: Bar counts
ax.bar_label(
    bars_member, fmt=lambda x: f'{x/1_000_000:.1f}M',
    padding=5, fontsize=14, fontweight='bold'
)
ax.bar_label(
    bars_casual, fmt=lambda x: f'{x/1_000_000:.1f}M',
    padding=5, fontsize=14, fontweight='bold'
)

plt.savefig(
    viz_png / 'fig_A2_Total_weekday_vs_weekend_rides.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% fig_3_2 Proportion of weekday vs weekend rides by membership status

pivot_weekend_prop = (
    sum_weekend
    .pivot(index='day_type', columns='member_casual', values='percentage')
    .reindex(['Weekday', 'Weekend'])
)

member_prop = pivot_weekend_prop['member'].values / 100
casual_prop = pivot_weekend_prop['casual'].values / 100

x = np.arange(len(pivot_weekend_prop.index))
labels = pivot_weekend_prop.index

fig, ax = plt.subplots(figsize=(10, 8))

WIDTH = 0.6

bars_member = ax.bar(
    x,
    member_prop,
    width=WIDTH,
    label='Member',
    zorder=2,
)
bars_casual = ax.bar(
    x,
    casual_prop,
    width=WIDTH,
    bottom=member_prop,
    label='Casual',
    zorder=2,
)

ax.set_title('Proportion of Weekday vs Weekend\nRides by Membership Status')
ax.set_xlabel('Weekday vs Weekend')
ax.set_ylabel('Proportion of Rides')

ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f'{val*100:.0f}%')
)

ax.grid(False, axis='x')
ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

# Annotation: Bar percentages
for i, (m, c) in enumerate(zip(member_prop, casual_prop)):
    ax.text(x[i], m/2, f'{m:.0%}', ha='center', va='center',
            fontweight='bold', color='white', fontsize=14)
    ax.text(x[i], m + c/2, f'{c:.0%}', ha='center', va='center',
            fontweight='bold', color='black', fontsize=14)

plt.savefig(
    viz_png / 'fig_3_2_Proportion_of_weekday_vs_weekend_rides.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% ---- Time of day ----

#%% fig_4.1 Total rides per start hour by membership status

pivot_hour_counts = (
    sum_hour
        .pivot(index='start_hour', columns='member_casual', values='count')
        .fillna(0)
        .sort_index()
)

member_counts = pivot_hour_counts['member'].values
casual_counts = pivot_hour_counts['casual'].values

x = pivot_hour_counts.index.values

fig, ax = plt.subplots(figsize=(10, 8))

line_member, = ax.plot(
    x,
    member_counts,
    label='Member',
    zorder=3
)
line_casual, = ax.plot(
    x,
    casual_counts,
    label='Casual',
    zorder=3
)

ax.set_title('Total Rides per Start Hour\nby Membership Status')
ax.set_xlabel('Start Hour of Day')
ax.set_ylabel('Total Rides (in thousands)')

ax.set_xlim(0, 23.5)
ax.set_xticks(np.arange(0, 24, 4))
ax.set_xticks(np.arange(0, 24, 1), minor=True)

ymax = pivot_hour_counts.values.max() * 1.1
ax.set_ylim(0, ymax)
ax.yaxis.set_major_locator(MultipleLocator(100_000))
ax.yaxis.set_minor_locator(MultipleLocator(50_000))
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f'{x/1000:.0f}K')
)

ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

plt.savefig(
    viz_png / 'fig_4_1_Total_rides_per_start_hour.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% fig_A3 Proportion of rides per start hour by membership status

hour_total = pivot_hour_counts.sum(axis=1)
hour_prop = pivot_hour_counts.div(hour_total, axis=0).fillna(0)

member_prop = hour_prop['member'].values
casual_prop = hour_prop['casual'].values

x = hour_prop.index.values

fig, ax = plt.subplots(figsize=(10, 8))

ax.stackplot(
    x,
    member_prop,
    casual_prop,
    labels=['Member', 'Casual'],
    alpha=1,
    zorder=2
)

ax.set_title('Hourly Proportion of Rides\nby Membership Status')
ax.set_xlabel('Start Hour of Day')
ax.set_ylabel('Hourly Market Share')

ax.set_xlim(0, 23)
ax.set_xticks(np.arange(0, 24, 4))
ax.set_xticks(np.arange(0, 24, 1), minor=True)

ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f'{val*100:.0f}%')
)

ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

plt.savefig(
    viz_png / 'fig_A3_Proportion_of_rides_per_start_hour.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% fig_4.2 Ride density per start hour and day of week by membership status

pivot_density_day_hour_m = (
    density_day_hour_m
        .pivot(index='start_weekday', columns='start_hour', values='count')
        .reindex(index=range(7), columns=range(24), fill_value=0)
)

pivot_density_day_hour_c = (
    density_day_hour_c
        .pivot(index='start_weekday', columns='start_hour', values='count')
        .reindex(index=range(7), columns=range(24), fill_value=0)
)

pivot_density_day_hour_m = pivot_density_day_hour_m.div(
    pivot_density_day_hour_m.sum(axis=1), axis=0
)
pivot_density_day_hour_c = pivot_density_day_hour_c.div(
    pivot_density_day_hour_c.sum(axis=1), axis=0
)

day_labels  = WEEKDAY_LABELS
hour_labels = list(range(24))

max_share = max(
    pivot_density_day_hour_m.values.max(),
    pivot_density_day_hour_c.values.max()
)

# ---- Combined heat maps

with plt.rc_context({'figure.autolayout': False}):

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    sns.heatmap(
        pivot_density_day_hour_m,
        cmap='rocket_r',
        vmin=0,
        vmax=max_share,
        linewidths=.75,
        cbar=False,
        ax=axes[0]
    )
    sns.heatmap(
        pivot_density_day_hour_c,
        cmap='rocket_r',
        vmin=0,
        vmax=max_share,
        linewidths=.75,
        cbar=False,
        ax=axes[1]
    )

    for ax in axes:
        ax.set_xlabel('Start Hour of Day')

        major_pos = np.arange(0, 24, 4) + 0.5
        ax.set_xticks(major_pos)
        ax.set_xticklabels([str(h) for h in range(0, 24, 4)], rotation=0)

        minor_pos = np.arange(0, 24, 1) + 0.5
        ax.set_xticks(minor_pos, minor=True)

        ax.set_yticks(np.arange(7) + 0.5)
        ax.set_yticklabels(WEEKDAY_LABELS, rotation=0)
        ax.tick_params(axis='y', labelleft=True)

        sns.despine(ax=ax)

    axes[0].set_ylabel('Day of Week')
    axes[1].set_ylabel('')

    axes[0].set_title(
        'Members',
        fontsize=plt.rcParams['axes.labelsize'],
        pad=plt.rcParams['axes.labelpad'],
        fontweight='bold'
    )
    axes[1].set_title(
        'Casual Riders',
        fontsize=plt.rcParams['axes.labelsize'],
        pad=plt.rcParams['axes.labelpad'],
        fontweight='bold'
    )

    fig.suptitle(
        'Ride Density per Day and Hour\nby Membership Status',
        fontsize=plt.rcParams['axes.titlesize'],
        fontweight='bold',
        y=1.01
    )

    fig.tight_layout(rect=[0, 0, 0.93, 1])

    box = axes[-1].get_position()
    cax = fig.add_axes([box.x1 + 0.015, box.y0, 0.02, box.height])

    mappable = axes[1].collections[0]
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label('Share')
    cbar.outline.set_visible(True)

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.savefig(
        viz_png / 'fig_4_2a_Ride_density_per_start_hour_and_day.png',
        bbox_inches="tight",
        dpi=300
    )
    plt.show()

# ---- Individual heat maps

# Member

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    pivot_density_day_hour_m,
    cmap='rocket_r',
    vmin=0,
    vmax=max_share,
    linewidths=.75,
    cbar_kws={'label': 'Share'},
    ax=ax
)

cbar = ax.collections[0].colorbar

cbar.outline.set_visible(True)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1.5)

fig.suptitle(
    'Ride Density per Day & Hour',
    fontsize=plt.rcParams['axes.titlesize'],
    fontweight='bold',
    y=1.01
)
ax.set_title(
    'Members',
    fontsize=plt.rcParams['axes.labelsize'],
    pad=plt.rcParams['axes.labelpad'],
    fontweight='bold'
    )
ax.set_xlabel('Start Hour of Day')
ax.set_ylabel('Day of Week')

ax.set_xticks(np.arange(0, 24, 4) + 0.5)
ax.set_xticks(np.arange(0, 24, 1) + 0.5, minor=True)

ax.set_yticks(np.arange(7) + 0.5)
ax.set_yticklabels(WEEKDAY_LABELS, rotation=0)

sns.despine(ax=ax)
for s in ax.spines.values():
    s.set_visible(True)

plt.savefig(
    viz_png / 'fig_4_2b_Member_ride_density_per_start_hour_and_day.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

# Casual

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    pivot_density_day_hour_c,
    cmap='rocket_r',
    vmin=0,
    vmax=max_share,
    linewidths=.75,
    cbar_kws={'label': 'Share'},
    ax=ax
)

cbar = ax.collections[0].colorbar

cbar.outline.set_visible(True)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1.5)

fig.suptitle(
    'Ride Density per Day & Hour',
    fontsize=plt.rcParams['axes.titlesize'],
    fontweight='bold',
    y=1.01
)
ax.set_title(
    'Casual Riders',
    fontsize=plt.rcParams['axes.labelsize'],
    pad=plt.rcParams['axes.labelpad'],
    fontweight='bold'
    )
ax.set_xlabel('Start Hour of Day')
ax.set_ylabel('Day of Week')

ax.set_xticks(np.arange(0, 24, 4) + 0.5)
ax.set_xticks(np.arange(0, 24, 1) + 0.5, minor=True)

ax.set_yticks(np.arange(7) + 0.5)
ax.set_yticklabels(WEEKDAY_LABELS, rotation=0)

sns.despine(ax=ax)
for s in ax.spines.values():
    s.set_visible(True)

plt.savefig(
    viz_png / 'fig_4_2c_Casual_ride_density_per_start_hour_and_day.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% ---- Ride duration ----

#%% fig_5.1 Distribution of ride duration by membership status

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
member_color = colors[0]
casual_color = colors[1]

palette = {
    'Member': member_color,
    'Casual': casual_color
}

# Boxplot data preparation
dist_duration_overall = (
    df[['ride_duration_min', 'member_casual']]
    .assign(membership_group=lambda d: d['member_casual'].map({
        'member': 'Member',
        'casual': 'Casual'
    }))
    [['ride_duration_min', 'membership_group']]
)

median_values = (
    dist_duration_overall
    .groupby('membership_group')['ride_duration_min']
    .median()
)

fig, ax = plt.subplots(figsize=(10, 8))

sns.boxplot(
    data=dist_duration_overall,
    x='ride_duration_min',
    y='membership_group',
    hue='membership_group',
    order=['Member', 'Casual'],
    palette=palette,
    saturation=1,
    showfliers=False,
    boxprops={'edgecolor': 'black', 'linewidth': 1.5},
    whiskerprops={'color': 'black', 'linewidth': 1.5},
    capprops={'color': 'black', 'linewidth': 1.5},
    medianprops={'color': 'black', 'linewidth': 1.5},
    ax=ax
)

ax.set_title('Distribution of Ride Durations\nby Membership Status')
ax.set_xlabel('Ride Duration (minutes)')
ax.set_ylabel('Membership Status')

ax.set_xlim(left=0)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))

ax.grid(which='minor', axis='x', linestyle='-', alpha=0.2, zorder=0)

y_positions = {
    'Member': 0,
    'Casual': 1
}

for group, median in median_values.items():
    y = y_positions[group]

    ax.annotate(
        f'Median = {median:.1f} min',
        xy=(median, y),
        xytext=(median, y - .5),
        va='center',
        ha='center',
        fontsize=14,
        fontweight='bold'
    )

    ax.vlines(
        x=median,
        ymin=y - 0.40,
        ymax=y + 0.40,
        colors='black',
        linewidth=1.5,
        zorder=3
    )

ax.text(
    0.99, 0.01,
    'Boxes show IQR (Q1–Q3); whiskers truncated; outliers hidden',
    transform=ax.transAxes,
    ha='right', va='bottom',
    fontsize=11, color='dimgray'
)


plt.savefig(
    viz_png / 'fig_5_1_Distribution_of_ride_duration.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% fig_5.2 Median ride duration per weekday by membership status

pivot_duration_weekday = duration_weekday.pivot(
    index='start_weekday',
    columns='member_casual',
    values='median'
)

x = np.arange(len(pivot_duration_weekday.index))
WIDTH = 0.38

fig, ax = plt.subplots(figsize=(10, 8))

bars_member = ax.bar(
    x - WIDTH/2,
    pivot_duration_weekday['member'],
    width=WIDTH,
    label='Member',
    zorder=2
)
bars_casual = ax.bar(
    x + WIDTH/2,
    pivot_duration_weekday['casual'],
    width=WIDTH,
    label='Casual',
    zorder=2
)

ax.set_title('Median Ride Duration per Weekday\nby Membership Status')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Median Duration (minutes)')

ax.set_xticks(x)
ax.set_xticklabels(WEEKDAY_LABELS)

ymax = pivot_duration_weekday.max().max() * 1.2
ax.set_ylim(0, ymax)
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))

ax.grid(False, axis='x')
ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

# Annotation: Bar counts
ax.bar_label(
    bars_member, fmt='%.1f', padding=3, fontsize=12, fontweight='bold'
)
ax.bar_label(
    bars_casual, fmt='%.1f', padding=3, fontsize=12, fontweight='bold'
)

plt.savefig(
    viz_png / 'fig_5_2_Median_ride_duration_per_weekday.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% ---- Top locations ----

#%% fig_6.1 Top start stations by membership status

xmax = max([
    df[df[col] != 'no_station_recorded']['count'].max()
    for df, col in [
        (top_start_m, 'start_station_name'),
        (top_start_c, 'start_station_name'),
        (top_end_m,   'end_station_name'),
        (top_end_c,   'end_station_name')
    ]
]) * 1.15

with plt.rc_context({'figure.autolayout': False}):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    member_color = colors[0]
    casual_color = colors[1]

    for i, (member_df, casual_df, station_col, label) in enumerate([
        (top_start_m, top_start_c, 'start_station_name', 'Start'),
        (top_end_m,   top_end_c,   'end_station_name',   'End')
    ]):
        suffix = chr(ord('a') + i)

        member_top = (
            member_df[member_df[station_col] != 'no_station_recorded']
            .head(10)
            .sort_values('count')
        )
        casual_top = (
            casual_df[casual_df[station_col] != 'no_station_recorded']
            .head(10)
            .sort_values('count')
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

        for ax, data, color, group in zip(
            axes,
            [member_top, casual_top],
            [member_color, casual_color],
            ['Members', 'Casual Riders']
        ):
            bars = ax.barh(data[station_col], data['count'], color=color)

            ax.set_xlim(0, xmax)

            ax.set_title(
                f'{group}', fontsize=22, pad=22, fontweight='bold'
            )
            ax.set_xlabel('Total Rides')
            if ax is axes[0]:
                ax.set_ylabel('Station Name')
            else:
                ax.set_ylabel('')

            ax.tick_params(axis='both', labelsize=14)

            ax.xaxis.set_major_locator(MultipleLocator(10_000))
            ax.xaxis.set_minor_locator(MultipleLocator(5_000))
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda x, pos: f'{x/1000:.0f}K')
            )

            ax.grid(False, axis='y')
            ax.grid(
                which='minor', axis='x',
                linestyle='-', alpha=0.2, zorder=0
            )

            # Annotation: Bar counts
            for bar in bars:
                w = bar.get_width()
                ax.text(w * 1.01, bar.get_y() + bar.get_height()/2,
                        f'{w/1000:.1f}K', va='center',
                        fontsize=12, fontweight='bold')

        fig.suptitle(
            f'Top 10 {label} Stations by Membership Status',
            fontsize=plt.rcParams['axes.titlesize'],
            y=1.01,
            fontweight='bold'
        )

        fig.tight_layout()
        fig.subplots_adjust(wspace=1.25)

        plt.savefig(
            viz_png / f'fig_6_1{suffix}_Top_{label.lower()}_stations.png',
            bbox_inches="tight",
            dpi=300
        )
        plt.show()

#%% ---- Round trips ----

#%% fig_7.1 Total one-way vs round-trips by membership status

pivot_roundtrip_counts = sum_roundtrip.pivot(
    index='is_roundtrip',
    columns='member_casual',
    values='count'
)

x = np.arange(len(pivot_roundtrip_counts.index))
WIDTH = 0.38

fig, ax = plt.subplots(figsize=(10, 8))

bars_member = ax.bar(
    x - WIDTH / 2,
    pivot_roundtrip_counts['member'],
    width=WIDTH,
    label='Member',
    zorder=2
)
bars_casual = ax.bar(
    x + WIDTH / 2,
    pivot_roundtrip_counts['casual'],
    width=WIDTH,
    label='Casual',
    zorder=2
)

ax.set_title('Total One-Way vs Round-Trips\nby Membership Status')
ax.set_xlabel('One-Way vs Round-Trips')
ax.set_ylabel('Total Rides (in millions)')

ax.set_xticks(x)
ax.set_xticklabels(['One-Way', 'Round-Trip'])

ymax = pivot_roundtrip_counts.max().max() * 1.25
ax.set_ylim(0, ymax)
ax.set_yticks(np.arange(0, ymax, 1_000_000))
ax.set_yticks(np.arange(0, ymax, 500_000), minor=True)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f'{val/1_000_000:.0f}M')
)

ax.grid(False, axis='x')
ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

# Annotation: Bar counts
ax.bar_label(
    bars_member, fmt=lambda x: f'{x/1_000_000:.2f}M',
    padding=5, fontsize=14, fontweight='bold'
)
ax.bar_label(
    bars_casual, fmt=lambda x: f'{x/1_000_000:.2f}M',
    padding=5, fontsize=14, fontweight='bold'
)

plt.savefig(
    viz_png / 'fig_7_1_Total_one_way_vs_round_trips.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% fig_7.2 Proportion of one-way vs round-trips by membership status

pivot_roundtrip_prop = (
    sum_roundtrip
    .pivot(index='is_roundtrip', columns='member_casual', values='percentage')
    .reindex([0, 1])
)

member_prop = pivot_roundtrip_prop['member'].values / 100
casual_prop = pivot_roundtrip_prop['casual'].values / 100

x = np.arange(len(pivot_roundtrip_prop.index))
labels = ['One-Way', 'Round-Trip']

fig, ax = plt.subplots(figsize=(10, 8))

WIDTH = 0.6

bars_member = ax.bar(
    x,
    member_prop,
    width=WIDTH,
    label='Member',
    zorder=2
)
bars_casual = ax.bar(
    x,
    casual_prop,
    width=WIDTH,
    bottom=member_prop,
    label='Casual',
    zorder=2
)

ax.set_title('Proportion of One-Way vs Round-Trips\nby Membership Status')
ax.set_xlabel('Trip Type')
ax.set_ylabel('Proportion of Rides')

ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f'{val*100:.0f}%')
)

ax.grid(False, axis='x')
ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

# Annotation: Bar percentages
for i, (m, c) in enumerate(zip(member_prop, casual_prop)):
    ax.text(x[i], m/2, f'{m:.0%}', ha='center', va='center',
            fontweight='bold', color='white', fontsize=14)
    ax.text(x[i], m + c/2, f'{c:.0%}', ha='center', va='center',
            fontweight='bold', color='black', fontsize=14)

plt.savefig(
    viz_png / 'fig_7_2_Proportion_of_one_way_vs_round_trips.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% ---- Additional visuals ----

#%% fig_A4 Total rides by rideable type and membership status

pivot_biketype_counts = sum_biketype.pivot(
    index='rideable_type',
    columns='member_casual',
    values='count'
)

x = np.arange(len(pivot_biketype_counts.index))
WIDTH = 0.38

fig, ax = plt.subplots(figsize=(10, 8))

bars_member = ax.bar(
    x - WIDTH/2,
    pivot_biketype_counts['member'],
    width=WIDTH,
    label='Member',
    zorder=2
)
bars_casual = ax.bar(
    x + WIDTH/2,
    pivot_biketype_counts['casual'],
    width=WIDTH,
    label='Casual',
    zorder=2
)

ax.set_title('Total Classic vs Electric Bike Rides\nby Membership Status')
ax.set_xlabel('Classic vs Electric Bikes')
ax.set_ylabel('Total Rides (in millions)')

ax.set_xticks(x)
ax.set_xticklabels(['Classic Bike', 'Electric Bike'])

ymax = pivot_biketype_counts.max().max() * 1.2
ax.set_ylim(0, ymax)
ax.set_yticks(np.arange(0, ymax, 1_000_000))
ax.set_yticks(np.arange(0, ymax, 500_000), minor=True)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f'{val/1_000_000:.0f}M')
)

ax.grid(False, axis='x')
ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

# Annotate bars with counts
ax.bar_label(
    bars_member, fmt=lambda x: f'{x/1_000_000:.1f}M',
    padding=5, fontsize=14, fontweight='bold'
)
ax.bar_label(
    bars_casual, fmt=lambda x: f'{x/1_000_000:.1f}M',
    padding=5, fontsize=14, fontweight='bold'
)

plt.savefig(
    viz_png / 'fig_A4_Total_rides_by_rideable_type.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

#%% fig_A5 Proportion of rides by rideable type and membership status

pivot_biketype_prop = (
    sum_biketype
    .pivot(index='rideable_type', columns='member_casual', values='percentage')
)

member_prop = pivot_biketype_prop['member'].values / 100
casual_prop = pivot_biketype_prop['casual'].values / 100

x = np.arange(len(pivot_biketype_prop.index))
labels = ['Classic Bike', 'Electric Bike']
WIDTH = 0.6

fig, ax = plt.subplots(figsize=(10, 8))

bars_member = ax.bar(
    x,
    member_prop,
    width=WIDTH,
    label='Member',
    zorder=2
)
bars_casual = ax.bar(
    x,
    casual_prop,
    width=WIDTH, bottom=member_prop,
    label='Casual',
    zorder=2
)

ax.set_title(
    'Proportion of Classic vs Electric Bike\nRides by Membership Status'
)
ax.set_xlabel('Rideable Type')
ax.set_ylabel('Proportion of Rides')

ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylim(0, 1)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f'{val*100:.0f}%')
)

ax.grid(False, axis='x')
ax.grid(which='minor', axis='y', linestyle='-', alpha=0.2, zorder=0)
ax.legend()

# Annotation: Bar percentages
for i, (m, c) in enumerate(zip(member_prop, casual_prop)):
    ax.text(x[i], m/2, f'{m:.0%}', ha='center', va='center',
            fontweight='bold', color='white', fontsize=14)
    ax.text(x[i], m + c/2, f'{c:.0%}', ha='center', va='center',
            fontweight='bold', color='black', fontsize=14)

plt.savefig(
    viz_png / 'fig_A5_Proportion_by_rideable_type.png',
    bbox_inches="tight",
    dpi=300
)
plt.show()

# Close figures and free resources
plt.close('all')
