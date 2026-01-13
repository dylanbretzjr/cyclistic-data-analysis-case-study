#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cyclistic Bike-Share Analysis Case Study Python Script (Part 3 of 3)
- Build interactive point map

@author: Dylan Bretz Jr.

Description:
    Imports cleaned dataset and generates an interactive,
    density-weighted point map visualizing ride start and
    end locations by membership status using Plotly and
    custom JavaScript to synchronize pan/zoom.

Inputs:
    - '../data/processed/divvy-combined-cleaned.csv'

Outputs:
    - '../viz/html/membership_ride_location_map.html'

Dependencies:
    - config.py module for standardized file paths

Usage:
    Run this script as the third part of the Cyclistic Bike-Share
    Analysis Case Study pipeline.

Note:
    A random sample of 200,000 points is used for safe rendering.

Sections:
    0. CONFIG
    1. LOAD CLEANED DATA
    2. BUILD MAPS
    3. LINK PAN/ZOOM
    4. EXPORT HTML
"""

# =============================================================================
#%% 0. CONFIG
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import config

# ---- GLOBAL CONSTANTS ----

GRID = 0.001
MAX_POINTS = 200_000

CHICAGO_LON_RANGE = [-87.92, -87.47]
CHICAGO_LAT_RANGE = [41.64, 42.09]

DEFAULT_CENTER = {
    'lat': sum(CHICAGO_LAT_RANGE) / 2,
    'lon': sum(CHICAGO_LON_RANGE) / 2,
}

LEFT_X  = [0.07, 0.47]
RIGHT_X = [0.48, 0.88]
TOP_Y   = [0.51, 1.00]
BOT_Y   = [0.00, 0.49]

# ---- UTILITY FUNCTIONS ----

def make_binned_data(df_in, lat_col, lon_col):
    """
    Downsamples data (if necessary) and bins coordinates to grid
    to reduce rendering load while preserving density patterns.
    """
    df_map = df_in.dropna(subset=[lat_col, lon_col]).copy()

    if len(df_map) > MAX_POINTS:
        df_map = df_map.sample(MAX_POINTS, random_state=42)

    df_map['lat_bin'] = (df_map[lat_col] / GRID).round() * GRID
    df_map['lon_bin'] = (df_map[lon_col] / GRID).round() * GRID

    df_bins = (
        df_map.groupby(['lat_bin', 'lon_bin'])
              .size()
              .reset_index(name='count')
    )
    return df_bins


def compute_opacity(counts):
    """
    Calculates marker opacity using sigmoid function on log-transformed counts.
    This highlights high-density areas without washing out low-density paths.
    """
    c = counts.to_numpy()
    c_log = np.log1p(c)
    c_norm = (c_log - c_log.min()) / (c_log.max() - c_log.min() + 1e-9)

    k = 10.0
    x0 = 0.25
    c_sig = 1 / (1 + np.exp(-k * (c_norm - x0)))
    c_sig = (c_sig - c_sig.min()) / (c_sig.max() - c_sig.min() + 1e-9)

    return 0.15 + 0.70 * c_sig


def build_bins(member_type, kind):
    """Wrapper to filter data and calculate bins and opacity."""
    df_sub = df[df['member_casual'] == member_type]

    if kind == 'start':
        bins = make_binned_data(df_sub, 'start_lat', 'start_lng')
        kind_label = 'Start'
    else:
        bins = make_binned_data(df_sub, 'end_lat', 'end_lng')
        kind_label = 'End'

    alpha = compute_opacity(bins['count'])
    member_label = 'Members' if member_type == 'member' else 'Casual Riders'
    return bins, alpha, kind_label, member_label

def make_trace(bins, alpha, title, subplot_name, sizeref):
    """Create single Plotly Scattermap trace."""
    return go.Scattermap(
        lat=bins['lat_bin'],
        lon=bins['lon_bin'],
        mode='markers',
        subplot=subplot_name,
        customdata=bins[['count']],
        hovertemplate=(
            f'{title}<br>'
            'Count: %{customdata[0]}<br>'
            'Lat: %{lat:.4f}<br>'
            'Lon: %{lon:.4f}<extra></extra>'
        ),
        marker={
            'size': bins['count'],
            'sizemode': 'area',
            'sizeref': sizeref,
            'sizemin': 2.5,
            'color': bins['count'],
            'opacity': alpha,
            'coloraxis': 'coloraxis',
        },
        name=title,
        showlegend=False,
    )

def map_layout(domain_x, domain_y):
    """
    Set styling and positioning for each map panel.
    """
    return {
        'domain': {'x': domain_x, 'y': domain_y},
        'style': 'carto-positron',
        'center': DEFAULT_CENTER,
        'bounds': {
            'west': CHICAGO_LON_RANGE[0],
            'east': CHICAGO_LON_RANGE[1],
            'south': CHICAGO_LAT_RANGE[0],
            'north': CHICAGO_LAT_RANGE[1],
        },
    }

# =============================================================================
#%% 1. LOAD CLEANED DATA
# =============================================================================

clean_path = config.CLEAN_PATH
if not clean_path.exists():
    raise FileNotFoundError(f'Cleaned data not found at {config.CLEAN_PATH}')

df = pd.read_csv(clean_path, parse_dates=['started_at', 'ended_at'])

# =============================================================================
#%% 2. BUILD MAPS
# =============================================================================

bins_sm, alpha_sm, _, _ = build_bins('member', 'start')
bins_em, alpha_em, _, _ = build_bins('member', 'end')
bins_sc, alpha_sc, _, _ = build_bins('casual', 'start')
bins_ec, alpha_ec, _, _ = build_bins('casual', 'end')

GLOBAL_MAX = int(max(
    bins_sm['count'].max(),
    bins_em['count'].max(),
    bins_sc['count'].max(),
    bins_ec['count'].max(),
))
SIZEREF = GLOBAL_MAX / (25 ** 2)

fig = go.Figure()

fig.add_trace(make_trace(bins_sm, alpha_sm, 'Members + Start', 'map', SIZEREF))
fig.add_trace(make_trace(bins_em, alpha_em, 'Members + End',   'map2', SIZEREF))
fig.add_trace(make_trace(bins_sc, alpha_sc, 'Casual + Start',  'map3', SIZEREF))
fig.add_trace(make_trace(bins_ec, alpha_ec, 'Casual + End',    'map4', SIZEREF))

fig.update_layout(
    map=map_layout(LEFT_X, TOP_Y),
    map2=map_layout(RIGHT_X, TOP_Y),
    map3=map_layout(LEFT_X, BOT_Y),
    map4=map_layout(RIGHT_X, BOT_Y),

    coloraxis={
        'colorscale': 'Viridis',
        'cmin': 0,
        'cmax': GLOBAL_MAX,
        'colorbar': {
            'title': {
                'text': '<b>Ride Density</b>',
                'side': 'right',
                'font': {'size': 22, 'color': 'black'},
            },
            'thickness': 30,
            'len': 0.95,
            'x': 0.90,
            'y': 0.5,
            'tickmode': 'array',
            'tickvals': [0, 1000, 2000, 3000, 4000, 5000],
            'ticktext': ['0', '1000', '2000', '3000', '4000', '5000'],
            'tickfont': {'size': 16, 'color': 'black'},
            'ticks': 'outside',
            'ticklen': 8,
            'tickwidth': 1.5,
            'outlinecolor': 'black',
            'outlinewidth': 1.5,
        },
    },

    font={'color': 'black'},
    font_family='DejaVu Sans, Verdana, Arial, sans-serif',
    title={
        'text': '<b>Ride Locations by Membership Status</b>',
        'x': 0.5,
        'xanchor': 'center',
        'y': 0.98,
        'font': {'size': 28},
    },

    margin={'l': 75, 'r': 0, 't': 125, 'b': 0},

    annotations=[
        {
            'text': '<b>Start Location</b>',
            'x': 0.20,
            'y': 1.06,
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 22, 'color': 'black'}
        },
        {
            'text': '<b>End Location</b>',
            'x': 0.73,
            'y': 1.06,
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 22, 'color': 'black'}},
        {
            'text': '<b>Members</b>',
            'x': 0.05,
            'y': 0.85,
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'xanchor': 'right',
            'textangle': -90,
            'font': {'size': 22, 'color': 'black'}
        },
        {
            'text': '<b>Casual Riders</b>',
            'x': 0.05,
            'y': 0.15,
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'xanchor': 'right',
            'textangle': -90,
            'font': {'size': 22, 'color': 'black'}
        },
    ],
    uirevision='locked-view',
    dragmode='pan',
)

# =============================================================================
#%% 3. LINK PAN/ZOOM
# =============================================================================
# Steps:
# - Listen for plotly_relayout events (pan/zoom) on any map subplot
# - Copy center/zoom/bearing/pitch changes to the other subplots

SYNC_JS = r"""
var gd = document.getElementsByClassName('plotly-graph-div')[0];
var maps = ['map', 'map2', 'map3', 'map4'];

var syncing = false;

function findSourceMap(e){
  for (var i=0; i<maps.length; i++){
    var m = maps[i];
    var prefix = m + '.';
    for (var k in e){
      if(k.startsWith(prefix)) return m;
    }
  }
  return null;
}

function buildUpdates(e){
  var source = findSourceMap(e);
  if(!source) return null;

  var updates = {};
  var viewKeys = ['center', 'zoom', 'bearing', 'pitch'];

  viewKeys.forEach(function(vk){
    var key = source + '.' + vk;
    if(e[key] !== undefined){
      maps.forEach(function(target){
        if(target === source) return;
        updates[target + '.' + vk] = e[key];
      });
    }
  });

  return Object.keys(updates).length ? updates : null;
}

// IMPORTANT: sync ONLY on plotly_relayout (drag end), never during relayouting
gd.on('plotly_relayout', function(e){
  if(syncing) return;

  var updates = buildUpdates(e);
  if(!updates) return;

  syncing = true;
  Plotly.relayout(gd, updates).then(function(){
    syncing = false;
  }).catch(function(){
    syncing = false;
  });
});
"""

# =============================================================================
#%% 4. EXPORT HTML
# =============================================================================

# Write linked HTML
# Ensure output directory exists
viz_html = config.VIZ_PATH / 'html'
viz_html.mkdir(parents=True, exist_ok=True)

pio.write_html(
    fig,
    file=viz_html / 'membership_ride_location_map.html',
    auto_open=True,
    include_plotlyjs='cdn',
    post_script=SYNC_JS,
    config={
        'scrollZoom': True,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'membership_ride_location_map',
            'height': 900,
            'width': 1600,
            'scale': 2
        }
    }
)
