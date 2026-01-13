#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 12:58:56 2025

@author: Dylan Bretz Jr.

Cyclistic Bike-Share Analysis Case Study Python Configuration Module
- Define global file paths

Description:
    Defines paths for raw data, cleaned data, visualizations, and styles.

Usage:
    Import this module in other scripts to access standardized file paths.
"""

from pathlib import Path

# ---- PATH CONFIGURATION ----

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Fallback for IDE
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == 'src':
        PROJECT_ROOT = PROJECT_ROOT.parent

# Input/Output Directories
RAW_PATH = PROJECT_ROOT / 'data' / 'raw' / 'csv-files'
CLEAN_PATH = (
    PROJECT_ROOT / 'data' / 'processed' / 'divvy-combined-cleaned.csv'
)
VIZ_PATH = PROJECT_ROOT / 'viz'

# Styles
STYLE_PATH = PROJECT_ROOT / 'styles' / 'divvy.mplstyle'
