"""
Water SCARCE Criticality Assessment Script – Compliance with Social Standards Dimension
-----------------------------------------------------------------------------

This script calculates the social compliance component of Water 
SCARCE. The indicator reflects the extent to which countries and 
regions meet environmental performance standards related to biodiversity 
conservation, ecosystem quality, and air quality.

Main outputs:
    - Scaled values for each category (0–1)
    - Aggregated water compliance with social standards score for use in the overall criticality assessment
    - Map with global compliance with social standards distribution on regional and country level
    
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import time
import xarray as xr
import rioxarray
import cftime
import netCDF4 as nc
from sklearn.preprocessing import MinMaxScaler

import cartopy.crs as ccrs
import cartopy

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import mapclassify

from math import radians, sin

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import mapping
from shapely.ops import nearest_points
import shapefile as shp
import fiona
from tqdm import tqdm
import country_converter as coco

from data_processing import process_all_sectors, save_to_excel

# Initialize the country converter
cc = coco.CountryConverter()

# %% Categories calculation

# ---- COMPLIANCE WITH SOCIAL STANDARDS

# Load the GeoPackage
intersections_regions = gpd.read_file(
    'intersections_regions.gpkg',
    layer='indicators_data'
    )

intersections_regions_extra = gpd.read_file(
    'intersections_regions_extra.gpkg',
    layer='indicators'
    )

# Remove rows where the country column has empty value
intersections_regions_extra = intersections_regions_extra.dropna(
    subset=['country']
    )

# Set a scaler to scale the results between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

# ---- HUMAN RIGHTS ABUSE

# Sample data
social_data = intersections_regions_extra[[
    'country',
    'country code',
    'province',
    'main basins',
    'geometry'
]]

# Process all three sectors
output_data_agriculture, output_data_manufacturing, output_data_mining = process_all_sectors()

# Initialize lists to hold the filtered DataFrames for each sector
agriculture_indicators = []
manufacturing_indicators = []
mining_indicators = []

# Define the target filters
flow_filter = ['Children in employment, total',
               'Drinking water coverage',
               'Frequency of forced labour',
               'Evidence of violations of laws and employment regulations',
               'Global Peace Index '
               ]
category_filter = ['Social flows/Workers/Child labour', 
                   'Social flows/Local Community/Safe and healthy living conditions',
                   'Social flows/Workers/Forced Labour',
                   'Social flows/Workers/Social benefits, legal issues',
                   'Social flows/Society/Health and Safety'
                   ]

# Function to filter sector data
def extract_sector_data(output_data, sector_name):
    indicators = []
    for country, df in output_data.items():
        # Check if required columns exist
        if all(col in df.columns for col in ['Flow', 'Category', 'Amount', 'Unit']):
            # Use .isin() to filter multiple values
            filtered_df = df[
                df['Flow'].isin(flow_filter) & 
                df['Category'].isin(category_filter)
            ].copy()

            if not filtered_df.empty:
                filtered_df['Country'] = country  # Add the country column
                indicators.append(filtered_df)
        else:
            print(f"Missing columns in {sector_name} data for {country}")

    # Return concatenated DataFrame or empty DataFrame
    return pd.concat(indicators, ignore_index=True) if indicators else pd.DataFrame()

# Extract indicators for each sector
agriculture_indicators_social = extract_sector_data(output_data_agriculture, 'agriculture')
manufacturing_indicators_social = extract_sector_data(output_data_manufacturing, 'manufacturing')
mining_indicators_social = extract_sector_data(output_data_mining, 'mining')

# Merge all sectoral data into one DataFrame
all_sectors_df = pd.concat(
    [agriculture_indicators_social, manufacturing_indicators_social, mining_indicators_social],
    ignore_index=True
)

# Compute average per Country, Flow, and Category
average_social_indicators = (
    all_sectors_df.groupby(['Country', 'Flow', 'Category'])['Amount']
    .mean()
    .reset_index()
    .rename(columns={'Amount': 'Average_Amount'})
)

# Add country codes
average_social_indicators['country code'] = cc.convert(names=average_social_indicators['Country'], to='ISO3')

# Define manual corrections for known deviations
replaced_country_codes = {
    'ADO': 'AND',
    'ROM': 'ROU',
    'ZAR': 'COD',
    'WBG': 'PSE',
    'TMP': 'TLS',
    'KSV': 'XKX'
}

# Apply replacements
average_social_indicators['country code'] = average_social_indicators['country code'].replace(replaced_country_codes)

# Remove rows where country code is "not found"
average_social_indicators = average_social_indicators[average_social_indicators['country code'] != "not found"]

# Pivot for easier processing
pivot_df = average_social_indicators.pivot_table(
    index=['Country', 'country code'], 
    columns='Flow', 
    values='Average_Amount'
).reset_index()

# Invert indicators where necessary
pivot_df['Drinking water coverage'] = 100 - pivot_df['Drinking water coverage']
# Global Peace Index is already high=bad, keep as is
# Children in employment, total: already high=bad
# Frequency of forced labour: already high=bad

# Now scale all indicators between 0 and 1
indicators_only = pivot_df.drop(columns=['Country', 'country code'])
scaled_indicators = pd.DataFrame(scaler.fit_transform(indicators_only), columns=indicators_only.columns)
scaled_indicators[['Country', 'country code']] = pivot_df[['Country', 'country code']]

# Sum scaled indicators
scaled_indicators['social_compliance_raw'] = scaled_indicators.drop(columns=['Country', 'country code']).sum(axis=1)

# Scale final score between 0 and 1 again (optional but good practice)
scaled_indicators['social compliance scaled'] = MinMaxScaler().fit_transform(
    scaled_indicators[['social_compliance_raw']]
)

# Mege on 'Country' (from scaled_indicators) and 'country' (from social_data)
social_standards_compliance = scaled_indicators.merge(
    social_data,
    on='country code',
    how='left'
).drop(columns=['Country'])

# Convert to GeoDataFrame
social_standards_compliance = gpd.GeoDataFrame(social_standards_compliance, crs='EPSG:6933')

# Save as CSV file
social_standards_compliance.to_csv('social_standards_compliance_all.csv', index=False)
social_standards_compliance[['country', 'country code', 'social compliance scaled', 'geometry']].to_csv('social_standards_compliance_final.csv', index=False)

#%% Plot
#Reproject

social_standards_compliance = social_standards_compliance.to_crs(epsg=4326)  # Natural Earth projection 

# Define number of equal bins and create the bins from 0 to 1
num_bins = 10
bins = np.linspace(0, 1, num_bins + 1)

# Classify data using pd.cut instead of qcut (qcut = quantiles, cut = equal intervals)
bin_labels = [f'{i + 1}' for i in range(num_bins)]
social_standards_compliance['quantile_class'] = pd.cut(
    social_standards_compliance['social compliance scaled'],
    bins=bins,
    labels=bin_labels,
    include_lowest=True
)

# Create the plot
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

# Plot the map with the 'supply risk score' column using quantiles
social_standards_compliance.plot(
    column='social compliance scaled',  # Use the 'supply risk score' column for coloring
    cmap='viridis',
    scheme='user_defined',
    classification_kwds={'bins': bins[1:-1]},  # exclude 0 and 1 from classification bins
    legend=False,  # We'll make our own legend
    ax=ax,
    edgecolor='none',
    linewidth=0.0,
    missing_kwds={'color': 'lightgrey', 'label': 'No data'}
)

# Create custom legend with circular markers matching colormap
cmap = plt.cm.get_cmap('viridis', num_bins)
for i in range(num_bins):
    low, high = bins[i], bins[i + 1]
    label = f'{low:.2f} - {high:.2f}'
    color = cmap(i / (num_bins - 1))  # map i to color space [0,1]
    ax.scatter([], [], color=color, label=label, marker='o', s=100)

legend_elements = []  # Initialize the list for the legend elements

# Add the "No value" entry with light grey color to the legend
legend_elements.append(ax.scatter([], [], color='lightgrey', label='No value', marker='o', s=100))

# Add the custom legend
ax.legend(title='Social compliance', loc='lower left', frameon=False)

# Set title for the plot
# ax.set_title('Compliance with social standards', fontsize=16)

# Save
plt.savefig('social_compliance_map.svg', bbox_inches='tight')
plt.savefig('social_compliance_map.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()