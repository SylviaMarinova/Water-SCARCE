"""
Water SCARCE Criticality Assessment – Indicator Preparation Script
-----------------------------------------------------------

This script collects, processes, and harmonises all input indicators used for 
the assessment of water criticality. The prepared indicators are subsequently 
used in the three main dimensions of the Water SCARCE:
    1. Supply Risk
    2. Vulnerability
    3. Compliance with Environmental Standards
    4. Compliance with Social Standards

Scope:
    - Integrates environmental, hydrological, and socio-economic datasets
    - Produces a harmonised geospatial dataset for subsequent calculations
    - Ensures consistent scales, units, and coverage across all indicators

Main steps:
    1. Load and merge raw datasets.
    2. Overlay bioms, regions and countries.
    3. Preprocess variables (unit conversions, cleaning, handling missing values).
    4. Ensure consistent spatial resolution and projection.
    5. Store prepared indicators as GeoDataFrames for further use in each dimension.
    6. Export harmonised indicator datasets to be used by the Supply Risk,
       Vulnerability, and Compliance dimensions.

Outputs:
    - GeoPackages with harmonized indicators
    - CSV files containing attribute tables
    - Ready-to-use geospatial layers for integration in the three assessment dimensions

This script provides the backbone for the entire Water Criticality framework, 
ensuring transparency, comparability, and reproducibility of the assessment.

"""

import geopandas as gpd
import numpy as np
import pandas as pd
import time
import xarray as xr
import rioxarray
import cftime
import netCDF4 as nc
from scipy.spatial import cKDTree
from difflib import SequenceMatcher
import glob

import cartopy.crs as ccrs
import cartopy

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns
import mapclassify

from math import radians, sin

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Point
from shapely.geometry import mapping
from shapely.ops import nearest_points
import shapefile as shp
import fiona
from tqdm import tqdm

import country_converter as coco
import os
from custom_py.watergapextraction import watergapextraction

# Initialize the country converter
cc = coco.CountryConverter()

# %% Data import

def main():
    # Call the functions from data.py
    ned_admin_0, ned_admin_1 = watergapextraction.import_natural_earth_data()
    wg22, extracted_data = watergapextraction.extract_watergap_data()
    indicators = watergapextraction.process_watergap_indicators(extracted_data)
    indicators = watergapextraction.apply_thresholds(wg22, indicators)
    raster_df, coordinates_lat, coordinates_lon = watergapextraction.create_raster()
    raster_df = watergapextraction.assign_data_to_raster(raster_df, coordinates_lat, coordinates_lon, indicators, wg22)
    land_grid_cell_ids = watergapextraction.import_biomes()
    raster_df = watergapextraction.filter_land_cells(raster_df, land_grid_cell_ids)
    raster_df = watergapextraction.import_and_merge_hydrosheds_basins(raster_df)

    raster_df = raster_df.to_crs('EPSG:6933') # CRS transformation
    
    return raster_df, ned_admin_1, ned_admin_0

if __name__ == "__main__":
    Water_SCARCE_indicators, ned_admin_1, ned_admin_0 = main()
    print("Data Import Completed!")

print(Water_SCARCE_indicators.total_bounds)

# Delete the completely empty rows accross of the columns
Water_SCARCE_indicators = Water_SCARCE_indicators.dropna(thresh=Water_SCARCE_indicators.shape[1] - 19)

# %%% Countries and regions

#%% Regions

# Create regions geodataframe 
gdf_regions = ned_admin_1.to_crs('EPSG:6933') # Set the corect coordinate system
gdf_regions_modified = gdf_regions[['adm1_code', 'name', 'type_en', 'geonunit', 'geometry', 'admin']]

# test which columns have unique identifiers for the geometries provided for the regions
unique_columns = [col for col in gdf_regions_modified.columns if gdf_regions_modified[col].is_unique]
print('Columns with unique values:', unique_columns)

# Define the Cylindrical Equal-Area projection (change EPSG if needed)
cea_crs = 'EPSG:6933'  # Common Cylindrical Equal-Area projection
gdf_regions_modified = gdf_regions_modified.to_crs(cea_crs)

# add areas
gdf_regions_modified['area_region'] = gdf_regions_modified.geometry.area
gdf_regions_modified['area_region_km2'] = gdf_regions_modified['area_region'] / 1e6

#%% Countries

# Create regions geodataframe 
gdf_countries = ned_admin_0.to_crs('EPSG:6933') # Set the corect coordinate system
gdf_countries_modified = gdf_countries[['SOVEREIGNT', 'ADMIN', 'geometry']]

# Test which columns have unique identifiers for the geometries provided for the regions
unique_columns = [col for col in gdf_countries_modified.columns if gdf_countries_modified[col].is_unique]
print('Columns with unique values:', unique_columns)

# Change to the Cylindrical Equal-Area projection (change EPSG if needed)
gdf_countries_modified = gdf_countries_modified.to_crs(cea_crs)

# Add areas
gdf_countries_modified['area_country'] = gdf_countries_modified.geometry.area
gdf_countries_modified['area_country_km2'] = gdf_countries_modified['area_country'] / 1e6

# Changes in the indicators geodataframe
# Add coordinate reference system
Water_SCARCE_indicators = Water_SCARCE_indicators.to_crs('EPSG:6933')
# Add areas
Water_SCARCE_indicators['cell_area'] = Water_SCARCE_indicators.geometry.area
Water_SCARCE_indicators['cell_area_km2'] = Water_SCARCE_indicators['cell_area'] / 1e6

Water_SCARCE_indicators = Water_SCARCE_indicators.drop_duplicates(subset='cell_id', keep='first')

# Save GeoDataFrame to GeoPackage
gdf_countries_modified.to_file('countries_modified.gpkg', layer='countries', driver='GPKG')

#%% Overlap countries with grid cells -> then the consumption in mm per country is calculated
# -> then multiplication of the consumption in mm with country areas to get volumes     

# Regions and countries
#Test the bounds of systems to itersect - should be in meters
print(gdf_regions_modified.total_bounds)
print(Water_SCARCE_indicators.total_bounds)

# Perform spatial intersection
intersections_regions = (
    gpd.overlay(
        Water_SCARCE_indicators, 
        gdf_regions_modified, 
        how='intersection')
)

# Rename columns
intersections_regions = intersections_regions.rename(
    columns={'adm1_code':'adm_code', 'name':'province', 'type_en':'type', 'admin':'country'}
)

# Flatten the list if needed (in case you now have a list of Polygons or MultiPolygons)
intersections_regions = intersections_regions.explode('geometry')

# Calculate the area of each intersected part
intersections_regions['intersected_area'] = intersections_regions.geometry.area

intersections_regions['total_area_per_id'] = (
    intersections_regions.groupby('cell_id')['intersected_area'].transform('sum')
)

# Calculate the share of each area relative to the total area of the same cell_id
intersections_regions['intersected_area_share'] = (
    intersections_regions['intersected_area'] / intersections_regions['total_area_per_id']
    )

intersections_regions['num_basins'] = intersections_regions.groupby('country')['main basins'].transform('nunique')

# %% Add and compute indicators

# %%% WaterGAP indicators

# %%%% Actual consumptive water use

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_consumption'] = (
    intersections_regions['consumption'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['consumption_m3'] = (
    intersections_regions['cell_weighted_consumption']
    ) * intersections_regions['intersected_area'] / 1000

# Water consumption per basin
intersections_regions['consumption m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['consumption_m3'].transform('sum')

# Water consumption per region
intersections_regions['consumption m3 per region'] = intersections_regions.groupby(['country', 'province'])['consumption_m3'].transform('sum')

# Water consumption per country
intersections_regions['consumption m3 per country'] = intersections_regions.groupby('country')['consumption_m3'].transform('sum')

#intersections_regions.to_csv('consumption3.csv', index=False)

#intersections_regions.to_csv('World.csv', index=False)

# Set thresholds to deal with the outliners
threshold_consc = intersections_regions['consumption m3 per country'].quantile(0.85)  # 85th percentile
threshold_consr = intersections_regions['consumption m3 per region'].quantile(0.85)  # 85th percentile
threshold_consb = intersections_regions['consumption m3 per basin'].quantile(0.85)  # 85th percentile

intersections_regions['consumption m3 per country'] = intersections_regions['consumption m3 per country'].clip(upper=threshold_consc)
intersections_regions['consumption m3 per region'] = intersections_regions['consumption m3 per region'].clip(upper=threshold_consr)
intersections_regions['consumption m3 per basin'] = intersections_regions['consumption m3 per basin'].clip(upper=threshold_consb)

# Save to shapefile
#intersections_regions.to_file('consumption.shp', driver='ESRI Shapefile')

# Save data
#output_shapefile = 'C:/Users/Sylvia/Desktop/Work/PhD 2023/Python/test222.shp'  # Specify the desired file path
#intersections_regions.to_file(output_shapefile)

# %%%% Potential consumptive water use in livestock sector

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_livestock_consumption'] = (
    intersections_regions['potential_consumption_livestock_sector'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['livestock_consumption_m3'] = (
    intersections_regions['cell_weighted_livestock_consumption'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in livestock sector per basin
intersections_regions['livestock consumption m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['livestock_consumption_m3'].transform('sum')

# Water consumption in livestock sector per region
intersections_regions['livestock consumption m3 per region'] = intersections_regions.groupby(['country', 'province'])['livestock_consumption_m3'].transform('sum')

# Water consumption in livestock sector per country
intersections_regions['livestock consumption m3 per country'] = intersections_regions.groupby('country')['livestock_consumption_m3'].transform('sum')

# %%%% Potential consumptive water use in irrigation sector

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_irrigation_consumption'] = (
    intersections_regions['potential_consumption_irrigation_sector'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['irrigation_consumption_m3'] = (
    intersections_regions['cell_weighted_irrigation_consumption'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in irrigation sector per basin
intersections_regions['irrigation consumption m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['irrigation_consumption_m3'].transform('sum')

# Water consumption in irrigation sector per region
intersections_regions['irrigation consumption m3 per region'] = intersections_regions.groupby(['country', 'province'])['irrigation_consumption_m3'].transform('sum')

# Water consumption in irrigation sector per country
intersections_regions['irrigation consumption m3 per country'] = intersections_regions.groupby('country')['irrigation_consumption_m3'].transform('sum')

# %%%% Potential consumptive water use in irrigation sector from groundwater resources

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_irrigation_consumption_gw'] = (
    intersections_regions['potential_withdrawal_irrigation_sector_groundwater'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['irrigation_consumption_m3_gw'] = (
    intersections_regions['cell_weighted_irrigation_consumption_gw'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in manufacturing sector per basin
intersections_regions['irrigation consumption m3 per basin gw'] = intersections_regions.groupby(['country', 'main basins'])['irrigation_consumption_m3_gw'].transform('sum')

# Water consumption in manufacturing sector per region
intersections_regions['irrigation consumption m3 per region gw'] = intersections_regions.groupby(['country', 'province'])['irrigation_consumption_m3_gw'].transform('sum')

# Water consumption in manufacturing sector per country
intersections_regions['irrigation consumption m3 per country gw'] = intersections_regions.groupby('country')['irrigation_consumption_m3_gw'].transform('sum')

# %%%% Potential consumptive water use in manufacturing sector

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_manufacturing_consumption'] = (
    intersections_regions['potential_consumption_manufacturing_sector'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['manufacturing_consumption_m3'] = (
    intersections_regions['cell_weighted_manufacturing_consumption'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in manufacturing sector per basin
intersections_regions['manufacturing consumption m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['manufacturing_consumption_m3'].transform('sum')

# Water consumption in manufacturing sector per region
intersections_regions['manufacturing consumption m3 per region'] = intersections_regions.groupby(['country', 'province'])['manufacturing_consumption_m3'].transform('sum')

# Water consumption in manufacturing sector per country
intersections_regions['manufacturing consumption m3 per country'] = intersections_regions.groupby('country')['manufacturing_consumption_m3'].transform('sum')

# %%%% Potential consumptive water use in manufacturing sector from groundwater resources

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_manufacturing_consumption_gw'] = (
    intersections_regions['potential_consumption_manufacturing_sector_groundwater'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['manufacturing_consumption_m3_gw'] = (
    intersections_regions['cell_weighted_manufacturing_consumption_gw'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in manufacturing sector per basin
intersections_regions['manufacturing consumption m3 per basin gw'] = intersections_regions.groupby(['country', 'main basins'])['manufacturing_consumption_m3_gw'].transform('sum')

# Water consumption in manufacturing sector per region
intersections_regions['manufacturing consumption m3 per region gw'] = intersections_regions.groupby(['country', 'province'])['manufacturing_consumption_m3_gw'].transform('sum')

# Water consumption in manufacturing sector per country
intersections_regions['manufacturing consumption m3 per country gw'] = intersections_regions.groupby('country')['manufacturing_consumption_m3_gw'].transform('sum')

# %%%% Potential domestic consumptive water use

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_domestic_consumption'] = (
    intersections_regions['potential_consumption_domestic_sector'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['domestic_consumption_m3'] = (
    intersections_regions['cell_weighted_domestic_consumption'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in domestic sector per basin
intersections_regions['domestic consumption m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['domestic_consumption_m3'].transform('sum')

# Water consumption in domestic sector per region
intersections_regions['domestic consumption m3 per region'] = intersections_regions.groupby(['country', 'province'])['domestic_consumption_m3'].transform('sum')

# Water consumption in domestic sector per country
intersections_regions['domestic consumption m3 per country'] = intersections_regions.groupby('country')['domestic_consumption_m3'].transform('sum')

# %%%% Potential consumptive thermoelectric water use

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_thermoelectric_consumption'] = (
    intersections_regions['potential_consumption_thermoelectric_sector'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['thermoelectric_consumption_m3'] = (
    intersections_regions['cell_weighted_thermoelectric_consumption'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in thermoelectric sector per basin
intersections_regions['thermoelectric consumption m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['thermoelectric_consumption_m3'].transform('sum')

# Water consumption in thermoelectric sector per region
intersections_regions['thermoelectric consumption m3 per region'] = intersections_regions.groupby(['country', 'province'])['thermoelectric_consumption_m3'].transform('sum')

# Water consumption in thermoelectric sector per country
intersections_regions['thermoelectric consumption m3 per country'] = intersections_regions.groupby('country')['thermoelectric_consumption_m3'].transform('sum')

# %%%% Potential withdrawal from irrigation sector

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_withdrawal_irrigation_sector'] = (
    intersections_regions['potential_withdrawal_irrigation_sector'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['withdrawal_irrigation_sector_m3'] = (
    intersections_regions['cell_weighted_withdrawal_irrigation_sector'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in thermoelectric sector per basin
intersections_regions['irrigation withdrawal m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['withdrawal_irrigation_sector_m3'].transform('sum')

# Water consumption in thermoelectric sector per region
intersections_regions['irrigation withdrawal m3 per region'] = intersections_regions.groupby(['country', 'province'])['withdrawal_irrigation_sector_m3'].transform('sum')

# Water consumption in thermoelectric sector per country
intersections_regions['irrigation withdrawal m3 per country'] = intersections_regions.groupby('country')['withdrawal_irrigation_sector_m3'].transform('sum')

# %%%% Potential withdrawal from irrigation sector groundwater

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_withdrawal_irrigation_sector_groundwater'] = (
    intersections_regions['potential_withdrawal_irrigation_sector_groundwater'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['withdrawal_irrigation_sector_groundwater_m3'] = (
    intersections_regions['cell_weighted_withdrawal_irrigation_sector_groundwater'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in thermoelectric sector per basin
intersections_regions['irrigation withdrawal groundwater m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['withdrawal_irrigation_sector_groundwater_m3'].transform('sum')

# Water consumption in thermoelectric sector per region
intersections_regions['irrigation withdrawal groundwater m3 per region'] = intersections_regions.groupby(['country', 'province'])['withdrawal_irrigation_sector_groundwater_m3'].transform('sum')

# Water consumption in thermoelectric sector per country
intersections_regions['irrigation withdrawal groundwater m3 per country'] = intersections_regions.groupby('country')['withdrawal_irrigation_sector_groundwater_m3'].transform('sum')

# %%%% Potential withdrawal from manufacturing sector

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_withdrawal_manufacturing_sector'] = (
    intersections_regions['potential_withdrawal_manufacturing_sector'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['withdrawal_manufacturing_sector_m3'] = (
    intersections_regions['cell_weighted_withdrawal_manufacturing_sector'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in thermoelectric sector per basin
intersections_regions['manufacturing withdrawal m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['withdrawal_manufacturing_sector_m3'].transform('sum')

# Water consumption in thermoelectric sector per region
intersections_regions['manufacturing withdrawal m3 per region'] = intersections_regions.groupby(['country', 'province'])['withdrawal_manufacturing_sector_m3'].transform('sum')

# Water consumption in thermoelectric sector per country
intersections_regions['manufacturing withdrawal m3 per country'] = intersections_regions.groupby('country')['withdrawal_manufacturing_sector_m3'].transform('sum')

# %%%% Potential withdrawal from manufacturing sector groundwater

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_withdrawal_manufacturing_sector_groundwater'] = (
    intersections_regions['potential_withdrawal_manufacturing_sector_groundwater'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['withdrawal_manufacturing_sector_groundwater_m3'] = (
    intersections_regions['cell_weighted_withdrawal_manufacturing_sector_groundwater'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in thermoelectric sector per basin
intersections_regions['manufacturing withdrawal groundwater m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['withdrawal_manufacturing_sector_groundwater_m3'].transform('sum')

# Water consumption in thermoelectric sector per region
intersections_regions['manufacturing withdrawal groundwater m3 per region'] = intersections_regions.groupby(['country', 'province'])['withdrawal_manufacturing_sector_groundwater_m3'].transform('sum')

# Water consumption in thermoelectric sector per country
intersections_regions['manufacturing withdrawal groundwater m3 per country'] = intersections_regions.groupby('country')['withdrawal_manufacturing_sector_groundwater_m3'].transform('sum')

# %%%% Potential withdrawal from domestic sector

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_withdrawal_domestic_sector'] = (
    intersections_regions['potential_withdrawal_domestic_sector'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['withdrawal_domestic_sector_m3'] = (
    intersections_regions['cell_weighted_withdrawal_domestic_sector'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in thermoelectric sector per basin
intersections_regions['domestic withdrawal m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['withdrawal_domestic_sector_m3'].transform('sum')

# Water consumption in thermoelectric sector per region
intersections_regions['domestic withdrawal m3 per region'] = intersections_regions.groupby(['country', 'province'])['withdrawal_domestic_sector_m3'].transform('sum')

# Water consumption in thermoelectric sector per country
intersections_regions['domestic withdrawal m3 per country'] = intersections_regions.groupby('country')['withdrawal_domestic_sector_m3'].transform('sum')

# %%%% Potential withdrawal total

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_withdrawal_total'] = (
    intersections_regions['potential_withdrawal_total'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['withdrawal_total_m3'] = (
    intersections_regions['cell_weighted_withdrawal_total'] / 1000
    ) * intersections_regions['intersected_area']

# Water consumption in thermoelectric sector per basin
intersections_regions['withdrawal total m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['withdrawal_total_m3'].transform('sum')

# Water consumption in thermoelectric sector per region
intersections_regions['withdrawal total m3 per region'] = intersections_regions.groupby(['country', 'province'])['withdrawal_total_m3'].transform('sum')

# Water consumption in thermoelectric sector per country
intersections_regions['withdrawal total m3 per country'] = intersections_regions.groupby('country')['withdrawal_total_m3'].transform('sum')

# %%%% Runoff

# Total runoff per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_runoff'] = (
    intersections_regions['runoff_total'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['runoff_total_m3'] = (
    intersections_regions['cell_weighted_runoff']
    ) * intersections_regions['intersected_area'] / 1000

# Runoff per basin
intersections_regions['runoff m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['runoff_total_m3'].transform('sum')

# Runoffn per region
intersections_regions['runoff m3 per region'] = intersections_regions.groupby(['country', 'province'])['runoff_total_m3'].transform('sum')

# Runoff per country
intersections_regions['runoff m3 per country'] = intersections_regions.groupby('country')['runoff_total_m3'].transform('sum')

threshold_runoffc = intersections_regions['runoff m3 per country'].quantile(0.85)  # 85th percentile
threshold_runoffr = intersections_regions['runoff m3 per region'].quantile(0.85)  # 85th percentile
threshold_runoffb = intersections_regions['runoff m3 per basin'].quantile(0.85)  # 85th percentile

intersections_regions['runoff m3 per country'] = intersections_regions['runoff m3 per country'].clip(upper=threshold_runoffc)
intersections_regions['runoff m3 per region'] = intersections_regions['runoff m3 per region'].clip(upper=threshold_runoffr)
intersections_regions['runoff m3 per basin'] = intersections_regions['runoff m3 per basin'].clip(upper=threshold_runoffb)

# %%%% Storage

# Calculation the total water storage
intersections_regions['water_storage'] = (
                      intersections_regions['lake_storage_global'] +
                      intersections_regions['wetland_storage_global'] +
                      intersections_regions['reservoir_storage']
)/ 100  # Normalising by dividing by 100 - This means that water stocks can be used for at least 100 years, even if no renewability occurs

# Calculate the water storage per grid cell
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_water_storage'] = (
    intersections_regions['water_storage'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['water_storage_m3'] = (
    intersections_regions['cell_weighted_water_storage'] 
    ) * intersections_regions['intersected_area'] / 1000

# Water storage per basin
intersections_regions['storage m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['water_storage_m3'].transform('sum')

# Water storage per region
intersections_regions['storage m3 per region'] = intersections_regions.groupby(['country', 'province'])['water_storage_m3'].transform('sum')

# Water storage per country
intersections_regions['storage m3 per country'] = intersections_regions.groupby('country')['water_storage_m3'].transform('sum')

# %%%% Net water abstraction

# Sum surface and groundwater abstraction
intersections_regions['total_abstraction'] = (
    intersections_regions['abstraction_surface'] +
    intersections_regions['abstraction_groundwater']
    )

# Calculate the total abstraction per grid cell
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_abstraction'] = (
    intersections_regions['total_abstraction'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
intersections_regions['total_abstraction_m3'] = (
    intersections_regions['cell_weighted_abstraction'] 
    ) * intersections_regions['intersected_area'] / 1000

# Water abstraction per basin
intersections_regions['abstraction m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['total_abstraction_m3'].transform('sum')

# Water abstraction per region
intersections_regions['abstraction m3 per region'] = intersections_regions.groupby(['country', 'province'])['total_abstraction_m3'].transform('sum')

# Water abstraction per country
intersections_regions['abstraction m3 per country'] = intersections_regions.groupby('country')['total_abstraction_m3'].transform('sum')

# %%%% Precipitation

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_precipitation'] = (
    intersections_regions['precipitation'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
#intersections_regions['precipitation_m3'] = (
#    intersections_regions['cell_weighted_precipitation'] 
#    ) * intersections_regions['intersected_area'] / 1000

# Water consumption per basin
intersections_regions['precipitation mm per basin'] = intersections_regions.groupby(['country', 'main basins'])['cell_weighted_precipitation'].transform('sum')

# Water consumption per region
intersections_regions['precipitation mm per region'] = intersections_regions.groupby(['country', 'province'])['cell_weighted_precipitation'].transform('sum')

# Water consumption per country
intersections_regions['precipitation mm per country'] = intersections_regions.groupby('country')['cell_weighted_precipitation'].transform('sum')

# %%%% Actual evapotranspiration

# Consumpion per region
# Cell weighted values - fraction of the consumption per cell share
intersections_regions['cell_weighted_actual_evapotranspiration'] = (
    intersections_regions['actual_evapotranspiration'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
#intersections_regions['evapotranspiration_m3'] = (
#    intersections_regions['cell_weighted_withdrawal'] 
#    ) * intersections_regions['intersected_area'] / 1000

# Water consumption per basin
intersections_regions['actual evapotranspiration m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['cell_weighted_actual_evapotranspiration'].transform('sum')

# Water consumption per region
intersections_regions['actual evapotranspiration m3 per region'] = intersections_regions.groupby(['country', 'province'])['cell_weighted_actual_evapotranspiration'].transform('sum')

# Water consumption per country
intersections_regions['actual evapotranspiration m3 per country'] = intersections_regions.groupby('country')['cell_weighted_actual_evapotranspiration'].transform('sum')

# %%%% Potential evapotranspiration

# Consumpion per region
# Cell weighted values - fraction of the evapotranspiration per cell share
intersections_regions['cell_weighted_potential_evapotranspiration'] = (
    intersections_regions['potential_evapotranspiration'] * 
    intersections_regions['intersected_area_share']
)

# Compute volume (mm * area in m², converting mm to m by dividing by 1000)
#intersections_regions['evapotranspiration_m3'] = (
#    intersections_regions['cell_weighted_withdrawal'] 
#    ) * intersections_regions['intersected_area'] / 1000

# Potential evapotranspiration per basin
intersections_regions['potential evapotranspiration m3 per basin'] = intersections_regions.groupby(['country', 'main basins'])['cell_weighted_potential_evapotranspiration'].transform('sum')

# Potential evapotranspiration per region
intersections_regions['potential evapotranspiration m3 per region'] = intersections_regions.groupby(['country', 'province'])['cell_weighted_potential_evapotranspiration'].transform('sum')

# Potential evapotranspiration per country
intersections_regions['potential evapotranspiration m3 per country'] = intersections_regions.groupby('country')['cell_weighted_potential_evapotranspiration'].transform('sum')

# Adapt the adm code column --> change name to country code and consider only the 1st 3 letters
intersections_regions['country code'] = intersections_regions['adm_code'].str.slice(0, 3)

# Define manual corrections for the country codes
replaced_country_codes_intersections = {
    'WEB': 'PSE',
    'SDS': 'SSD',
    'KOS': 'XKX'
}

# Apply replacements
intersections_regions['country code'] = intersections_regions['country code'].replace(replaced_country_codes_intersections)

# %%%% groundwater recharge

# %%% BIER factors

# Import the BIER factors
BIER_runoff_gdf = gpd.read_file(
    './data/bier-runoff/BIER_runoff.shp'
    )

BIER_ann_average = BIER_runoff_gdf[['BIER_ru_12','geometry']] # Select only the annual average factor and coordinates column

# Ensure the combined GeoDataFrame has a consistent CRS
BIER_ann_average = BIER_ann_average.set_crs('EPSG:6933', allow_override=True)

intersections_regions_extra = gpd.sjoin_nearest(
    intersections_regions,    # Target polygons
    BIER_ann_average,         # Source points
    how='left', 
    distance_col='distance'   # Adds a column with the distance
)

# %%% Human Development Index (HDI)

HDI_data = pd.read_csv(
    './data/hdi/HDR23-24_Composite_indices_complete_time_series.csv',
    encoding='ISO-8859-1'
    )

# Extract the HDI
HDI = HDI_data[['iso3', 'country', 'hdi_2022', 'hdicode']].rename(columns={'iso3': 'country code'})

# Extract population
pop_countries = HDI_data[
    ['iso3', 'country', 'pop_total_2019', 'pop_total_2020', 'pop_total_2021', 'pop_total_2022']
    ].rename(columns={'iso3': 'country code'})

# %%% PPI and WGI

PPI_WGI = pd.read_csv(
    './data/ppi/PPI_final.csv',
    encoding='ISO-8859-1'
    )

# Add country codes
PPI_WGI['country code'] = cc.convert(names=PPI_WGI['country'], to='ISO3')

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
PPI_WGI['country code'] = PPI_WGI['country code'].replace(replaced_country_codes)

# %%% GDP from World Bank national accounts data, and OECD National Accounts data files

# GDP is also given in the NED data set
#GDP = pd.read_csv(
#    './data/gdp/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_9.csv',
#    encoding='ISO-8859-1'
#    )

#GDP = GDP[['ï»¿Country Name', 'Country Code', '2023', '2022', '2021', '2020']].rename(
#    columns={'ï»¿Country Name': 'country', 'Country Code': 'country code','2023': 'gdp 2023', '2022': 'gdp 2022', '2021': 'gdp 2021', '2020': 'gdp 2020'}
#    )

#rows_with_all_nans = GDP[GDP.iloc[:, -4:].isna().all(axis=1)]

# Take the most recent GDP data
#GDP['gdp'] = GDP[['gdp 2023', 'gdp 2022', 'gdp 2021', 'gdp 2020']].bfill(axis=1).iloc[:, 0]

# Import gdp data
gdp_undata = pd.read_csv('./data/gdp/Undata/UNdata_Export_20250608_220427436.csv')
gdp_percentage = pd.read_csv('./data/gdp/4.2_Structure_of_value_added.csv')

# Select only the GDP variable
gdp = gdp_undata[gdp_undata['Item'] == 'Gross Domestic Product (GDP)']

# Rename columns
gdp = gdp.rename(
    columns={'Country or Area': 'country', 'Value': 'gdp', 'Year':'year'}
    ).drop('Item', axis=1)

# Select the most recent year for each country
gdp = gdp.loc[
    gdp.groupby('country')['year'].idxmax()
].reset_index(drop=True)

gdp['country'] = gdp['country'].replace({
    'TÃ¼rkiye': 'Türkiye',
    'CuraÃ§ao': 'Curaçao'
})

# Adapt the adm code column --> change name to country code and consider only the 1st 3 letters
gdp['country code'] = cc.convert(names=gdp['country'], to='ISO3')

gdp_percentage['country'] = gdp_percentage['country'].replace({
    'TÃ¼rkiye': 'Türkiye',
    'CuraÃ§ao': 'Curaçao'
})

# Adapt the adm code column --> change name to country code and consider only the 1st 3 letters
gdp_percentage['country code'] = cc.convert(names=gdp_percentage['country'], to='ISO3')

gdp = gdp.merge(
    gdp_percentage[['country code','agriculture %','industry %','manufacturing %','services %']],
    how='left',
    on='country code'
)

gdp['argiculture gdp'] = (gdp['agriculture %'] / 100) * gdp['gdp']
gdp['manufacturing gdp'] = (gdp['manufacturing %'] / 100) * gdp['gdp']

# %%% OECD indicators

OECD = pd.read_csv(
    './data/oecd/OECD,DF_DP_LIVE,+all.csv',
    encoding='ISO-8859-1'
    )

# List of indicators to keep
OECD_selected_indicators = [
    'Value added by activity',
    'Water withdrawals',
    'Wastewater treatment',
    'GDP',
    'Population',
    'Household disposable income'
    ]

OECD_selected_years = [
    '2022',
    '2021',
    '2020',
    '2019',
    '2018'
    ]

# Filter rows where Indicator is in the selected list
filtered_OECD = OECD[
    (OECD['Indicator'].isin(OECD_selected_indicators)) &
    (OECD['TIME_PERIOD'].isin(OECD_selected_years))
].drop(columns=[
    'STRUCTURE',
    'STRUCTURE_ID',
    'STRUCTURE_NAME',
    'INDICATOR',
    'SUBJECT',
    'FREQUENCY',
    'Time',
    'Observation Value',
    'OBS_STATUS',
    'Observation Status',
    'UNIT_MEASURE',
    'Unit of Measures',
    'UNIT_MULT',
    'Multiplier',
    'BASE_PER',
    'Base reference period'
])

# %%% World Bank indicators

WB = pd.read_csv(
    './data/wdi/WDICSV.csv',
    encoding='ISO-8859-1'
    )

# List of indicators to keep
WB_selected_indicators = [
    'Investment in water and sanitation with private participation'
    ' (current US$)',
    'Water withdrawals',
    'Level of water stress:'
    ' freshwater withdrawal as a proportion of available freshwater resources',
    'People using at least basic drinking water services (% of population)',
    'People using at least basic drinking water services, urban'
    ' (% of urban population)',
    'People using safely managed drinking water services (% of population)',
    'People with basic handwashing facilities including soap and water'
    ' (% of population)',
    'People with basic handwashing facilities including soap and water, rural'
    ' (% of rural population)',
    'People with basic handwashing facilities including soap and water, urban'
    ' (% of urban population)',
    'Public private partnerships investment in water and sanitation'
    ' (current US$)',
    'Investment in water and sanitation with private participation'
    ' (current US$)',
    'Public private partnerships investment in water and sanitation'
    ' (current US$)',
    'Level of water stress: freshwater withdrawal as a proportion of'
    ' available freshwater resources',
    'Agriculture, forestry, and fishing, value added (current US$)',
    'Industry (including construction), value added (current US$)',
    'Manufacturing, value added (current US$)',
    'Annual freshwater withdrawals, agriculture'
    ' (% of total freshwater withdrawal)',
    'Annual freshwater withdrawals, domestic'
    ' (% of total freshwater withdrawal)',
    'Annual freshwater withdrawals, industry'
    ' (% of total freshwater withdrawal)',
    'Adjusted net national income per capita (current US$)',
    'GNI per capita (current LCU)',
    'Population, total'
    ]

filtered_WB = WB[WB['Indicator Name'].isin(WB_selected_indicators)]
filtered_WB = filtered_WB[['Country Name', 'Country Code', 'Indicator Name', '2016','2017','2018','2019', '2020', '2021', '2022', '2023']]
# Fill NaN values across all columns with 0
#filtered_WB = filtered_WB.fillna(0)

# Convert the  year columns into rows
df_long = filtered_WB.melt(
    id_vars=['Country Name', 'Country Code', 'Indicator Name'], 
    var_name='Year', 
    value_name='Value'
)

# Pivot so Indicator Name becomes column headers
WB_final = df_long.pivot_table(
    index=['Country Name', 'Country Code', 'Year'], 
    columns='Indicator Name', 
    values='Value'
    ).reset_index()

# Remane columns
WB_final = WB_final.rename(columns={'Country Name': 'country', 'Country Code': 'country code',
         'Adjusted net national income per capita (current US$)': 'net income',
         'Year': 'year', 'Population, total': 'population'})

# Value added
# Agriculture

WB_value_added_agriculture = pd.read_csv(
    './data/wdi/API_NV.AGR.TOTL.CD_DS2_en_csv_v2_24046.csv',
    encoding='ISO-8859-1'
    )

WB_value_added_agriculture = WB_value_added_agriculture.rename(
    columns={'ï»¿Country Name': 'country', 'Country Code': 'country code'}
    ).drop(columns='Indicator Code')

#value_added_agriculture = WB_value_added_agriculture[
#    ['country', 'country code', 'Indicator Name', '2016','2017','2018','2019', '2020', '2021', '2022', '2023']
#    ]

# Melt the DataFrame (convert years into rows)
value_added_agriculture = WB_value_added_agriculture.melt(
    id_vars=['country', 'country code', 'Indicator Name'], 
    var_name='Year', 
    value_name='Value'
)

# Sort by country and Year (ascending or descending to ensure we find the most recent year with a value)
value_added_agriculture = value_added_agriculture.sort_values(by=['country', 'Year'], ascending=[True, False])

# Drop rows where 'Value' is NaN
value_added_agriculture = value_added_agriculture.dropna(subset=['Value'])

# Keep only the first (most recent) value per country (and indicator, if needed)
value_added_agriculture = value_added_agriculture.groupby(['country', 'Indicator Name'], as_index=False).first()

# Create a new column with the 'Value' column values renamed
value_added_agriculture['agriculture value added'] = value_added_agriculture['Value']

# Remove the 'Value' column as it's no longer needed
value_added_agriculture = value_added_agriculture.drop(columns={'Value', 'Indicator Name'})

# Manufacturing

WB_value_added_manufacturing = pd.read_csv(
    './data/wdi/API_NV.IND.MANF.CD_DS2_en_csv_v2_21511.csv',
    encoding='ISO-8859-1'
    )

WB_value_added_manufacturing = WB_value_added_manufacturing.rename(
    columns={'ï»¿Country Name': 'country', 'Country Code': 'country code'}
    ).drop(columns='Indicator Code')

value_added_manufacturing = WB_value_added_manufacturing[
    ['country', 'country code', 'Indicator Name', '2016','2017','2018','2019', '2020', '2021', '2022', '2023']
    ]

# Melt the DataFrame (convert years into rows)
value_added_manufacturing = value_added_manufacturing.melt(
    id_vars=['country', 'country code', 'Indicator Name'], 
    var_name='Year', 
    value_name='Value'
)

# Sort by country and Year (ascending or descending to ensure we find the most recent year with a value)
value_added_manufacturing = value_added_manufacturing.sort_values(by=['country', 'Year'], ascending=[True, False])

# Drop rows where 'Value' is NaN
value_added_manufacturing = value_added_manufacturing.dropna(subset=['Value'])

# Keep only the first (most recent) value per country (and indicator, if needed)
value_added_manufacturing = value_added_manufacturing.groupby(['country', 'Indicator Name'], as_index=False).first()

# Create a new column with the 'Value' column values renamed
value_added_manufacturing['manufacturing value added'] = value_added_manufacturing['Value']

# Remove the 'Value' column as it's no longer needed
value_added_manufacturing = value_added_manufacturing.drop(columns={'Value', 'Indicator Name'})

# %%% Gross value added UNdata

# UNdata GVA
value_added_UN = pd.read_csv(
    './data/undata/UNdata_Export_20250608_155903265_total.csv',
    encoding='ISO-8859-1'
    )

# Filter to a specific year (e.g., 2023)
filtered_year = value_added_UN[value_added_UN['Year'] == 2023]

# Pivot: country as index temporarily
value_added = filtered_year.pivot_table(
    index=['Country or Area', 'Year'],
    columns='Item',
    values='Value',
    aggfunc='first'
).reset_index()

# Select only relevant columns
value_added = value_added[[
    'Country or Area',
    'Year',
    'Agriculture, hunting, forestry, fishing (ISIC A-B)',
    'Manufacturing (ISIC D)',
    'Total Value Added',
]].copy()

# Change columns' names
value_added = value_added.rename(columns={
    'Country or Area': 'country',
    'Year': 'year',
    'Agriculture, hunting, forestry, fishing (ISIC A-B)': 'agriculture gva',
    'Manufacturing (ISIC D)': 'manufacturing gva',
    'Total Value Added': 'total gva'
})

# Fix country names
value_added['country'] = value_added['country'].replace({
    'CuraÃ§ao': 'Curacao',
    'TÃ¼rkiye': 'Turkey',
    'United Republic of Tanzania: Zanzibar': 'Tanzania'
})

# Add country codes
value_added['country code'] = cc.convert(names=value_added['country'], to='ISO3')

# Apply replacements
value_added['country code'] = value_added['country code'].replace(replaced_country_codes)

# %%% Employment

# ILOSTAT employment data 
employment_data = pd.read_csv(
    './data/employment/EMP_TEMP_SEX_STE_ECO_NB_A-20250608T2223.csv',
    dtype={'obs_value': float},      # ensure this is treated as numeric
    low_memory=False                 # avoids chunk-based dtype guessing
)

# Load the additional file
employment_additional = pd.read_csv(
    './data/employment/Employment_additional.csv',
    dtype={'obs_value': float},
    low_memory=False
)

# Combine both DataFrames by stacking rows
combined_employment = pd.concat([employment_data, employment_additional], ignore_index=True)

employment = combined_employment[[
    'ref_area.label',
    'sex.label',
    'classif1.label',
    'classif2.label',
    'time',
    'obs_value'
]].copy()

# Clean column names
employment.columns = employment.columns.str.strip().str.lower().str.replace('.', '_')

# Ensure time is numeric (if it's a string)
employment['time'] = pd.to_numeric(employment['time'], errors='coerce')

# Get the most recent year per country
most_recent_years = (
    employment.groupby('ref_area_label')['time']
    .max()
    .reset_index()
    .rename(columns={'time': 'year'})
)

# Keep rows for most recent year
employment_recent = employment.merge(
    most_recent_years,
    on='ref_area_label'
).query('time == year')

# Filter for Total only in sex_label
employment_total = employment_recent[
    employment_recent['sex_label'] == 'Total'
]

broad_sector_data = employment_total[
    employment_total['classif2_label'].str.startswith('Economic activity (Broad sector):')
]

# Pivot only the filtered data
employment_broad = broad_sector_data.pivot_table(
    index=['ref_area_label', 'year'],
    columns='classif2_label',
    values='obs_value',
    aggfunc='first'
).reset_index()

# Select only relevant columns
employment = employment_broad[[
    'ref_area_label',
    'year',
    'Economic activity (Broad sector): Agriculture',
    'Economic activity (Broad sector): Industry',
    'Economic activity (Broad sector): Total',
]].copy()

#aggr_sector_data = employment_total[
#    employment_total['classif2_label'].str.startswith('Economic activity (Aggregate):')
#]

# Pivot only the filtered data
#employment_aggr = aggr_sector_data.pivot_table(
#    index=['ref_area_label', 'year'],
#    columns='classif2_label',
#    values='obs_value',
#    aggfunc='first'
#).reset_index()

# Select only relevant columns
#employment = employment_aggr[[
#    'ref_area_label',
#    'year',
#    'Economic activity (Aggregate): Agriculture',
#    'Economic activity (Aggregate): Manufacturing',
#    'Economic activity (Aggregate): Total',
#]].copy()

# Change columns' names
employment = employment.rename(columns={
    'ref_area_label': 'country',
    'Economic activity (Broad sector): Agriculture': 'agriculture employment',
    'Economic activity (Broad sector): Industry': 'manufacturing employment',
    'Economic activity (Broad sector): Total': 'total employment'
})

# Add the UN data 
extra_data_employment = pd.read_csv(
    './data/employment/UNdata_employment.csv',
    dtype={'obs_value': float},      # ensure this is treated as numeric
    low_memory=False                 # avoids chunk-based dtype guessing
)

# Standardise column names
extra_data_employment.columns = extra_data_employment.columns.str.strip().str.lower().str.replace('.', '', regex=False)

# Rename columns to a consistent format
new_data_employment = extra_data_employment.rename(columns={
    'country or area': 'country',
    'year': 'year',
    'subclassification': 'sector',
    'value': 'value'
})

# Filter only 'Total men and women'
new_data_employment = new_data_employment[new_data_employment['sex'] == 'Total men and women']

# Pivot by sector, keeping country and year
employment_new = new_data_employment.pivot_table(
    index=['country', 'year'],
    columns='sector',
    values='value',
    aggfunc='first'
).reset_index()

# Columns for sectors
sectors = ['Agriculture, Hunting and Forestry', 'Manufacturing', 'Total']

# DataFrame with your data
# employment_new = pd.read_csv(...) or however you get your data

rows = []

for country, group in employment_new.groupby('country'):
    for sector in sectors:
        # Filter rows where sector value is not NaN
        valid_rows = group[group[sector].notna()]
        if not valid_rows.empty:
            # Find row with the max year for that sector
            most_recent_row = valid_rows.loc[valid_rows['year'].idxmax()]
            rows.append(most_recent_row)

# Combine all selected rows
employment_new = pd.DataFrame(rows).drop_duplicates()

# Reset index
employment_new = employment_new.reset_index(drop=True)

# Change columns' names
employment_new = employment_new.rename(columns={
    'Agriculture, Hunting and Forestry': 'agriculture employment',
    'Manufacturing': 'manufacturing employment',
    'Total': 'total employment'
})

# Get the most recent non-NaN value per sector per country
employment_new = (
    employment_new
    .sort_values('year', ascending=False)
    .groupby('country', as_index=False)
    .agg(lambda x: x.dropna().iloc[0] if x.dropna().size else np.nan)
)

# Identify countries missing in employment
missing_countries = set(employment_new['country']) - set(employment['country'])

# Filter employment_new for only missing countries
employment_new_missing = employment_new[employment_new['country'].isin(missing_countries)]

# Concatenate the original employment with the missing countries from employment_new
combined_employment = pd.concat([employment, employment_new_missing], ignore_index=True)

# Sort alphabetically by country
combined_employment = combined_employment.sort_values(by='country')

# Reset the index (drop=True to avoid adding the old index as a column)
combined_employment = combined_employment.reset_index(drop=True)

# Fix ambiguous country names before converting
combined_employment['country'] = combined_employment['country'].replace({
    'Macau, China': 'Macau'}
)

# Add country codes
combined_employment['country code'] = cc.convert(names=combined_employment['country'], to='ISO3')

# Apply replacements
combined_employment['country code'] = combined_employment['country code'].replace(replaced_country_codes)

# Multiply by hundred
combined_employment['agriculture employment'] = combined_employment['agriculture employment'] * 1000
combined_employment['manufacturing employment'] = combined_employment['manufacturing employment'] * 1000
combined_employment['total employment'] = combined_employment['total employment'] * 1000

# %%% Aquastat

# AQUASTAT wastewater
AQUASTAT_wastewater = pd.read_csv(
    './data/aquastat/AQUASTAT+wastewater.csv'
    )

AQUASTAT_resources = pd.read_csv(
    './data/aquastat/AQUASTAT+Water+resources+and+use.csv')

AQUASTAT_external = pd.read_csv(
    './data/aquastat/AQUASTAT+external+renewable+water+resources.csv'
    )

# List of indicators to keep
AQUASTAT_selected_indicators = [
    'Groundwater: entering the country (total)', # 10^9 m3/year
    'Surface water: accounted flow of border rivers',
    'Surface water: entering the country (total)',
    'Surface water: inflow not submitted to treaties',
    'Surface water: inflow submitted to treaties',
    'Total dam capacity', # km3
    'Dam capacity per capita', # m3/inhab
    'Surface water: accounted flow of border rivers',
    'Surface water produced internally',
    'Dependency ratio', # unit % 
    'Groundwater produced internally', # 10^9 m3/year
    'Surface water produced internally' # 10^9 m3/year
    ]

AQUASTAT = AQUASTAT_resources[
    AQUASTAT_resources['variable'].isin(AQUASTAT_selected_indicators)
    ]

#AQUASTAT_external[
#    AQUASTAT_external['variable'].isin(AQUASTAT_selected_indicators)
#    ]

# Pivot the data so 'Variable' becomes column headers and 'Value' becomes the data
aquastat = AQUASTAT.pivot_table(
    index=['country', 'year'],  # Keep 'Area' and 'Year' as row identifiers
    columns='variable',  # Turn 'Variable' values into column headers
    values='value'  # Values from 'Value' column go into new columns
).reset_index()

# Rename columns to remove multi-index
aquastat.columns.name = None

# Convert 'Year' column to integer (if not already)
aquastat['year'] = aquastat['year'].astype(int)

# Keep only the most recent year per country
aquastat = aquastat.loc[aquastat.groupby('country')['year'].idxmax()].reset_index(drop=True)

# Add country codes
aquastat['country code'] = cc.convert(names=aquastat['country'], to='ISO3')

# Apply replacements of country codes
aquastat['country code'] = aquastat['country code'].replace(replaced_country_codes)

# Set NaNs in the column to 0
aquastat[['Dam capacity per capita', 'Dependency ratio']] = aquastat[['Dam capacity per capita', 'Dependency ratio']].fillna(0) # Set NaNs in the column to 0

# %% Conflict or cooperation database

CC = pd.read_excel(
    './data/conflict_cooperation/'
    'WARICC+dataset+v10.xlsx'
    )

conflict_cooperation = CC[[
    'cname',
    'lat_coordin',
    'long_coordin',
    'wes'
    ]]

conflict_cooperation = conflict_cooperation.copy()  # Make a copy to avoid warnings
conflict_cooperation['wes country'] = conflict_cooperation.groupby('cname')['wes'].transform('mean')

# Sum abstraction by basin
conflict_cooperation = conflict_cooperation.groupby('cname', as_index=False).agg({
    'wes': 'mean',  # Sum the abstraction column
    'lat_coordin': 'first',
    'long_coordin': 'first'
    })

# Add new column with ISO3 codes
conflict_cooperation['country code'] = cc.convert(names=conflict_cooperation['cname'], to='ISO3')

# Apply replacements of country codes
conflict_cooperation['country code'] = conflict_cooperation['country code'].replace(replaced_country_codes)

# Set NaNs in the column to 0
conflict_cooperation['wes'] = conflict_cooperation['wes'].fillna(0) # Set NaNs in the column to 0

# %% Water tariff

# Get a list of all CSV files in the folder
tariff_files = glob.glob('./data/tariff/*.csv')

# Read and combine all CSV files into one DataFrame
tariff = (
    pd.concat([pd.read_csv(file) for file in tariff_files], ignore_index=True)
    .iloc[:, :-1]  # Drop the last column
    .rename(columns={'City': 'city', 'Country': 'country'})  # Rename 'City' to 'city'
    .drop(columns=['Utility - City (15 m3)', 'Utility'], errors='ignore')
#   .loc[lambda df: df['Service'] == 'Water']  # Filter only 'Water'
    .loc[lambda df: df['Date'].astype(str).str.contains(r'\b\d{4}\b', na=False)]  # Keep only 4-digit years
    )

# Delete the last 4 columns
tariff = tariff.iloc[:, :-4]

# Pivot table to get all services side-by-side
tariff = tariff.pivot_table(
    index=['country', 'Date'],
    columns='Service',
    values='1 m3',  # <-- Replace with your actual column name if different
    aggfunc='first'
).reset_index()

# Sort the DataFrame by Date (most recent first)
tariff = tariff.sort_values(by='Date', ascending=False)

# Filter out rows where all values in 'Wastewater', 'Water', and 'Water and Wastewater' are NaN or 0
tariff = tariff[(tariff[['Wastewater', 'Water', 'Water and Wastewater']].notna()).any(axis=1)]
tariff = tariff[(tariff[['Wastewater', 'Water', 'Water and Wastewater']] != 0).any(axis=1)]

# Function to filter rows with valid non-zero and non-NaN values
def filter_valid_rows(group):
    # Remove rows where all values in 'Wastewater', 'Water', and 'Water and Wastewater' are zero or NaN
    valid_rows = group[
        ~((group[['Wastewater', 'Water', 'Water and Wastewater']] == 0) | 
           (group[['Wastewater', 'Water', 'Water and Wastewater']].isna())).all(axis=1)
    ]
    
    # Return the most recent valid row for this group (if any valid row exists)
    return valid_rows.iloc[0] if not valid_rows.empty else None

# Separate the 'Country' column before applying the function
grouped_tariff = tariff.groupby('country', group_keys=False)

# Apply the function to each group
filtered_rows = []

# Iterate through the groups manually to avoid the deprecation warning
for _, group in grouped_tariff:
    valid_row = filter_valid_rows(group)
    if valid_row is not None:
        filtered_rows.append(valid_row)

# Combine the results into a new DataFrame
tariffs = pd.DataFrame(filtered_rows)

# Reset index to make the result a clean DataFrame
tariffs = tariffs.reset_index(drop=True)

# Add total_tariff column
tariffs['total_tariff'] = tariffs['Water and Wastewater']

# Where "Water and Wastewater" is missing, use water + wastewater
missing_combo = tariffs['total_tariff'].isna()
tariffs.loc[missing_combo, 'Wastewater'] = tariffs.loc[missing_combo, 'Wastewater'].fillna(
    tariffs.loc[missing_combo, 'Water']
)
tariffs.loc[missing_combo, 'Water'] = tariffs.loc[missing_combo, 'Water'].fillna(
    tariffs.loc[missing_combo, 'Wastewater']
)
tariffs.loc[missing_combo, 'total_tariff'] = (
    tariffs.loc[missing_combo, 'Water'] + tariffs.loc[missing_combo, 'Wastewater']
)

# Define manual corrections
corrections = {
    'Djubuti': 'Djibouti',
    'Hawai': 'United States',
    'Island': 'Iceland',
    'Northern Ireland': 'United Kingdom',
    'Timor-Lest': 'Timor-Leste',
    'Transnistria': 'Moldova',  # Technically part of Moldova
}

# Apply corrections
tariffs['country'] = tariffs['country'].replace(corrections)

# Add new column with ISO3 codes
tariffs['country code'] = cc.convert(names = tariffs['country'], to='ISO3')

# Apply replacements of country codes
tariffs['country code'] = tariffs['country code'].replace(replaced_country_codes)

# %% Income

# Data from the International Labour Organization (https://rshiny.ilo.org/dataexplorer03/?lang=en&segment=indicator&id=EAR_4MMN_CUR_NB_A)
income = pd.read_csv(
    './data/income/EAR_4MMN_CUR_NB_A-filtered-2025-04-16.csv'
    )

income = (
    income.rename(columns={'ref_area.label': 'country', 'classif1.label': 'currency', 'obs_value': 'statutory nominal gross monthly minimum wage'})  # Rename columns
          .drop(columns={'source.label', 'note_indicator.label', 'note_source.label'}, errors='ignore')  # Drop unnecessary column if exists
          .loc[lambda df: df['currency'] == 'Currency: U.S. dollars']  # Optional: Filter only 'Water'
)

# Keep the most recent row per country
income = income.loc[income.groupby('country')['time'].idxmax()].reset_index()

# Delete the index column
income = income.drop(columns=['index'])

# Add new column with ISO3 codes
income['country code'] = cc.convert(names=income['country'], to='ISO3')

# Apply replacements of country codes
income['country code'] = income['country code'].replace(replaced_country_codes)

# %% Wastewater treatment

# Domestic, thermoelectric and manufacturing water use components [km³/year]
# according to 26 Shiklomanov regions (Shiklomanov, 2000)
# between 1950 and 2010 (in an interval of 10 years).
# (County-level data are available on request)
Water_treatment_regions = pd.read_csv(
    './data/wastewater/Water treatment_regions.csv'
    )

# %% EPI

# Directory containing the files
directory_epi = r'C:\Users\Sylvia\Desktop\Work\PhD 2023\Python\sil-new\data\epi'

# List of all the files (existing and new)
epi_ww_files = [
    'WWT_ind_na.csv',
    'WWR_ind_na.csv',
    'WWC_ind_na.csv',
    'WWG_ind_na.csv'
]

epi_env_files = [
    'BER_ind_na.csv',
    'MKP_ind_na.csv',
    'MHP_ind_na.csv',
    'MPE_ind_na.csv',
    'PAR_ind_na.csv',
    'SPI_ind_na.csv',
    'TBN_ind_na.csv',
    'TKP_ind_na.csv',
    'PAE_ind_na.csv',
    'PHL_ind_na.csv',
    'RLI_ind_na.csv',
    'SHI_ind_na.csv',
    'HPE_ind_na.csv',
    'HFD_ind_na.csv',
    'OZD_ind_na.csv',
    'NOD_ind_na.csv',
    'SOE_ind_na.csv',
    'COE_ind_na.csv',
    'VOE_ind_na.csv',
]

# Function to read and process the files
def process_files(files, sector_name):
    df_list = []
    for file in files:
        file_path = os.path.join(directory_epi, file)
        
        # Check if the file exists
        if os.path.isfile(file_path):
            print(f"Reading file: {file_path}")
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            
            # Identify the dynamic column (i.e., the indicator column)
            dynamic_col = [col for col in df.columns if "ind.2024" in col]
            
            if dynamic_col:
                df = df[['iso', 'country'] + dynamic_col]  # Select iso, country, and the indicator column
                
                # Rename the indicator column to include the file name (e.g., 'WWT.ind.2024' -> 'ww treated')
                df = df.rename(columns={dynamic_col[0]: f"{dynamic_col[0]}_{file.replace('.csv', '')}"})
                df_list.append(df)
        else:
            print(f"File {file_path} not found!")
    
    # Concatenate all DataFrames from the list into a single DataFrame
    if df_list:
        combined_df = pd.concat(df_list, axis=0, ignore_index=True)
        print(f"{sector_name} files merged successfully.")
    else:
        print(f"No valid files found for {sector_name}.")
        combined_df = pd.DataFrame()

    return combined_df

# Process both lists of files (epi_ww and epi_env) using the function
epi_ww = process_files(epi_ww_files, 'Wastewater Indicators')
epi_env = process_files(epi_env_files, 'Environmental Indicators')

# Merge the data on 'iso' and 'country' (if necessary) to ensure one row per country with all indicators
epi_ww = epi_ww.groupby(['iso', 'country'], as_index=False).first()
epi_env = epi_env.groupby(['iso', 'country'], as_index=False).first()

# Rename columns for clarity
epi_ww = epi_ww.rename(columns={
    'WWG.ind.2024_WWG_ind_na': 'ww generated',
    'WWC.ind.2024_WWC_ind_na': 'ww collected',
    'WWT.ind.2024_WWT_ind_na': 'ww treated',
    'WWR.ind.2024_WWR_ind_na': 'ww reused',
    'iso': 'country code'
})

epi_env = epi_env.rename(columns={
    'BER.ind.2024_BER_ind_na': 'bioclimatic ecosystem resilience',  # New column for BER
    'MKP.ind.2024_MKP_ind_na': 'marine kba protection',
    'MHP.ind.2024_MHP_ind_na': 'marine habitat protection',
    'MPE.ind.2024_MPE_ind_na': 'marine protection stringency',
    'PAR.ind.2024_PAR_ind_na': 'protected areas representativeness index',
    'SPI.ind.2024_SPI_ind_na': 'species protection index',
    'TBN.ind.2024_TBN_ind_na': 'terrestrial biome protection (national weights)',
    'TKP.ind.2024_TKP_ind_na': 'terrestrial kba protection',
    'PAE.ind.2024_PAE_ind_na': 'protected area effectiveness',
    'PHL.ind.2024_PHL_ind_na': 'protected human land',
    'RLI.ind.2024_RLI_ind_na': 'red list index',
    'SHI.ind.2024_SHI_ind_na': 'species habitat index',
    'HPE.ind.2024_HPE_ind_na': 'anthropogenic PM2.5 exposure',
    'HFD.ind.2024_HFD_ind_na': 'household solid fuels',
    'OZD.ind.2024_OZD_ind_na': 'ozone exposure',
    'NOD.ind.2024_NOD_ind_na': 'NO2 exposure',
    'SOE.ind.2024_SOE_ind_na': 'SO2 exposure',
    'COE.ind.2024_COE_ind_na': 'CO exposure',
    'VOE.ind.2024_VOE_ind_na': 'VOC exposure',
    'iso': 'country code'
})

# %% Pfister data --> Assessing the environmental impacts of freshwater consumption in LCA

# Data from the paper (Pfister et al., 2009)
ecosystem_quality = pd.read_csv(
    './data/social/Factors EI99HA-points.csv'
    )

ecosystem_quality = ecosystem_quality.rename(columns={
    'Country code (ISO 3166-1)': 'country code'  # New column for BER
})

# %% Population

# Import the population data
population_countries = pd.read_csv('./data/population/Undata/UNdata_Export_20250612_120552933_total.csv')

# Rename 'ADMIN' to 'country'
population_countries = population_countries.rename(
    columns={'Country or Area': 'country', 'Value': 'population', 'Year(s)':'year'}
    ).drop('Variant', axis=1)

# Select the most recent year for each country
population = population_countries.loc[
    population_countries.groupby('country')['year'].idxmax()
].reset_index(drop=True)

population['country'] = population['country'].replace({
    'TÃ¼rkiye': 'Türkiye',
    'CuraÃ§ao': 'Curaçao'
})

# Adapt the adm code column --> change name to country code and consider only the 1st 3 letters
population['country code'] = cc.convert(names=population['country'], to='ISO3')

# Multiply by 1000 since the dataset is available in thousands
population['population'] = population['population'] * 1000

# %% Desalination

# Import the population data
desalination = pd.read_csv('./data/desalination/desalination.csv')

desalination['country'] = desalination['country'].replace({
    'TÃ¼rkiye': 'Türkiye'
})

# Adapt the adm code column --> change name to country code and consider only the 1st 3 letters
desalination['country code'] = cc.convert(names=desalination['country'], to='ISO3')

# %% Merge the indicators to the main geodataframe

# Reorder columns: move 'country code' and 'country' to the front
cols = ['country', 'country code'] + [col for col in intersections_regions.columns if col not in ['country', 'country code']]
intersections_regions = intersections_regions[cols]

intersections_regions_extra = intersections_regions.merge(
    gdp[['country code', 'argiculture gdp', 'manufacturing gdp']],
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    PPI_WGI[['country code', 'ppi', 'wgi']],
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    value_added[['country code',
                 'agriculture gva',
                 'manufacturing gva',
                 'total gva'
                 ]],
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    combined_employment[['country code',
                         'agriculture employment',
                         'manufacturing employment',
                         'total employment'
                         ]],
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    conflict_cooperation[['country code', 'wes']],
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    income[['country code', 'statutory nominal gross monthly minimum wage']],
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    aquastat[['country code', 'Dependency ratio', 'Dam capacity per capita']],
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    tariffs[['country code', 'total_tariff']], 
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    epi_ww[[
        'country code', 
        'ww generated',
        'ww collected',
        'ww treated',
        'ww reused'
        ]],
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    epi_env[[
        'country code', 
        'bioclimatic ecosystem resilience',  # New column for BER
        'marine kba protection',
        'marine habitat protection',
        'marine protection stringency',
        'protected areas representativeness index',
        'species protection index',
        'terrestrial biome protection (national weights)',
        'terrestrial kba protection',
        'protected area effectiveness',
        'protected human land',
        'red list index',
        'species habitat index',
        'anthropogenic PM2.5 exposure',
        'household solid fuels',
        'ozone exposure',
        'NO2 exposure',
        'SO2 exposure',
        'CO exposure',
        'VOC exposure',
        ]],
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    population[['country code', 'population']], 
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    ecosystem_quality[['country code', 'Ecosystem quality [m2•yr/m3]', 'Ecosystem quality']], 
    how='left',
    on='country code'
)

intersections_regions_extra = intersections_regions_extra.merge(
    desalination[['country code', 'desalination capacity (m3/year) 10E9']], 
    how='left',
    on='country code'
)

#%% Last changes on the indicators files
# File with only WaterGAP indicators
# Convert to a GeoDataFrame
intersections_regions = gpd.GeoDataFrame(intersections_regions, geometry='geometry', crs='EPSG:6933')
intersections_regions = intersections_regions.loc[:, ~intersections_regions.columns.duplicated(keep='last')]

intersections_regions.to_file('intersections_regions.gpkg', layer='indicators', driver='GPKG') # Save output

# File with WaterGAP indicators + the rest
# Convert to a GeoDataFrame
intersections_regions_extra = gpd.GeoDataFrame(intersections_regions_extra, geometry='geometry', crs='EPSG:6933')
intersections_regions_extra = intersections_regions_extra.loc[:, ~intersections_regions_extra.columns.duplicated(keep='last')]

intersections_regions_extra.to_file('intersections_regions_extra.gpkg', layer='indicators', driver='GPKG') # Save output

#%% Plot variations

# Aggregate water consumption per province
#bg_consumption = (intersections_regions[intersections_regions['country'] == 'China']
#                 .groupby('province')['consumption_m3']
#                  .sum()
#                  .sort_values(ascending=False))


#bg_consumption.plot(column='consumption_m3', legend=True)
#bg_consumption.plot(column='consumption m3 per region', legend=True)
