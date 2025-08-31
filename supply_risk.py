"""
Water SCARCE Criticality Assessment Script – Supply Risk Dimension
----------------------------------

This script calculates the water supply risk dimension of Water SCARCE. 
The supply risk reflects both physical and socio-economic availability of freshwater 
resources and is structured into eight categories.

Physical availability sub-dimension includes:
    - Water depletion index

Socio-economic availability sub-dimension include:
    - Concentration of water production
    - Concentration of water reserves
    - Water demand growth
    - Water exploration
    - Water governance
    - Water affordability
    - Water reuse and recycling

The calculations follow the Water SCARCE framework (Marinova et al., 2025) with modifications.

Main outputs:
    - Scaled values for each category (0–1)
    - Aggregated water supply risk score for use in the overall criticality assessment
    - Map with global Water Deprivation Index (WDI) at the grid-cell level. 
    - Map with global supply risk distribution on regional and country level

"""

import country_converter as coco
import geopandas as gpd
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib import pyplot as plt
from shapely.geometry import mapping
from sklearn.preprocessing import MinMaxScaler

# Initialize the country converter
cc = coco.CountryConverter()

# %% Categories calculation

# ---- WATER SUPPLY RISK

# Load the GeoPackage
intersections_regions_extra = gpd.read_file(
    'intersections_regions_extra.gpkg',
    layer='indicators'
    )

intersections_regions = gpd.read_file(
    'intersections_regions.gpkg',
    layer='indicators'
    )

# Remove rows where the country column has empty value
intersections_regions_extra = intersections_regions_extra.dropna(
    subset=['country']
    )

# Set a scaler to scale the results between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

# %%%  Physical availability

# ---- WATER DEPLETION

# Sample data
water_depletion_data = intersections_regions_extra[[
    'country',
    'country code',
    'province',
    'main basins',
    'cell_weighted_consumption',
    'cell_weighted_water_storage',
    'cell_weighted_runoff',
    'cell_weighted_precipitation',
    'cell_weighted_potential_evapotranspiration',
    'geometry',
    ]]

# Create a GeoDataFrame
water_depletion = gpd.GeoDataFrame(water_depletion_data, crs='EPSG:6933')

# Set 'geometry' as the active geometry column
water_depletion = water_depletion.set_geometry('geometry')

water_depletion = water_depletion.fillna(0)

# Calculate the aridity index (PET/P)
water_depletion['aridity index'] = (
    water_depletion['cell_weighted_potential_evapotranspiration']
    / water_depletion['cell_weighted_precipitation']
    )

water_depletion['aridity index scaled'] = np.where(
    water_depletion['aridity index'] <= 10,
    0.1 * water_depletion['aridity index'],
    1
    )

# Compute CTA  on regional level
water_depletion['CTA'] = (
    water_depletion['cell_weighted_consumption']
    / (
       water_depletion['cell_weighted_runoff']
       + water_depletion['cell_weighted_water_storage']
       )
    )

# Calculate CTA
denominator = (
    water_depletion['cell_weighted_runoff']
    # to avoid division by zero
    + water_depletion['cell_weighted_water_storage']
    )

water_depletion['CTA'] = (
    water_depletion['cell_weighted_consumption']
    / denominator.replace(0, np.nan)
    )

# Use a fubnction to calculate the WDI. 1 is the maximim,
# -17 ensures the sigmoid curve increases when the CTA increases
# and number defines the steepness,
# it means the logistic function transitions very sharply (steep slope)
# between the lower and upper asymptotes
# 0.25 is the threshold of extreme water stress,
# above this number, the WDI receives a value of 1

# Nonlinear transformation of physical water scarcity into
# vulnerability to freshwater depletion
# This is important as in the upper and lower ranges of
# CTA* doubled scarcity does not necessarily
# lead to doubled vulnerability to depletion.
# WDI turns 1 above a CTA* of 0.25, which is regarded
# as the threshold of extreme water stress (WAVE paper).

# Apply the function using exponent
x = (-17.6404 * (water_depletion['CTA'] - 0.2508)).clip(lower=-700, upper=700)
water_depletion['CTA scaled'] = 1.0047 / (1 + np.exp(x))

water_depletion['CTA scaled'] = water_depletion['CTA scaled'].fillna(0)
water_depletion['CTA scaled'] = water_depletion['CTA scaled'].clip(upper=1.0)

# Take the highest value between WDI region arid
# (set to 1 for arid regions) and WDI region all
water_depletion['WDI'] = np.maximum(
    water_depletion['CTA scaled'],
    water_depletion['aridity index scaled']
    )

# Calculate the mean WDI region per country, directly within the GeoDataFrame
water_depletion['WDI country'] = (
    water_depletion.groupby('country')['WDI'].transform('mean')
    )

water_depletion['WDI region'] = (
    water_depletion.groupby('province')['WDI'].transform('mean')
    )

# water_depletion['water_depletion'] = (
#     water_depletion['WDI'] * (1 - water_depletion['BIER_ru_12'])
#     )

# For the water storage: we have to convert the water mass to volume.
# We have dams, lakes and wetland storage in kg mm

# Plot the WDI
# Define bins including values above 10
bins = [0, 0.01, 0.02, 0.20, 0.40, 0.60, 0.80, 1]
labels = [
    '0–0.01', '0.01–0.02', '0.02–0.20', '0.20–0.40',
    '0.40–0.60', '0.60–0.80', '0.80–1'
    ]

water_depletion['WDI_bin'] = pd.cut(
    water_depletion['WDI'],
    bins=bins,
    labels=labels,
    include_lowest=True
    )

# Create a discrete colormap from Viridis
cmap = plt.cm.viridis
norm = mcolors.BoundaryNorm(boundaries=bins, ncolors=cmap.N)

# Plot with categorical bins
fig, ax = plt.subplots(figsize=(15, 10))
water_depletion.plot(
    column='WDI_bin',
    cmap=cmap,
    linewidth=0,         # No border lines
    ax=ax,
    edgecolor='none',    # Hide grid cell edges
    legend=True
)

# Fix legend
leg = ax.get_legend()
leg.set_title('WDI')
for t in leg.get_texts():
    t.set_text(
        t.get_text().replace('(', '').replace(']', '').replace(',', '–')
        )

plt.tight_layout()
plt.show()

# Save
plt.savefig('water_depletion_map.svg', bbox_inches='tight')
plt.savefig('water_depletion_map.png', dpi=300, bbox_inches='tight')

# Save as CSV file
water_depletion.to_csv('water_depletion.csv', index=False)

# %%% Socio-economic availability

# ---- CONCENTRATION OF WATER PRODUCTION

# Prepare the data
concentration_of_production_data = intersections_regions_extra[[
    'country',
    'country code',
    'province',
    'main basins',
    'abstraction m3 per basin',
    'abstraction m3 per region',
    'abstraction m3 per country',
    'geometry'
]].copy()

# Clean up negative and zero values to avoid division errors
concentration_of_production_data['abstraction m3 per country'] = (
    concentration_of_production_data['abstraction m3 per country']
    .clip(lower=1e-6)  # avoid division by zero
)

concentration_of_production_data['abstraction m3 per region'] = (
    concentration_of_production_data['abstraction m3 per region']
    .clip(lower=0)
)

# Convert to GeoDataFrame
concentration_of_production = gpd.GeoDataFrame(
    concentration_of_production_data,
    crs='EPSG:6933'
    )

concentration_of_production = (
    concentration_of_production.set_geometry('geometry')
    )

# Calculate abstraction fraction
concentration_of_production['abstraction_fraction'] = (
    concentration_of_production['abstraction m3 per region']
    / concentration_of_production['abstraction m3 per country']
    )

# Count number of basins per country
concentration_of_production['num_basins'] = (
    concentration_of_production.groupby('country')['province']
    .transform('nunique')
    )

# Initialize concentration score column
concentration_of_production['concentration_of_production'] = np.nan


def gini_coefficient(values):
    """Gini coefficient function."""
    values = np.array(values)
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    values = values[~np.isnan(values)]
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    return (
        (n + 1 - 2 * cumulative.sum() / cumulative[-1])
        / n if cumulative[-1] != 0 else 0
        )


# Apply Gini for multi-basin countries
multi_basin_mask = concentration_of_production['num_basins'] > 1

concentration_of_production.loc[
    multi_basin_mask,
    'concentration_of_production'] = (
        concentration_of_production[multi_basin_mask]
        .groupby('country')['abstraction_fraction']
        .transform(gini_coefficient)
        )

# Set concentration = 0 for single-basin countries
single_basin_mask = concentration_of_production['num_basins'] == 1
concentration_of_production.loc[
    single_basin_mask,
    'concentration_of_production'
    ] = 0

# Scale concentration between 0 and 1
concentration_of_production['concentration_of_production'] = (
    scaler.fit_transform(
        concentration_of_production[['concentration_of_production']]
        )
    )

# Replace remaining NaNs with 0 (optional fallback)
concentration_of_production['concentration_of_production'] = (
    concentration_of_production['concentration_of_production'].fillna(0)
    )

# Export to CSV
concentration_of_production.to_csv(
    'concentration_of_production.csv',
    index=False
    )

# ---- CONCENTRATION OF WATER RESERVES

# Sample data
concentration_of_reserves_data = intersections_regions_extra[[
    'country',
    'country code',
    'province',
    'main basins',
    'storage m3 per basin',
    'storage m3 per region',
    'storage m3 per country',
    'runoff m3 per basin',
    'runoff m3 per region',
    'runoff m3 per country',
    'geometry'
    ]].copy()

# Clean up negative values in 'per country' and 'per region'
concentration_of_reserves_data['storage m3 per country'] = (
    concentration_of_reserves_data['storage m3 per country'].clip(lower=0)
    )

concentration_of_reserves_data['storage m3 per region'] = (
    concentration_of_reserves_data['storage m3 per region'].clip(lower=0)
    )

# Create a geodataframe
concentration_of_reserves = gpd.GeoDataFrame(
    concentration_of_reserves_data,
    crs='EPSG:6933'
    )

# Set 'geometry' as the active geometry column
concentration_of_reserves = concentration_of_reserves.set_geometry('geometry')

# Sum runoff with the water storage per basin and country
concentration_of_reserves['total availability per basin'] = (
    concentration_of_reserves['storage m3 per basin']
    + concentration_of_reserves['runoff m3 per basin']
    )

# Sum runoff with the water storage per region and country
concentration_of_reserves['total availability per region'] = (
    concentration_of_reserves['storage m3 per region']
    + concentration_of_reserves['runoff m3 per region']
    )

# Sum runoff with the water storage per country
concentration_of_reserves['total availability per country'] = (
    concentration_of_reserves['storage m3 per country']
    + concentration_of_reserves['runoff m3 per country']
    )

# Calculate the fraction of region availability over the country availability
concentration_of_reserves['availability_fraction'] = (
    concentration_of_reserves['total availability per region']
    / concentration_of_reserves['total availability per country']
    )

# For country level, apply Gini Coefficient
# Add a new column to indicate whether the country has a single basin
concentration_of_reserves['num_basins'] = (
    concentration_of_reserves.groupby('country')['province']
    .transform('nunique')
    )

# Initialize a column for the final measure (Gini or fraction)
concentration_of_reserves['concentration_of_reserves'] = np.nan

# Apply Gini for countries with multiple basins
multi_basin_countries_r = concentration_of_reserves['num_basins'] > 1

concentration_of_reserves.loc[
    multi_basin_countries_r,
    'concentration_of_reserves'
    ] = (
        concentration_of_reserves[multi_basin_countries_r]
        .groupby('country')['availability_fraction']
        .transform(gini_coefficient)
        )

# Set concentration to 0 for single-basin countries
single_basin_countries_r = concentration_of_reserves['num_basins'] == 1

concentration_of_reserves.loc[
    single_basin_countries_r,
    'concentration_of_reserves'
    ] = 0

del multi_basin_countries_r, single_basin_countries_r

# Set concentration to 0 or NaN for countries with no basins
concentration_of_reserves['concentration_of_reserves'] = (
    concentration_of_reserves['concentration_of_reserves'].fillna(0)
    )

# Scale between 0 and 1
concentration_of_reserves['concentration_of_reserves'] = (
    scaler.fit_transform(
        concentration_of_reserves[['concentration_of_reserves']]
        )
    )

# Save as CSV file
concentration_of_reserves.to_csv(
    'concentration_of_reserves.csv',
    index=False
    )

# ---- WATER DEMAND GROWTH

# Import the population data
population_countries = pd.read_csv(
    './data/population/Undata/'
    'UNdata_Export_20250612_120552933_total.csv'
    )

# Rename 'ADMIN' to 'country'
population_countries = population_countries.rename(
    columns={
        'Country or Area': 'country',
        'Value': 'population',
        'Year(s)': 'year'
        }
    ).drop('Variant', axis=1)

population_countries['country'] = population_countries['country'].replace({
    'TÃ¼rkiye': 'Türkiye',
    'CuraÃ§ao': 'Curaçao'
})

# Adapt the adm code column
# --> change name to country code and consider only the 1st 3 letters
population_countries['country code'] = cc.convert(
    names=population_countries['country'],
    to='ISO3'
    )

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
population_countries['country code'] = (
    population_countries['country code'].replace(replaced_country_codes)
    )

# Multiply by 1000 since the dataset is available in thousands
population_countries['population'] = population_countries['population'] * 1000

# Import the consumption data
data = xr.open_dataset(
    './data/watergap22/'
    'watergap22e_gswp3-era5_atotuse_histsoc_monthly_1901_2023.nc'
    )

data['time'] = data['time'].astype('datetime64[ns]')

data.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
data.rio.write_crs('epsg:4326', inplace=True)

# Load and Prepare Consumption Dataset
extract_consumption = data  # Use the existing dataset

# Set CRS if not already defined
if not extract_consumption.rio.crs:
    extract_consumption.rio.write_crs('EPSG:4326', inplace=True)

# Extract the Last 5 Years of Data
# Get time values
time_values_consumption = extract_consumption['time'].values
# Select the last 5 years (assuming monthly data)
last_5_years_consumption = time_values_consumption[-60:]

# Select last 5 years of data
consumption_data = extract_consumption['atotuse'].sel(
    time=last_5_years_consumption
    )

# Ensure spatial dimensions are set for rioxarray
consumption_data = consumption_data.rio.set_spatial_dims(
    x_dim='lon',
    y_dim='lat',
    inplace=True
    )

# Import the countries data
gdf_countries_modified = gpd.read_file(
    'countries_modified.gpkg',
    layer='countries'
    )

# Ensure `gdf_countries_modified` Has the Same CRS
gdf_countries_modified = gdf_countries_modified.set_crs(
    gdf_countries_modified.crs,
    allow_override=True
    )

# Ensure `gdf_countries_modified` Has the Same CRS
gdf_countries_modified = gdf_countries_modified.to_crs(
    consumption_data.rio.crs
    )

# Clip Consumption Data Per Country
clipped_consumption = consumption_data.rio.clip(
    gdf_countries_modified.geometry.apply(mapping),
    gdf_countries_modified.crs,
    drop=True
    )

# Aggregate Consumption Per Country and Year
# Convert clipped consumption raster to a DataFrame
clipped_consumption_df = clipped_consumption.to_dataframe().reset_index()

# Convert time to year format
clipped_consumption_df['year'] = (
    pd.to_datetime(clipped_consumption_df['time']).dt.year
    )

# Perform a spatial join to assign each grid cell to a country
consumption_with_countries = gpd.sjoin(
    gpd.GeoDataFrame(
        clipped_consumption_df,
        geometry=gpd.points_from_xy(
            clipped_consumption_df['lon'],
            clipped_consumption_df['lat']
            ),
        crs=gdf_countries_modified.crs),
    gdf_countries_modified,
    how='inner',
    predicate='intersects'
    )

# Calculate consumption in m3 per area
# Drop the unnecessary columns
consumption_with_countries['consumption_m3'] = (
    consumption_with_countries['atotuse']  # kg/m²/s
    * 0.001  # Convert kg to m³
    * 31_536_000  # Convert per second to per year
    # Convert per m² to total area
    * consumption_with_countries['area_country']
    ).drop(
    columns=['time', 'spatial_ref', 'index_right']
    )

# Aggregate while keeping country, city, and geometry
consumption_countries = (
    consumption_with_countries
    .groupby(['ADMIN', 'year'])
    .agg({
        'consumption_m3': 'sum',  # Summing the volumes
        'lat': 'first',  # Keep the first
        'lon': 'first',
        'SOVEREIGNT': 'first',
        'geometry': 'first',
        'area_country': 'first',
        'area_country_km2': 'first'
        }
        ).reset_index()
    )

# Change columns' names
consumption_countries = consumption_countries.rename(columns={
    'ADMIN': 'country',
    'SOVEREIGNT': 'sovereignt'
    }
    )

# Adapt the adm code column
# --> change name to country code and consider only the 1st 3 letters
consumption_countries['country code'] = cc.convert(
    names=consumption_countries['country'],
    to='ISO3'
    )

consumption_countries = consumption_countries.drop('country', axis=1)

# Convert only where 'country code' is 'not found'
# mask = consumption_countries['country code'] == 'not found'
# consumption_countries.loc[mask, 'country code'] = coco.convert(
#    names=consumption_countries.loc[mask, 'sovereignt'],
#    to='ISO3'
# )

consumption_countries.loc[
    consumption_countries['sovereignt'] == 'Somaliland', 'country code'
    ] = 'SML'

# Prepare the dataframes before merging
consumption_countries = consumption_countries[
    consumption_countries['country code'] != 'not found'
    ]

population_countries = population_countries[population_countries[
    'country code'] != 'not found'
    ]

# Merge the two datasets on 'country' and 'year'
country_yearly_stats = pd.merge(
    consumption_countries,
    population_countries,
    on=['country code', 'year'],
    how='left'
    )

# Calculate per capita consumption
country_yearly_stats['consumption_per_capita'] = (
    country_yearly_stats['consumption_m3']
    / country_yearly_stats['population']
    )

# Convert back to GeoDataFrame to keep spatial information
country_yearly_stats = (
    gpd.GeoDataFrame(country_yearly_stats, geometry='geometry')
    )

# Calculate the year-to-year growth for each country
demand_growth = country_yearly_stats[[
    'country', 'country code', 'year', 'consumption_m3', 'population',
    'sovereignt', 'consumption_per_capita', 'lat', 'lon'
    ]]

# Sort the data by country and year to ensure correct time series order
demand_growth = demand_growth.sort_values(
    by=['country', 'country code', 'year']
    )

# Calculate the year-to-year growth for each country
demand_growth['yoy_growth'] = (
    demand_growth.groupby('country')
    ['consumption_per_capita']
    .transform(lambda x: x.pct_change())
    * 100
    )

# Convert to float and round to 2 decimal places
demand_growth['yoy_growth'] = (
    demand_growth['yoy_growth']
    .astype(float).round(2)
    )

# Calculate the 5-year rolling average growth for each country
demand_growth['5_year_avg_growth'] = (
    demand_growth.groupby('country')['yoy_growth'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
        )
    )

# Round the 5-year average growth to 2 decimal places
demand_growth['5_year_avg_growth'] = (
    demand_growth['5_year_avg_growth']
    .round(2)
    )

# Create a new DataFrame with only the average 5-year growth per country
demand_growth_country = (
    demand_growth.groupby(
        ['country', 'country code']
        )
    [['5_year_avg_growth']]
    .last()
    .reset_index()
)

# Replace negative values with 0
demand_growth_country['5_year_avg_growth'] = (
    demand_growth_country['5_year_avg_growth'].apply(lambda x: max(x, 0))
    )

# Log transformation to compress high values
# and spread out smaller ones more meaningfully:
demand_growth_country['log_scaled_growth'] = (
    np.log1p(demand_growth_country['5_year_avg_growth'])
    )

# Scale the index between 0 and 1
demand_growth_country['average demand scaled'] = scaler.fit_transform(
    demand_growth_country[['log_scaled_growth']]
)

# Set NaNs in the column to 0
demand_growth_country['average demand scaled'] = (
    demand_growth_country['average demand scaled'].fillna(0)
    )

# Save as CSV
population_countries.to_csv('population_countries.csv', index=False)
demand_growth.to_csv('demand_growth.csv', index=False)
demand_growth_country.to_csv('demand_growth_country.csv', index=False)

demand_growth_country.to_csv('demand_growth_country.csv', index=False)


demand_growth_country = pd.read_csv('demand_growth_country.csv')

# ---- WATER EXPLORATION

# Sample data
water_exploration_data = intersections_regions_extra[[
    'country',
    'country code',
    'province',
    'main basins',
    'type',
    'ppi',
    'geometry'
]].copy()

# Create a geodataframe
water_exploration = gpd.GeoDataFrame(water_exploration_data, crs='EPSG:6933')

# Set 'geometry' as the active geometry column
water_exploration = water_exploration.set_geometry('geometry')

# Calculate the water exploration category per country and per region
water_exploration['water exploration regions'] = (
    water_exploration['ppi'] *
    # The influence of water scarcity is reduced to 10% so that the importance
    # of the socio-economic restriction is given more weight
    (water_depletion['WDI region']/10)
     )

water_exploration['water exploration countries'] = (
    water_exploration['ppi'] *
    # The influence of water scarcity is reduced to 10% so that the importance
    # of the socio-economic re-striction is given more weight
    (water_depletion['WDI country']/10)
     )

# Scale between 0 and 1
water_exploration[[
    'water exploration regions scaled',
    'water exploration countries scaled'
    ]] = scaler.fit_transform(
        water_exploration[[
            'water exploration regions',
            'water exploration countries'
            ]]
        )

# Save as CSV file
water_exploration.to_csv('water_exploaration.csv', index=False)

# Plot the results

# water_exploration.plot(
#     column='water exploration countries scaled', legend=True
#     )

# water_exploration.plot(
#     column='water exploration regions scaled', legend=True
#     )

# test_bulg = water_exploration[
#    (water_exploration['country'] == 'Bulgaria') &
#    (water_exploration['type'] == 'Province')
#    ]

# test_bulg.plot(column='water exploration countries scaled', legend=True)
# test_bulg.plot(column='water exploration regions scaled', legend=True)


# ---- WATER GOVERNANCE

# Sample data
water_governance_data = intersections_regions_extra[[
    'country',
    'country code',
    'province',
    'main basins',
    'type',
    'wes',
    'wgi',
    'geometry'
]].copy()

# Create a geodataframe
water_governance = gpd.GeoDataFrame(water_governance_data, crs='EPSG:6933')

# Set 'geometry' as the active geometry column
water_governance = water_governance.set_geometry('geometry')

# Modilfy and scale the conflict and cooperation colomn (wes)
# Set NaNs in the column to 0
water_governance['wes'] = water_governance['wes'].fillna(0)
scaler2 = MinMaxScaler(feature_range=(-0.05, 0.05))

water_governance['wes scaled'] = (
    scaler2.fit_transform(water_governance[['wes']])
    )

# Scale WGI between 0 and 1
water_governance['wgi scaled'] = (
    scaler.fit_transform(water_governance[['wgi']])
    )

# Calculate the water exploration category per country and per region
water_governance['water governance regions'] = (
    water_governance['wgi scaled']
    * (water_depletion['WDI region']) / 10
    - water_governance['wes']
    )

water_governance['water governance countries'] = (
    (water_governance['wgi scaled'])
    * (water_depletion['WDI country'] / 10)
    - water_governance['wes scaled']
    )

water_governance[[
    'water governance regions scaled',
    'water governance countries scaled'
    ]] = scaler.fit_transform(
        water_governance[[
            'water governance regions',
            'water governance countries'
            ]]
        )

# Save as CSV file
water_governance.to_csv(
    'water_governance.csv',
    index=False
    )

water_governance.plot(
    column='water governance countries scaled',
    legend=True
    )

# ---- WATER AFFORDABILITY

# Sample data
water_affordability_data = intersections_regions_extra[[
    'country',
    'country code',
    'province',
    'main basins',
    'type',
    'statutory nominal gross monthly minimum wage',
    'total_tariff',
    'geometry'
    ]].copy()

# Create a geodataframe
water_affordability = gpd.GeoDataFrame(
    water_affordability_data,
    crs='EPSG:6933'
    )

# Set 'geometry' as the active geometry column
water_affordability = water_affordability.set_geometry('geometry')

# Calculate the water affordability using the tariff and min wage.
# Tariff is multiplied by 3 to reflect the average global usage
# of water (3 m3 per month per person).
# Low usage is considered. We multiply by 12 to scale to year
water_affordability['water_affordability'] = (
    (water_affordability['total_tariff']*3)
    / water_affordability['statutory nominal gross monthly minimum wage']
    )

water_affordability['water affordability scaled'] = scaler.fit_transform(
    water_affordability[['water_affordability']]
    )

# Save as CSV file
water_affordability.to_csv('water_affordability.csv', index=False)

# ---- WATER REUSE AND RECYCLING

# Sample data
# Make an explicit copy to avoid SettingWithCopyWarning
water_reuse = intersections_regions_extra[[
    'country',
    'country code',
    'province',
    'type',
    'main basins',
    'ww generated',
    'ww collected',
    'ww treated',
    'ww reused',
    'geometry'
    ]].copy()

# Define indicator weights in the final composite index
weights = {
    'ww generated': 10,
    'ww collected': 40,
    'ww treated': 40,
    'ww reused': 10
}

# Compute weighted wastewater index
water_reuse['ww index'] = (
    water_reuse[list(weights.keys())].mul(weights).sum(axis=1)
    / sum(weights.values())
    / 100
    )

# Normalise the index between 0 and 1
water_reuse['ww index'] = scaler.fit_transform(water_reuse[['ww index']])

# Reverse the scale (1 becomes 0, and 0 becomes 1)
water_reuse['ww index scaled'] = 1 - water_reuse['ww index']

# Keep only relevant columns
water_reuse = water_reuse[[
    'country',
    'country code',
    'province',
    'type',
    'main basins',
    'ww index scaled',  # Scaled & reversed wastewater index
    'geometry'
    ]]

# Save as CSV file
water_reuse.to_csv('water_reuse.csv', index=False)

# ---- FINAL RESULT

# Start with the base GeoDataFrame
supply_risk = (
    intersections_regions_extra[[
        'country', 'province', 'main basins', 'geometry'
        ]]
    .copy()
    )

# List of all GeoDataFrames to combine
geo_dataframes_supply_risk = [
     water_depletion,
     concentration_of_production,
     concentration_of_reserves,
     water_exploration,
     water_governance,
     water_affordability,
     water_reuse
    ]

# Loop through each GeoDataFrame and add columns to 'supply risk'
for gdf in geo_dataframes_supply_risk:
    # Loop through each column in the current GeoDataFrame
    for column in gdf.columns:
        if column != 'geometry':  # Don't overwrite the geometry column
            supply_risk[column] = gdf[column]

supply_risk = supply_risk.merge(
    # Select only necessary columns
    demand_growth_country[['country code', 'average demand scaled']],
    on='country code',
    how='left'
    )

# Calculate overall supply risk score (average of the indicators)
supply_risk['supply risk score region'] = (
    supply_risk[[
        'WDI region',
        'concentration_of_production',
        'concentration_of_reserves',
        'average demand scaled',
        'water exploration regions scaled',
        'water governance regions scaled',
        'water affordability scaled',
        'ww index scaled'
        ]]
    .sum(axis=1, min_count=8)
    )

# Replace NaNs with 0 if you want a fully filled output
# suply_risk = suply_risk.fillna(0)

# Calculate overall supply risk score (average of the indicators)
supply_risk['supply risk score country'] = (
    supply_risk[
        ['WDI country',
         'concentration_of_production',
         'concentration_of_reserves',
         'average demand scaled',
         'water exploration countries scaled',
         'water governance countries scaled',
         'water_affordability',
         'ww index scaled']
        ]
    .sum(axis=1, min_count=8)
)

# Apply MinMaxScaler to scale the scored for region and country results
supply_risk[['supply risk score region scaled']] = (
    scaler.fit_transform(supply_risk[['supply risk score region']])
    )

# Apply MinMaxScaler
supply_risk[['supply risk score country scaled']] = (
    scaler.fit_transform(supply_risk[['supply risk score country']])
    )

# Keep only relevant columns in the final supply_risk dataframe
columns_to_keep = [
    'country',
    'country code',
    'province',
    'main basins',
    'geometry',
    'WDI region',
    'WDI country',
    'concentration_of_production',
    'concentration_of_reserves',
    'average demand scaled',
    'water exploration regions scaled',
    'water governance regions scaled',
    'water exploration countries scaled',
    'water governance countries scaled',
    'water_affordability',
    'ww index scaled',
    'supply risk score region scaled',
    'supply risk score country scaled'
    ]

supply_risk = supply_risk[columns_to_keep]

# Rename a column
supply_risk_analysis = supply_risk.drop_duplicates(
    subset=['country']
    )

supply_risk_analysis = supply_risk_analysis.dropna(
    subset=['supply risk score country scaled']
    )

# Convert to GeoDataFrame to keep spatial properties
supply_risk = gpd.GeoDataFrame(supply_risk, crs='EPSG:6933')

# Save as CSV file
supply_risk.to_csv('supply_risk_all.csv', index=False)

(
 supply_risk[[
     'country', 'country code',
     'supply risk score country scaled', 'geometry'
     ]]
 .sort_values(by='country')
 .to_csv('supply_risk_final.csv', index=False)
 )

# %% Plot the results
# ---- Regional results

# Reproject to WGS84 for natural Earth plotting
supply_risk = supply_risk.to_crs(epsg=4326)

# Define number of bins and create equal-width bins between 0 and 1
num_bins = 10
bins = np.linspace(0, 1, num_bins + 1)
bin_labels = [f'{i + 1}' for i in range(num_bins)]

# Create a new column classifying scores into bins (NaNs remain NaN)
supply_risk['quantile_class'] = pd.cut(
    supply_risk['supply risk score region scaled'],
    bins=bins,
    labels=bin_labels,
    include_lowest=True
)

# Create the plot
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

# Plot the map, using equal interval classification
supply_risk.plot(
    column='supply risk score region scaled',
    cmap='viridis',
    scheme='user_defined',
    classification_kwds={'bins': bins[1:-1]},  # exclude 0 and 1 from legend
    legend=False,
    ax=ax,
    edgecolor='none',
    linewidth=0.0,
    missing_kwds={'color': 'lightgrey', 'label': 'No data'}
)

# Create custom circular legend
cmap = plt.cm.get_cmap('viridis', num_bins)
for i in range(num_bins):
    low, high = bins[i], bins[i + 1]
    label = f'{low:.2f} – {high:.2f}'
    color = cmap(i / (num_bins - 1))  # normalize to 0–1
    ax.scatter([], [], color=color, label=label, marker='o', s=100)

# Add "No data" legend entry
ax.scatter([], [], color='lightgrey', label='No data', marker='o', s=100)

# Final legend and map styling
ax.legend(title='Supply Risk', loc='lower left', frameon=False)
# ax.set_title('Supply Risk Score by Country', fontsize=16)
# ax.axis('off')

# Save output
plt.savefig('region_supply_risk_map_scaled.svg', bbox_inches='tight')
plt.savefig('region_supply_risk_map_scaled.png', dpi=300, bbox_inches='tight')
plt.show()


# %% Plot the results
# ---- Country results

# Reproject to WGS84 for natural Earth plotting
supply_risk = supply_risk.to_crs(epsg=4326)

# Define number of bins and create equal-width bins between 0 and 1
num_bins = 10
bins = np.linspace(0, 1, num_bins + 1)
bin_labels = [f'{i + 1}' for i in range(num_bins)]

# Create a new column classifying scores into bins (NaNs remain NaN)
supply_risk['quantile_class'] = pd.cut(
    supply_risk['supply risk score country scaled'],
    bins=bins,
    labels=bin_labels,
    include_lowest=True
)

# Create the plot
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

# Plot the map, using equal interval classification
supply_risk.plot(
    column='supply risk score country scaled',
    cmap='viridis',
    scheme='user_defined',
    classification_kwds={'bins': bins[1:-1]},  # exclude 0 and 1 from legend
    legend=False,
    ax=ax,
    edgecolor='none',
    linewidth=0.0,
    missing_kwds={'color': 'lightgrey', 'label': 'No data'}
)

# Create custom circular legend
cmap = plt.cm.get_cmap('viridis', num_bins)
for i in range(num_bins):
    low, high = bins[i], bins[i + 1]
    label = f'{low:.2f} – {high:.2f}'
    color = cmap(i / (num_bins - 1))  # normalize to 0–1
    ax.scatter([], [], color=color, label=label, marker='o', s=100)

# Add "No data" legend entry
ax.scatter([], [], color='lightgrey', label='No data', marker='o', s=100)

# Final legend and map styling
ax.legend(title='Supply Risk', loc='lower left', frameon=False)
# ax.set_title('Supply Risk Score by Country', fontsize=16)
# ax.axis('off')

# Save output
plt.savefig('country_supply_risk_map_scaled.svg', bbox_inches='tight')

plt.savefig(
    'country_supply_risk_map_scaled.png',
    dpi=300,
    bbox_inches='tight'
    )

plt.show()
