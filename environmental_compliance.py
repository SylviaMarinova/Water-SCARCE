"""
Water SCARCE Criticality Assessment Script – Compliance with Environmental Standards Dimension
-----------------------------------------------------------------------------

This script calculates the environmental compliance component of Water 
SCARCE. The indicator reflects the extent to which countries and 
regions meet environmental performance standards related to biodiversity 
conservation, ecosystem quality, and air quality.

Main outputs:
    - Scaled values for each category (0–1)
    - Aggregated water compliance with environmental standards score for use in the overall criticality assessment
    - Map with global compliance with environmental standards distribution on regional and country level
    - Total criticality values
    - Top and bottom countries
    
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# %% Categories calculation

# ---- COMPLIANCE WITH ENVIRONMENTAL STANDARDS

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

# ---- ENVIRONMENTAL PERFORMANCE

# Sample data
# Make an explicit copy to avoid SettingWithCopyWarning
env_standards_compliance_data = intersections_regions_extra[[  
    'country',
    'country code',
    'province',
    'type',
    'main basins',
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
    'Ecosystem quality [m2•yr/m3]', 
    'Ecosystem quality',
    'geometry'
]]

# Create a GeoDataFrame
env_standards_compliance = gpd.GeoDataFrame(env_standards_compliance_data, crs='EPSG:6933')

# Set 'geometry' as the active geometry column
env_standards_compliance = env_standards_compliance.set_geometry('geometry')

# Define indicator weights in the final composite index
weights_biodiversity_all = {
    'marine kba protection': 12,
    'marine habitat protection': 12,
    'marine protection stringency': 2,
    'terrestrial biome protection (national weights)': 10,
    'terrestrial kba protection': 10,
    'protected areas representativeness index': 12,
    'protected area effectiveness': 2,
    'protected human land': 2,
    'red list index': 12,
    'bioclimatic ecosystem resilience': 2,
    'species habitat index': 8,
    'species protection index': 16,
}

weights_biodiversity_terrestrial_only = {
    k: v for k, v in weights_biodiversity_all.items()
    if not k.startswith('marine')
}

weights_air_quality = {
    'anthropogenic PM2.5 exposure': 38,
    'household solid fuels': 38,
    'ozone exposure': 9,
    'NO2 exposure': 6,
    'SO2 exposure': 3,
    'CO exposure': 3,
    'VOC exposure': 3,
}

# Add a column to flag marine data presence
env_standards_compliance['has_marine_data'] = env_standards_compliance[
    ['marine kba protection', 'marine habitat protection', 'marine protection stringency']
].notna().all(axis=1)

# General function to compute weighted average of indicators
def compute_weighted_score(row, weights):
    valid_keys = [k for k in weights if pd.notna(row[k])]
    if not valid_keys:
        return np.nan

    filtered_weights = {k: weights[k] for k in valid_keys}
    values = pd.Series({k: row[k] for k in valid_keys})

    weighted = values * pd.Series(filtered_weights)
    return weighted.sum() / sum(filtered_weights.values())

def compute_biodiversity_score(row, weights_with_marine, weights_without_marine):
    # Choose the appropriate weights
    weights = weights_with_marine if row['has_marine_data'] else weights_without_marine

    # Select only available (non-NaN) indicators
    valid_keys = [k for k in weights if pd.notna(row[k])]
    if not valid_keys:
        return np.nan  # No usable indicators

    # Filter weights and values for available data
    filtered_weights = {k: weights[k] for k in valid_keys}
    values = pd.Series({k: row[k] for k in valid_keys})

    # Compute weighted average with available indicators
    weighted = values * pd.Series(filtered_weights)
    return weighted.sum() / sum(filtered_weights.values())

# Apply biodiversity score (marine logic used here)
env_standards_compliance['biodiversity & habitat'] = env_standards_compliance.apply(
    compute_biodiversity_score,
    axis=1,
    args=(weights_biodiversity_all, weights_biodiversity_terrestrial_only)
)/100

# Apply air quality score (no marine logic, just standard weighted average)
env_standards_compliance['air quality'] = env_standards_compliance.apply(
    compute_weighted_score,
    axis=1,
    args=(weights_air_quality,)
)/100

# Scale between 0 and 1
env_standards_compliance['biodiversity & habitat'] = scaler.fit_transform(env_standards_compliance[['biodiversity & habitat']])

# ED index
env_standards_compliance['ecosystem quality'] = env_standards_compliance_data['Ecosystem quality']

# Reverse the ecosystem quality values
env_standards_compliance['ecosystem quality norm'] = (
    env_standards_compliance['ecosystem quality'].max() - env_standards_compliance['ecosystem quality']
)

# Apply min-max scaling to bring between 0 and 1
env_standards_compliance[['ecosystem quality norm']] = scaler.fit_transform(
    env_standards_compliance[['ecosystem quality norm']]
)

# Compute weighted index
components = pd.concat([
    env_standards_compliance['biodiversity & habitat'],
    env_standards_compliance['air quality'],
    env_standards_compliance['ecosystem quality norm']
], axis=1)

env_standards_compliance['env_compliance'] = components.sum(axis=1).where(components.notna().all(axis=1))

# Normalize between 0 and 1
env_standards_compliance['env_compliance_norm'] = scaler.fit_transform(
    env_standards_compliance[['env_compliance']]
)

# Reverse the scale (1 becomes 0, and 0 becomes 1)
env_standards_compliance['env_compliance_scaled'] = 1 - env_standards_compliance['env_compliance_norm']

# Convert to GeoDataFrame to keep spatial properties
env_standards_compliance = gpd.GeoDataFrame(env_standards_compliance, crs='EPSG:6933')

# Save as CSV file
env_standards_compliance.to_csv('env_standards_compliance_all.csv', index=False)
env_standards_compliance[['country', 'country code', 'env_compliance_scaled', 'geometry']] \
    .sort_values(by='country') \
    .to_csv('env_standards_compliance_final.csv', index=False)

#%% Plot the results
# ---- Country results

# Reproject to WGS84 for natural Earth plotting
env_standards_compliance = env_standards_compliance.to_crs(epsg=4326)

# Define number of bins and create equal-width bins between 0 and 1
num_bins = 10
bins = np.linspace(0, 1, num_bins + 1)
bin_labels = [f'{i + 1}' for i in range(num_bins)]

# Create a new column classifying scores into bins (NaNs remain NaN)
env_standards_compliance['quantile_class'] = pd.cut(
    env_standards_compliance['env_compliance_scaled'],
    bins=bins,
    labels=bin_labels,
    include_lowest=True
)

# Create the plot
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

# Plot the map, using equal interval classification
env_standards_compliance.plot(
    column='env_compliance_scaled',
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
ax.legend(title='Environmental compliance', loc='lower left', frameon=False)
#ax.set_title('Supply Risk Score by Country', fontsize=16)

# Disable gridlines but keep axes & ticks
ax.grid(False)

# Save output
plt.savefig('environmental_compliance_map.svg', bbox_inches='tight')
plt.savefig('environmental_compliance_map.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Total criticality calculation

supply_risk = pd.read_csv(
    './supply_risk_final.csv',
    encoding='ISO-8859-1'
    )

vulnerability = pd.read_csv(
    './vulnerability_final.csv',
    encoding='ISO-8859-1'
    )

social = pd.read_csv(
    './social_standards_compliance_final.csv',
    encoding='ISO-8859-1'
    )

environmental = pd.read_csv(
    './env_standards_compliance_final.csv',
    encoding='ISO-8859-1'
    )

sr = supply_risk[['country code', 'country', 'supply risk score country scaled']].drop_duplicates(subset='country code')
v = vulnerability[['country code', 'vulnerability score scaled']].drop_duplicates(subset='country code')
s = social[['country code', 'social compliance scaled']].drop_duplicates(subset='country code')
e = environmental[['country code', 'env_compliance_scaled']].drop_duplicates(subset='country code')

criticality_score = sr.merge(v, on='country code', how='outer') \
                      .merge(s, on='country code', how='outer') \
                      .merge(e, on='country code', how='outer')

# Rename for clarity
criticality_score = criticality_score.rename(columns={
    'supply risk score country scaled': 'SR',
    'vulnerability score scaled': 'V',
    'social compliance scaled': 'S',
    'env_compliance_scaled': 'E'
})

criticality_score = criticality_score.dropna().reset_index(drop=True)

criticality_score['criticality_score'] = (
    np.sqrt(
        criticality_score['SR']**2 + 
        criticality_score['V']**2 + 
        criticality_score['S']**2 + 
        criticality_score['E']**2) / np.sqrt(4)
)

# Reset the index afterwards
criticality_score = criticality_score.reset_index(drop=True)

# Scale between 0 and 1
criticality_score['criticality_score_scaled'] = scaler.fit_transform(criticality_score[['criticality_score']])

criticality_score.sort_values(by='country').to_csv('criticality_score_final.csv', index=False)

#%% Plot as chart

# Ensure country names are strings
criticality_score['country'] = criticality_score['country'].astype(str)

criticality_score = criticality_score.dropna(subset=['criticality_score_scaled'])

# Sort by score
sorted_scores = criticality_score.sort_values(by='criticality_score_scaled', ascending=False)

# Top 10 and Bottom 10
top10 = sorted_scores.head(10)
bottom10 = sorted_scores.tail(10)

sns.set(style="whitegrid")

fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Top 10 plot
sns.barplot(
    x='criticality_score_scaled', 
    y='country', 
    data=top10, 
    hue='country',
    palette='Reds_r', 
    dodge=False,
    legend=False,
    ax=axs[0]
)
axs[0].set_title('Leading contributors by criticality score')
axs[0].set_xlabel('Overall criticality score')
axs[0].set_ylabel('Country')

min_top10 = top10['criticality_score_scaled'].min()
max_top10 = top10['criticality_score_scaled'].max()
axs[0].set_xlim(min_top10 * 0.95, max_top10 * 1.0)

axs[0].set_xticklabels([f"{tick:.2f}" for tick in axs[0].get_xticks()])
axs[1].set_xticklabels([f"{tick:.2f}" for tick in axs[1].get_xticks()])

# Bottom 10 plot
sns.barplot(
    x='criticality_score_scaled', 
    y='country', 
    data=bottom10, 
    hue='country',
    palette='Greens', 
    dodge=False,
    legend=False,
    ax=axs[1]
)
axs[1].set_title('Lowest contributors by criticality score')
axs[1].set_xlabel('Overall criticality score')
axs[1].set_ylabel('')

min_bottom10 = bottom10['criticality_score_scaled'].min()
max_bottom10 = bottom10['criticality_score_scaled'].max()
axs[1].set_xlim(min_bottom10 * 0.95, max_bottom10 * 1.0)

axs[0].set_xticklabels([f"{tick:.2f}" for tick in axs[0].get_xticks()])
axs[1].set_xticklabels([f"{tick:.2f}" for tick in axs[1].get_xticks()])

plt.tight_layout()

# Save the figure before showing it
plt.savefig('criticality_chart.svg', bbox_inches='tight')
plt.savefig('criticality_chart.png', dpi=300, bbox_inches='tight')

plt.show()

#%% Another option

# Make copies and add 'group' column
top10 = top10.copy()
bottom10 = bottom10.copy()
top10['group'] = 'Top 10'
bottom10['group'] = 'Bottom 10'

# Combine data
combined = pd.concat([top10, bottom10])
combined = combined.sort_values(by='criticality_score_scaled', ascending=False)

# Generate colors from colormaps
top_colors = plt.cm.Reds_r(np.linspace(0.4, 1, len(top10)))    # darker reds
bottom_colors = plt.cm.Greens(np.linspace(0.4, 1, len(bottom10)))  # darker greens

# Map colors to the dataframe
color_map = list(top_colors) + list(bottom_colors)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))

sns.barplot(
    x='criticality_score_scaled',
    y='country',
    data=combined,
    dodge=False,
    palette=color_map
)

plt.xlabel('Overall criticality score')
plt.ylabel('Country')
plt.title('Top and bottom criticality contributors')
plt.tight_layout()
plt.show()

#%% Merge the dataframes with the geometries

# Convert geometry from WKT (if stored as text in CSV)
if supply_risk['geometry'].dtype == object:
    supply_risk['geometry'] = gpd.GeoSeries.from_wkt(supply_risk['geometry'])

# Merge them all on 'country'
merged = supply_risk.merge(vulnerability, on='country', how='left')\
                    .merge(social, on='country', how='left')\
                    .merge(environmental, on='country', how='left')

# Convert back to GeoDataFrame
merged_gdf = gpd.GeoDataFrame(merged, geometry='geometry')

#%%%
#%% Load data for the clustering and Euclidean distance
supply_risk = pd.read_csv('./supply_risk_final.csv', encoding='ISO-8859-1')
vulnerability = pd.read_csv('./vulnerability_final.csv', encoding='ISO-8859-1')
social = pd.read_csv('./social_standards_compliance_final.csv', encoding='ISO-8859-1')
environmental = pd.read_csv('./env_standards_compliance_final.csv', encoding='ISO-8859-1')

#%% Prepare criticality score
sr = supply_risk[['country code', 'country', 'supply risk score country scaled']].drop_duplicates(subset='country code')
v = vulnerability[['country code', 'vulnerability score scaled']].drop_duplicates(subset='country code')
s = social[['country code', 'social compliance scaled']].drop_duplicates(subset='country code')
e = environmental[['country code', 'env_compliance_scaled']].drop_duplicates(subset='country code')

criticality_score = sr.merge(v, on='country code', how='outer') \
                      .merge(s, on='country code', how='outer') \
                      .merge(e, on='country code', how='outer')

criticality_score = criticality_score.rename(columns={
    'supply risk score country scaled': 'SR',
    'vulnerability score scaled': 'V',
    'social compliance scaled': 'S',
    'env_compliance_scaled': 'E'
}).dropna().reset_index(drop=True)

criticality_score['criticality_score'] = np.sqrt(
    criticality_score['SR']**2 + 
    criticality_score['V']**2 + 
    criticality_score['S']**2 + 
    criticality_score['E']**2
) / np.sqrt(4)

# Scale between 0 and 1
scaler = MinMaxScaler()
criticality_score['criticality_score_scaled'] = scaler.fit_transform(
    criticality_score[['criticality_score']]
)

#%% Select top 10 and bottom 10 critical countries
sorted_df = criticality_score.sort_values(by='criticality_score_scaled', ascending=False)
top_10 = sorted_df.head(10)
bottom_10 = sorted_df.tail(10)
subset = pd.concat([top_10, bottom_10]).reset_index(drop=True)

#%% Hierarchical clustering_version 2
# Use only the criticality dimensions
X = criticality_score[['SR', 'V', 'S', 'E']].values
Z = linkage(X, method='ward', metric='euclidean')

# Set number of clusters
num_clusters = 3
clusters = fcluster(Z, num_clusters, criterion='maxclust')

# Identify top/middle/bottom criticality countries
sorted_countries = criticality_score.sort_values('criticality_score_scaled')
n = len(sorted_countries)
lowest_countries = sorted_countries.head(3)['country'].tolist()
middle_countries = sorted_countries.iloc[n//2-1:n//2+2]['country'].tolist()  # 3 middle
highest_countries = sorted_countries.tail(3)['country'].tolist()

# Assign colors to clusters
cluster_colors = {1:'green', 2:'blue', 3:'red'}
country_to_cluster = dict(zip(criticality_score['country'], clusters))

# Function to assign color to each branch based on the cluster of leaves under it
def link_color_func(link_id):
    if link_id < len(X):
        country = criticality_score['country'].iloc[link_id]
        return cluster_colors[country_to_cluster[country]]
    else:
        left = int(Z[link_id - len(X), 0])
        right = int(Z[link_id - len(X), 1])
        left_color = link_color_func(left)
        right_color = link_color_func(right)
        return left_color if left_color == right_color else 'gray'

# Plot dendrogram
plt.figure(figsize=(14,6))
dendro = dendrogram(
    Z,
    labels=criticality_score['country'].values,
    leaf_rotation=90,
    link_color_func=link_color_func
)

ax = plt.gca()

# Highlight top/middle/bottom countries
for tick in ax.get_xticklabels():
    country = tick.get_text()
    if country in highest_countries:
        tick.set_color('red')
        tick.set_fontweight('bold')
    elif country in middle_countries:
        tick.set_color('blue')
        tick.set_fontweight('bold')
    elif country in lowest_countries:
        tick.set_color('green')
        tick.set_fontweight('bold')

# Add cluster labels under the corresponding leaves
leaf_positions = np.array(dendro['leaves'])
n_leaves = len(leaf_positions)

# We'll assign one x-position for each cluster: left, center, right
cluster_positions = [
    int(n_leaves * 1.5),  # left
    int(n_leaves * 5.5),  # center
    int(n_leaves * 8.8)   # right
]

# Cluster numbers in order
cluster_nums = [1, 2, 3]

for i, cluster_num in enumerate(cluster_nums):
    x_pos = cluster_positions[i]
    plt.text(
        x_pos, -1, f"Cluster {cluster_num}", 
        ha='center', va='top', fontsize=10, 
        color=cluster_colors[cluster_num]  # assign a different color to each
    )

#plt.ylim(bottom=-1.2)  # make space for cluster labels
plt.tight_layout()
plt.show()

#%% Highlight countries based on individual dimensions, ignore the total score
n_high_low = 5  # top/bottom n countries
n_middle = 5    # middle n countries
dimensions = ['SR', 'V', 'S', 'E']

# Rank countries per dimension
ranks = criticality_score[dimensions].rank(method='min', ascending=False)

# Boolean masks
high_counts = ranks <= n_high_low
mid_counts = (ranks > len(criticality_score)//2 - n_middle//2) & \
             (ranks <= len(criticality_score)//2 + n_middle//2)
low_counts = ranks >= len(criticality_score) - n_high_low + 1

# Sum across dimensions
high_sum = high_counts.sum(axis=1)
mid_sum = mid_counts.sum(axis=1)
low_sum = low_counts.sum(axis=1)

# Assign categories based on majority in dimensions
highlight_category = []
for h, m, l in zip(high_sum, mid_sum, low_sum):
    if h >= 2:
        highlight_category.append('highest')
    elif m >= 2:
        highlight_category.append('middle')
    elif l >= 2:
        highlight_category.append('lowest')
    else:
        highlight_category.append(None)

criticality_score['highlight'] = highlight_category

#%% Hierarchical clustering
X = criticality_score[['SR', 'V', 'S', 'E']].values
Z = linkage(X, method='ward', metric='euclidean')

# Set number of clusters
num_clusters = 3
clusters = fcluster(Z, num_clusters, criterion='maxclust')

# Map countries to clusters
cluster_colors = {1:'green', 2:'blue', 3:'red'}
country_to_cluster = dict(zip(criticality_score['country'], clusters))

# Function to assign color to each branch based on the cluster of leaves under it
def link_color_func(link_id):
    if link_id < len(X):
        country = criticality_score['country'].iloc[link_id]
        return cluster_colors[country_to_cluster[country]]
    else:
        left = int(Z[link_id - len(X), 0])
        right = int(Z[link_id - len(X), 1])
        left_color = link_color_func(left)
        right_color = link_color_func(right)
        return left_color if left_color == right_color else 'gray'

#%% Plot dendrogram
plt.figure(figsize=(14,6))
dendro = dendrogram(
    Z,
    labels=criticality_score['country'].values,
    leaf_rotation=90,
    link_color_func=link_color_func
)

ax = plt.gca()

# Highlight countries based on individual dimensions
for tick in ax.get_xticklabels():
    country = tick.get_text()
    cat = criticality_score.loc[criticality_score['country']==country, 'highlight'].values[0]
    if cat == 'highest':
        tick.set_color('red')
        tick.set_fontweight('bold')
    elif cat == 'middle':
        tick.set_color('orange')
        tick.set_fontweight('bold')
    elif cat == 'lowest':
        tick.set_color('green')
        tick.set_fontweight('bold')

# Add cluster labels under leaves
leaf_positions = np.array(dendro['leaves'])
n_leaves = len(leaf_positions)
cluster_positions = [
    int(n_leaves * 1.5),  # left
    int(n_leaves * 5.5),  # center
    int(n_leaves * 8.8)   # right
]

cluster_nums = [1, 2, 3]

for i, cluster_num in enumerate(cluster_nums):
    x_pos = cluster_positions[i]
    plt.text(
        x_pos, -0.05, f"Cluster {cluster_num}", 
        ha='center', va='top', fontsize=10, 
        color=cluster_colors[cluster_num]
    )

plt.tight_layout()
plt.show()