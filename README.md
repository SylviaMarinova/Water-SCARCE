# Water SCARCE  

**Water Criticality Assessment Framework**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16994564.svg)](https://doi.org/10.5281/zenodo.16994564)

---

## Description

This repository contains all scripts, data, and workflows for the calculation of the **Water SCARCE characterisation factors (CFs)**.  

The framework integrates four main dimensions:
- **Supply Risk** – physical and socio-economic availability of water  
- **Vulnerability** – susceptibility of regions to supply risk  
- **Compliance with Environmental Standards** – biodiversity protection, air quality, and ecosystem health  
- **Compliance with Social Standards** – human rights performance  

The framework builds on harmonised geospatial indicators and produces country- and region-level scores on a global scale.  

All scripts are written in **Python**, using `GeoPandas`, `NumPy`, and `scikit-learn` for data processing.

---

## Outputs
- Harmonised indicator datasets (**GeoPackages, CSVs**)  
- Dimension-level scores for Supply Risk, Vulnerability, Environmental, and Social Compliance  
- Integrated **Water Criticality Index** maps and tables  

---

## Scripts

> `indicators_preparation.py` – Prepares and harmonises all environmental, hydrological, and socio-economic indicators  
> `supply_risk.py` – Calculates the Supply Risk dimension  
> `vulnerability.py` – Calculates the Vulnerability dimension  
> `environmental_compliance.py` – Calculates Environmental Standards compliance (biodiversity, air quality, ecosystem quality)  
> `social_compliance.py` – Calculates Social Standards compliance (human rights)  
> `criticality_cfs.py` – Integrates all dimensions into the final overall criticality  

---

## Data

All processed and raw input datasets (GeoPackages, CSVs) are archived on **Zenodo**:  

> [https://doi.org/10.5281/zenodo.16994564](https://doi.org/10.5281/zenodo.16994564)  

**Folders in this repo:**  
- `data/` – Processed and raw input datasets  
- `results/` – CSV outputs 




