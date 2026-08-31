## European Forest Disturbance Mapping

This repository contains the code for **forest disturbance mapping and validation across Europe**, developed in:

> **Alba Viana-Soto, Jorunn Anna Mense, Katja Kowalski, Jan Pauls, Fabian Gieseke & Cornelius Senf (2026).** 
> *Deep learning-based forest disturbance detection for Europe using Landsat time series.*  
> *DOI*

This repository extends the **European Forest Disturbance Atlas (EFDA)**, a Landsat-based approach for mapping annual forest disturbances across continental Europe since 1985. It includes further developments in disturbance mapping, validation, and area estimation. 🌳🌲🛰️🗺️


---

## Disturbance detection workflow

### 🛰️ Disturbance mapping

Deep-learning-based disturbance detection using Landsat time series. 
We implemented two temporal architectures for pixel-wise forest disturbance detection: a TempCNN adapted from Pelletier et al. (2019) and Perbet et al. (2024) and a 1D U-Net, adapted from Ronneberger et al. (2015). 

- [Disturbance mapping code](https://github.com/jorunnmense/deep_disturbance)

### 🎯 Accuracy assessment

Code for the sampling design, reference data processing, and accuracy assessment of the disturbance maps.

### 📊 Area estimation

Code for sample-based disturbance area estimation and associated uncertainty estimation.

---
## Citation

If you use this repository, please cite:
> Viana-Soto, A., Mense, J. A., Kowalski, K., Pauls, J., Gieseke, F., & Senf, C. (2026).  
> *Deep learning-based forest disturbance detection for Europe using Landsat time series.*  
> **Under review**
