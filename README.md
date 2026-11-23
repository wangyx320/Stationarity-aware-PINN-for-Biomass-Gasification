# Stationarity-aware-PINN-for-Biomass-Gasification

This repository provides source code, experimental data, thermochemical equilibrium data, and trained models for a stationarity-aware physics-informed neural network (PINN) framework for predicting syngas composition during biomass gasification.

The framework incorporates FactSage-derived thermochemical constraints and local thermal stationarity information to improve predictive accuracy and physical consistency across the entire gasification process, including start-up, steady-state operation, and shutdown.

## Features

- **Data Processing:** Load and preprocess experimental biomass gasification data, including axial temperature measurements and operating conditions.
- **Multiple Machine Learning Models:** Includes Random Forest (RF), XGBoost, and four neural network models (M3–M6) for comparative evaluation.
- **Physics-informed Learning:** Incorporates thermochemical equilibrium and temperature-gradient constraints derived from FactSage.
- **Stationarity-aware Weighting:** Adaptively adjusts physical constraint contributions based on local thermal stationarity.
- **Pre-trained Models:** Provides trained model files and associated data scalers for reproducing model predictions.
- **Syngas Composition Prediction:** Predicts H₂, CO, CO₂, and CH₄ compositions.
- **Reproducibility:** Provides model implementations, data resources, configuration files, and environment requirements associated with the manuscript.

## Model Description

The repository includes the following models:

- **RF:** Random Forest regression model.
- **XGBoost:** Extreme Gradient Boosting regression model.
- **M3:** Data-driven multilayer perceptron (MLP).
- **M4:** Physics-informed MLP incorporating thermochemical equilibrium composition constraints.
- **M5:** Physics-informed MLP incorporating thermochemical equilibrium composition and temperature-gradient constraints.
- **M6:** Stationarity-aware PINN incorporating adaptive weighting of physical constraints based on local thermal stationarity.

## Requirements

- Python 3.x

Required Python packages are listed in `requirements.txt`.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. Download or clone this repository.
2. Install the required packages using `requirements.txt`.
3. The experimental dataset, FactSage thermochemical data, and data-splitting information are stored in the `Resource/` folder.
4. Run the corresponding Python script in the `Model/` folder to train or evaluate each model.
5. Pre-trained model files and associated scalers are provided for model prediction.
6. Check and adjust the file paths in the scripts if necessary.

## File Structure

- **Resource/:** Contains experimental data, FactSage thermochemical equilibrium data, fixed data-splitting information, and relevant model configuration files.
- **Model/:** Contains source code and trained model files for RF, XGBoost, and M3–M6.
- **requirements.txt:** Lists the required Python packages and their versions.
- **README.md:** Project description, model information, and usage instructions.
- **LICENSE:** License information.

## Note

The source code, data, and trained models are intended for academic and research use.
