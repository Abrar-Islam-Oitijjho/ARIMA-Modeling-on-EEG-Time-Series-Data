# ARIMA Modeling on EEG Time-Series Data

Time-series analysis and forecasting of EEG data (ICP, AMP, and RAP) using ARIMA modeling. Includes stationarity checks, residual analysis, artifact detection, and visualization.

![GitHub stars](https://img.shields.io/github/stars/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data?style=social)
![GitHub forks](https://img.shields.io/github/forks/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data?style=social)
![GitHub issues](https://img.shields.io/github/issues/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data)
![GitHub last commit](https://img.shields.io/github/last-commit/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data)

<p>
  <img src="https://img.shields.io/badge/Language-Python-blue.svg" alt="Python" style="display:inline-block; margin-right:5px;">
  <img src="https://img.shields.io/badge/Library-NumPy-green.svg" alt="NumPy" style="display:inline-block; margin-right:5px;">
  <img src="https://img.shields.io/badge/Library-Pandas-green.svg" alt="Pandas" style="display:inline-block; margin-right:5px;">
  <img src="https://img.shields.io/badge/Library-Matplotlib-green.svg" alt="Matplotlib" style="display:inline-block; margin-right:5px;">
  <img src="https://img.shields.io/badge/Library-Statsmodels-green.svg" alt="Statsmodels" style="display:inline-block; margin-right:5px;">
  <img src="https://img.shields.io/badge/Library-Scikit--learn-green.svg" alt="Scikit-learn" style="display:inline-block;">
</p>


## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Result](#result)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Associated Publication](#associated-publication)
- [Contributing](#contributing)
- [Support](#support)
- [Acknowledgments](#acknowledgments)

## About

This project builds a full ARIMA-based time-series analysis pipeline for EEG time series, ICP, AMP, and RAP signals. It runs stationarity checks, performs ARIMA modeling, generates residual diagnostics, plots ACF and PACF of residuals, and evaluates differences between clean and artifact-affected segments. The workflow processes every patient and every physiological parameter, produces plots and summary tables, and stores all metrics for later statistical comparison. The goal is to characterize RAP behavior, identify artifact patterns, and support downstream modeling of cerebral compensatory reserve in TBI data.

By leveraging Python's  libraries such as NumPy, Pandas, Matplotlib, Statsmodels, and Scikit-learn, the project offers a streamlined workflow for researchers and practitioners to extract meaningful insights from complex EEG datasets. This work has been published in MDPI's Sensors. The link to that can be found in the following section. 


## Features

- ⚡ **Stationarity Checks**: Implementation of Augmented Dickey-Fuller test to ensure data stationarity.
- 🎯 **Time-Series Analysis**: Comprehensive analysis of EEG data (ICP, AMP, RAP) using ARIMA models.
- 🎨 **Visualization**: Clear and informative visualizations of time-series data, model forecasts, and residuals.
- 📈 **Forecasting**: Predict future EEG data values using fitted ARIMA models.
- 🔒 **Artifact Detection **: Methods and features to identify and mitigate artifacts in EEG data.
- 🛠️ **Customizable**: Modular code structure allows for easy customization and extension.


## Result

### ACF and PACF Plots of Residuals Before ARIMA
![ACF and PACF](plot/acf_pacf_org.jpg)

### ACF and PACF Plots of Residuals Before ARIMA
![ACF and PACF](plot/acf_pacf_ordered.jpg)

### Stationarity Check (p-values)
[ADF Test Original](csv/ADF_test_1min.csv)<br>
[KPSS Test Original](csv/ADF_test_1min_diff.csv)<br>
[ADF Test First Order Differenced](csv/KPSS_test_1min.csv)<br>
[KPSS Test First Order Differenced](csv/KPSS_test_1min_diff.csv)


### Optimal ARIMA Orders
[Optimal ARIMA Orders Clean](csv/arima_optimal_orders_clean.csv)<br>
[Optimal ARIMA Orders Artifacts](csv/arima_optimal_orders_artifact.csv)

### Scatterplots of Orders (Clean vs Artifact)
![ACF and PACF](plot/scatter_plot_clean_artifact.jpg)

## Quick Start

Clone the repository and run the main script:

```bash
git clone https://github.com/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data.git
cd ARIMA-Modeling-on-EEG-Time-Series-Data
pip install -r requirements.txt
jupyter notebook arima_analysis_main.ipynb
```

## Installation

### Prerequisites
- Python 3.7+
- pip 22.3+
- conda 24.11.3
- scikit-learn 1.0.2
- Pandas 1.3.5

## Project Structure

```
ARIMA-Modeling-on-EEG-Time-Series-Data/
├── code/               
    ├── data_preprocess.py          
    ├── arima_modeler.py            
    ├── artifact_analyzer.py     
    ├── artifact_detector.py
    ├── arima_analysis_main.ipynb
├── csv/            
├── plot/               
├── README.md              
└── requirements.txt                 
```

## Associated Publication

[Characterization of RAP Signal Patterns, Temporal Relationships, and Artifact Profiles Derived from Intracranial Pressure Sensors in Acute Traumatic Neural Injury](https://www.mdpi.com/1424-8220/25/2/586)

## Contributing

Contributions are welcome!

### Quick Contribution Steps
1. 🍴 Fork the repository
2. 🌟 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. ✅ Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request


## Support

- 📧 **Email**: abraroitijjho35@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data/issues)

## Acknowledgments

- 📚 **Libraries used**:
  - [NumPy](https://numpy.org/) - Numerical computing library
  - [Pandas](https://pandas.pydata.org/) - Data analysis library
  - [Matplotlib](https://matplotlib.org/) - Visualization library
  - [Statsmodels](https://www.statsmodels.org/stable/index.html) - Statistical modeling library
  - [Scikit-learn](https://scikit-learn.org/stable/) - Machine learning library
  - 
🌟 Special thanks: to the Main-Hub Laboratory, University of Manitoba, for their great support.
