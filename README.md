# ARIMA-Modeling-on-EEG-Time-Series-Data



# 🤖 ARIMA Modeling on EEG Time-Series Data

Time-series analysis and forecasting of EEG data (ICP, AMP, and RAP) using ARIMA modeling. Includes stationarity checks, residual analysis, artifact detection, and visualization.

![License](https://img.shields.io/github/license/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data)
![GitHub stars](https://img.shields.io/github/stars/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data?style=social)
![GitHub forks](https://img.shields.io/github/forks/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data?style=social)
![GitHub issues](https://img.shields.io/github/issues/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data)
![GitHub last commit](https://img.shields.io/github/last-commit/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data)

<img src="https://img.shields.io/badge/Language-Python-blue.svg" alt="Python">
<img src="https://img.shields.io/badge/Library-NumPy-green.svg" alt="NumPy">
<img src="https://img.shields.io/badge/Library-Pandas-green.svg" alt="Pandas">
<img src="https://img.shields.io/badge/Library-Matplotlib-green.svg" alt="Matplotlib">
<img src="https://img.shields.io/badge/Library-Statsmodels-green.svg" alt="Statsmodels">
<img src="https://img.shields.io/badge/Library-Scikit--learn-green.svg" alt="Scikit-learn">

## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Testing](#testing)
- [License](#license)
- [Support](#support)
- [Acknowledgments](#acknowledgments)

## About

This project focuses on the time-series analysis of EEG data using ARIMA (Autoregressive Integrated Moving Average) models. EEG data, specifically ICP (Intracranial Pressure), AMP (Amplitude), and RAP (Rate of change of Amplitude and Pressure), is analyzed to understand underlying patterns and forecast future values. The project addresses the need for robust time-series modeling techniques in neuroscience and biomedical engineering.

The primary goal is to provide a comprehensive framework for EEG data analysis, encompassing data preprocessing, stationarity checks, model fitting, residual analysis, and visualization. By leveraging Python's powerful libraries such as NumPy, Pandas, Matplotlib, Statsmodels, and Scikit-learn, the project offers a streamlined workflow for researchers and practitioners to extract meaningful insights from complex EEG datasets.

Key technologies include ARIMA models for time-series forecasting, statistical tests for stationarity (e.g., Augmented Dickey-Fuller test), and visualization techniques for data exploration and model evaluation. The project's unique selling point lies in its integration of multiple analytical steps into a cohesive pipeline, enabling users to efficiently analyze and interpret EEG data.

## ✨ Features

- 🎯 **Time-Series Analysis**: Comprehensive analysis of EEG data (ICP, AMP, RAP) using ARIMA models.
- ⚡ **Stationarity Checks**: Implementation of Augmented Dickey-Fuller test to ensure data stationarity.
- 🔒 **Artifact Detection**: Methods to identify and mitigate artifacts in EEG data.
- 🎨 **Visualization**: Clear and informative visualizations of time-series data, model forecasts, and residuals.
- 🛠️ **Customizable**: Modular code structure allows for easy customization and extension.
- 📈 **Forecasting**: Predict future EEG data values using fitted ARIMA models.

## 🎬 Demo

### Screenshots
![EEG Time Series Data](screenshots/eeg_time_series.png)
*Example of EEG time series data (ICP, AMP, RAP) before processing.*

![Stationarity Check](screenshots/stationarity_check.png)
*Augmented Dickey-Fuller test results for stationarity assessment.*

![ARIMA Model Forecast](screenshots/arima_forecast.png)
*ARIMA model forecast with confidence intervals.*

## 🚀 Quick Start

Clone the repository and run the main script:

```bash
git clone https://github.com/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data.git
cd ARIMA-Modeling-on-EEG-Time-Series-Data
pip install -r requirements.txt
python main.py
```

## 📦 Installation

### Prerequisites
- Python 3.7+
- pip
- Git

### Steps:

```bash
# Clone the repository
git clone https://github.com/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data.git
cd ARIMA-Modeling-on-EEG-Time-Series-Data

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load EEG data
data = pd.read_csv('data/eeg_data.csv', index_col='Timestamp')

# Fit ARIMA model
model = ARIMA(data['ICP'], order=(5, 1, 0))
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(data)-30, end=len(data)-1)

# Evaluate model
rmse = mean_squared_error(data['ICP'][-30:], predictions, squared=False)
print(f'RMSE: {rmse}')
```

### Advanced Examples

```python
# Example of grid search for optimal ARIMA parameters
import itertools

p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))

best_rmse = float('inf')
best_order = None

for order in pdq:
    try:
        model = ARIMA(data['ICP'], order=order)
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(data)-30, end=len(data)-1)
        rmse = mean_squared_error(data['ICP'][-30:], predictions, squared=False)
        if rmse < best_rmse:
            best_rmse = rmse
            best_order = order
        print(f'ARIMA{order} - RMSE: {rmse}')
    except:
        continue

print(f'Best ARIMA order: {best_order} with RMSE: {best_rmse}')
```

## ⚙️ Configuration

### Configuration File
The project uses a `config.json` file to store configuration parameters.

```json
{
  "data_path": "data/eeg_data.csv",
  "arima_order": [5, 1, 0],
  "test_size": 30,
  "icp_column": "ICP"
}
```

## 📁 Project Structure

```
ARIMA-Modeling-on-EEG-Time-Series-Data/
├── data/                  # EEG data files
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/               # Utility scripts
├── screenshots/           # Images for documentation
├── config.json            # Configuration file
├── main.py                # Main script
├── README.md              # Project documentation
├── requirements.txt       # Project dependencies
└── LICENSE                # License file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) (placeholder) for details.

### Quick Contribution Steps
1. 🍴 Fork the repository
2. 🌟 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. ✅ Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

### Development Setup
```bash
# Fork and clone the repo
git clone https://github.com/yourusername/ARIMA-Modeling-on-EEG-Time-Series-Data.git

# Install dependencies
pip install -r requirements.txt

# Create a new branch
git checkout -b feature/your-feature-name

# Make your changes and test

# Commit and push
git commit -m "Description of changes"
git push origin feature/your-feature-name
```

## Testing

To run the tests, execute the following command:

```bash
python -m unittest discover -s tests
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

## 💬 Support

- 📧 **Email**: your.email@example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data/issues)
- 📖 **Documentation**: [Full Documentation](https://docs.your-site.com) (placeholder)

## 🙏 Acknowledgments

- 📚 **Libraries used**:
  - [NumPy](https://numpy.org/) - Numerical computing library
  - [Pandas](https://pandas.pydata.org/) - Data analysis library
  - [Matplotlib](https://matplotlib.org/) - Visualization library
  - [Statsmodels](https://www.statsmodels.org/stable/index.html) - Statistical modeling library
  - [Scikit-learn](https://scikit-learn.org/stable/) - Machine learning library
- 👥 **Contributors**: Thanks to all [contributors](https://github.com/Abrar-Islam-Oitijjho/ARIMA-Modeling-on-EEG-Time-Series-Data/contributors)
```
