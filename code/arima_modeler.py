import ast
import itertools
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# statsmodels
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# optimization
import optuna



class ARIMAModeler:
    """
    Collection of ARIMA utilities: stationarity tests, grid search, optuna,
    residual extraction, plotting and residual descriptive statistics.
    """

    def stationarity_check(self, df: pd.DataFrame, parameters: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run ADF and KPSS tests for listed parameters.

        Returns two dicts: adf_pvalues and kpss_pvalues keyed by parameter name.
        """
        df_ind = df.dropna().reset_index(drop=True)

        adf_map = {key: None for key in parameters}
        kpss_map = {key: None for key in parameters}

        for col in parameters:
            if col not in df_ind.columns:
                adf_map[col] = 'N/A'
                kpss_map[col] = 'N/A'
                continue

            series = df_ind[col].dropna()
            if series.empty:
                adf_map[col] = 'N/A'
                kpss_map[col] = 'N/A'
                continue

            try:
                res_adf = adfuller(series)
                adf_map[col] = float(res_adf[1])  # p-value
            except Exception:
                adf_map[col] = 'N/A'

            try:
                res_kpss = kpss(series, nlags="auto")
                kpss_map[col] = float(res_kpss[1])
            except Exception:
                kpss_map[col] = 'N/A'

        return adf_map, kpss_map

    def ARIMA_modeling_using_grid_search(
        self,
        df: pd.DataFrame,
        parameters: List[str],
        p_range: List[int],
        d_range: List[int],
        q_range: List[int],
        eval_metric: str = 'AIC'
    ) -> Tuple[Dict[str, Tuple[int, int, int]], Dict[str, float]]:
        """Grid search over p,d,q for each parameter and return best orders and scores.

        p_range, d_range, q_range should be [min, max] inclusive boundaries.
        eval_metric in {'AIC', 'BIC', 'LLF'} selects which metric to minimize.
        """
        df_ind = df
        best_order_map = {key: None for key in parameters}
        best_score_map = {key: None for key in parameters}

        for col in parameters:
            if col not in df_ind.columns:
                best_order_map[col] = (None, None, None)
                best_score_map[col] = np.nan
                continue

            series = df_ind[col].dropna()
            p_values = range(p_range[0], p_range[1] + 1)
            d_values = range(d_range[0], d_range[1] + 1)
            q_values = range(q_range[0], q_range[1] + 1)

            best_params = {'p': None, 'd': None, 'q': None}
            best_score = float('inf')
            total_comb = len(p_values) * len(d_values) * len(q_values)

            for p, d, q in tqdm(itertools.product(p_values, d_values, q_values), total=total_comb,
                                 desc=f"Grid Search {col}"):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()

                    if eval_metric == 'AIC':
                        curr_metric = fitted.aic
                    elif eval_metric == 'BIC':
                        curr_metric = fitted.bic
                    else:
                        curr_metric = fitted.llf

                    if curr_metric < best_score:
                        best_params = {'p': p, 'd': d, 'q': q}
                        best_score = curr_metric
                except Exception:
                    # Continue when model fitting fails for this combo
                    continue

            best_order_map[col] = (best_params['p'], best_params['d'], best_params['q'])
            best_score_map[col] = best_score

        return best_order_map, best_score_map

    def optimal_arima_using_optuna(
        self,
        df: pd.DataFrame,
        parameters: List[str],
        p_range: List[int],
        d_range: List[int],
        q_range: List[int],
        n_trials: int = 30,
        eval_metric: str = 'AIC'
    ) -> Tuple[Dict[str, Tuple[int, int, int]], Dict[str, float]]:
        """Use Optuna to find optimal p,d,q for each parameter.

        Returns maps of best orders and corresponding best metric value.
        """
        df_ind = df
        best_order_map = {key: None for key in parameters}
        best_score_map = {key: None for key in parameters}

        for col in parameters:
            if col not in df_ind.columns:
                best_order_map[col] = (None, None, None)
                best_score_map[col] = np.nan
                continue

            series = df_ind[col].dropna()
            if series.empty:
                best_order_map[col] = (None, None, None)
                best_score_map[col] = np.nan
                continue

            def objective(trial):
                p = trial.suggest_int('p', p_range[0], p_range[1])
                d = trial.suggest_int('d', d_range[0], d_range[1])
                q = trial.suggest_int('q', q_range[0], q_range[1])

                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()

                    if eval_metric == 'AIC':
                        return float(fitted.aic)
                    elif eval_metric == 'BIC':
                        return float(fitted.bic)
                    else:
                        return float(fitted.llf)
                except Exception:
                    # Return large value so the trial is discouraged
                    return float('inf')

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            bp = study.best_params if study.best_trial is not None else {'p': None, 'd': None, 'q': None}
            bv = study.best_value if study.best_trial is not None else np.nan

            best_order_map[col] = (bp.get('p'), bp.get('d'), bp.get('q'))
            best_score_map[col] = bv

        return best_order_map, best_score_map

    def calculate_residuals(self, df: pd.DataFrame, parameters: List[str], order_map: Dict[str, Tuple[int, int, int]]) -> Dict[str, pd.Series]:
        """Fit ARIMA for each parameter according to order_map and return residual series
        aligned to the original dataframe index. If fitting fails, a NaN series is returned.
        """
        residuals_map = {key: None for key in parameters}

        for col in parameters:
            if col not in df.columns or order_map.get(col) is None:
                residuals_map[col] = pd.Series(data=[np.nan] * len(df), index=df.index)
                continue

            try:
                series = df[col].dropna()
                order = order_map[col]
                model = ARIMA(series, order=order)
                fitted = model.fit()
                resid = pd.Series(index=series.index, data=fitted.resid)
                # Reindex to full df index so positions of NaN line up with original data
                residuals_map[col] = resid.reindex(df.index)
            except Exception:
                residuals_map[col] = pd.Series(data=[np.nan] * len(df), index=df.index)

        return residuals_map

    def plot_residuals_acf_pacf(self, residuals: pd.Series, parameter_name: str, lags: int = 40, alpha: float = 0.05) -> None:
        """Plot ACF and PACF side-by-side for a residual series.
        Handles missing data by dropping NaNs before plotting.
        """
        try:
            res = residuals.dropna()
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plot_acf(res, lags=lags, ax=plt.gca(), alpha=alpha, auto_ylims=True)
            plt.title(f'Residuals ACF - {parameter_name}')

            plt.subplot(1, 2, 2)
            plot_pacf(res, lags=lags, ax=plt.gca(), alpha=alpha, auto_ylims=True)
            plt.title(f'Residuals PACF - {parameter_name}')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Unable to plot ACF/PACF for {parameter_name}: {e}")

    def residuals_descriptive_stats(self, parameters: List[str], residuals_map: Dict[str, pd.Series]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Return mean(abs), median(abs) and variance of residuals per parameter.
        NaNs are ignored.
        """
        mean_map = {}
        median_map = {}
        var_map = {}

        for col in parameters:
            resid = residuals_map.get(col)
            if resid is None:
                mean_map[col] = np.nan
                median_map[col] = np.nan
                var_map[col] = np.nan
                continue

            clean = resid.dropna().values
            if clean.size == 0:
                mean_map[col] = np.nan
                median_map[col] = np.nan
                var_map[col] = np.nan
                continue

            mean_map[col] = float(np.mean(np.abs(clean)))
            median_map[col] = float(np.median(np.abs(clean)))
            var_map[col] = float(np.var(clean))

        return mean_map, median_map, var_map

    def get_optimal_orders(self, df_merged: pd.DataFrame, parameters: List[str]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Given a dataframe where columns contain stringified orders like "(p,d,q)",
        return maps of lists of p's and q's for each parameter.
        """
        p_orders = {key: [] for key in parameters}
        q_orders = {key: [] for key in parameters}

        for col in parameters:
            if col not in df_merged.columns:
                continue

            for cell in df_merged[col].dropna():
                try:
                    order = ast.literal_eval(cell)
                    if isinstance(order, (list, tuple)) and len(order) >= 3:
                        p_orders[col].append(order[0])
                        q_orders[col].append(order[2])
                except Exception:
                    continue

        return p_orders, q_orders