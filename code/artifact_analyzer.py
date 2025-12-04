from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# scipy
from scipy import stats



class ArtifactAnalyzer:
    """Utilities to derive artifact timestamps and compare distributions of features
    between clean and artifact sets.
    """

    @staticmethod
    def get_true_artifacts(df_clean: pd.DataFrame, df_non_clean: pd.DataFrame) -> pd.DataFrame:
        """Return rows from df_non_clean whose timestamps are not present in df_clean.
        Both inputs should contain a 'DateTime' column; this function will parse to
        pandas datetime if necessary.
        """
        # Ensure datetime
        for df in (df_clean, df_non_clean):
            if not pd.api.types.is_datetime64_any_dtype(df['DateTime']):
                try:
                    df['DateTime'] = pd.to_datetime(df['DateTime'])
                except Exception:
                    # If parse fails, leave as-is
                    pass

        # Rows present in non_clean but not in clean
        clean_times = set(df_clean['DateTime'])
        mask = ~df_non_clean['DateTime'].isin(clean_times)
        return df_non_clean.loc[mask].copy()

    @staticmethod
    def mann_whitney_u_test(map_clean: Dict[str, List[float]], map_art: Dict[str, List[float]]):
        """Run Mann-Whitney U test for each key in map_clean vs map_art.
        Expects the same keys for both maps.
        Prints results.
        """
        for param in map_clean:
            a = np.array(map_clean[param])
            b = np.array(map_art.get(param, []))
            # If one group is empty, skip
            if a.size == 0 or b.size == 0:
                print(f"Skipping {param}: empty group")
                continue
            u_stat, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
            print(f"Mann-Whitney U Test for {param} — p-value: {p_value:.5f}")

    @staticmethod
    def visualize_scatterplot(map_clean: Dict[str, List[float]], map_art: Dict[str, List[float]], param_names: List[str] = None):
        """Scatter plots comparing clean vs artifact lists side-by-side for each key.
        """
        for param in map_clean:
            x = np.arange(len(map_clean[param]))
            plt.figure(figsize=(8, 4))
            plt.scatter(x, map_clean[param], label='Clean', marker='o')
            plt.scatter(x, map_art.get(param, []), label='Artifact', marker='x')
            plt.xlabel('Samples')
            plt.ylabel(param)
            plt.title(f'Scatter: {param}')
            plt.legend()
            plt.tight_layout()
            plt.show()

    @staticmethod
    def visualize_histogram(map_clean: Dict[str, List[float]], map_art: Dict[str, List[float]], bins: int = 10):
        for param in map_clean:
            plt.figure(figsize=(8, 4))
            plt.hist(map_clean[param], bins=bins, alpha=0.6, label='Clean')
            plt.hist(map_art.get(param, []), bins=bins, alpha=0.6, label='Artifact')
            plt.xlabel(param)
            plt.ylabel('Frequency')
            plt.title(f'Histogram: {param}')
            plt.legend()
            plt.tight_layout()
            plt.show()

    @staticmethod
    def visualize_boxplots(map_clean: Dict[str, List[float]], map_art: Dict[str, List[float]]):
        # Create side-by-side boxplots for each parameter
        for param in map_clean:
            data = [map_clean[param], map_art.get(param, [])]
            plt.figure(figsize=(6, 4))
            plt.boxplot(data, labels=['Clean', 'Artifact'])
            plt.title(f'Boxplot: {param}')
            plt.tight_layout()
            plt.show()