import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# ML / evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold


class ArtifactDetector:
    """Pipeline for creating windowed residual-based features and training a classic
    ML classifier for artifact detection.
    """

    def sliding_windows(self, df: pd.DataFrame, parameters: List[str], window_size: int, step_size: int):
        """Return list of DataFrame windows, corresponding timestamp ranges, and index ranges.

        df expected to have a 'DateTime' column and to be pre-sorted chronologically.
        """
        df_clean = df.dropna(subset=parameters).reset_index(drop=True)
        n = len(df_clean)
        windows = []
        timestamps = []
        indices = []

        # Centered windows: i iterates over window centers using step_size
        for center in range(0, n, step_size):
            start = max(0, center - window_size // 2)
            end = min(n, start + window_size)
            window = df_clean.loc[start:end - 1, parameters]
            windows.append(window)
            timestamps.append(df_clean.loc[start:end - 1, 'DateTime'])
            indices.append((start, end))

            # Stop when we reach the end
            if end == n:
                break

        return windows, timestamps, indices

    def get_window_labels(self, df_artifact: pd.DataFrame, timestamps: List[pd.Series], threshold: float) -> List[int]:
        """Compute binary labels for each window: 1 if proportion of artifact timestamps
        inside window exceeds threshold, else 0.
        """
        # Ensure DateTime type
        if not pd.api.types.is_datetime64_any_dtype(df_artifact['DateTime']):
            try:
                df_artifact['DateTime'] = pd.to_datetime(df_artifact['DateTime'])
            except Exception:
                pass

        artifact_times = df_artifact['DateTime'].sort_values()
        labels = []

        for ts in timestamps:
            if ts.empty:
                labels.append(0)
                continue

            start_time = ts.iloc[0]
            end_time = ts.iloc[-1]
            mask = (artifact_times >= start_time) & (artifact_times <= end_time)
            artifact_count = mask.sum()
            window_label = int((artifact_count / len(ts)) > threshold)
            labels.append(window_label)

        return labels

    def get_features(self, file_directory_res: str, file_directory_art: str, patients_list: List[str],
                     parameters: List[str], window_size: int, step_size: int, window_threshold: float) -> Dict[str, Any]:
        """For each patient file in patients_list, read residuals and artifact timestamp file,
        concatenate them, create sliding windows and compute per-window variance for each parameter.

        Returns a dict with 'Features' (DataFrame), 'Residuals' (concatenated residuals DataFrame),
        'Window_timestamps' and 'Window_indices'.
        """
        all_residuals = []
        all_artifacts = []

        for fname in patients_list:
            # Expectation: fname is a filename (e.g. 'TBI_001_non_clean.csv') or just 'TBI_001.csv'
            base = os.path.splitext(os.path.basename(fname))[0][:7]

            # Residuals file is expected to follow a naming convention
            res_file = base + '_non_clean.csv' if not fname.endswith('_non_clean.csv') else fname
            res_path = os.path.join(file_directory_res, res_file)
            if not os.path.exists(res_path):
                # Try using fname directly
                res_path = os.path.join(file_directory_res, fname)
                if not os.path.exists(res_path):
                    continue

            df_res = pd.read_csv(res_path)
            df_res = df_res.dropna(subset=parameters).reset_index(drop=True)

            art_file = base + '_Artifact.csv'
            art_path = os.path.join(file_directory_art, art_file)
            if os.path.exists(art_path):
                df_art = pd.read_csv(art_path)
            else:
                df_art = pd.DataFrame(columns=['DateTime'])

            # Create a true_artifacts column in residuals (binary)
            artifact_times = set(df_art['DateTime']) if 'DateTime' in df_art.columns else set()
            df_res['true_artifacts'] = df_res['DateTime'].isin(artifact_times).astype(int)
            df_res['Patient'] = base

            all_residuals.append(df_res)
            all_artifacts.append(df_art)

        if len(all_residuals) == 0:
            raise ValueError('No residuals files found for the provided patients_list and directories.')

        df_residuals_all = pd.concat(all_residuals, ignore_index=True)
        df_artifacts_all = pd.concat(all_artifacts, ignore_index=True) if len(all_artifacts) > 0 else pd.DataFrame(columns=['DateTime'])

        windows, timestamps, indices = self.sliding_windows(df_residuals_all, parameters, window_size, step_size)

        # Compute per-parameter variance for each window; windows may be empty -> produce NaNs
        window_feature_dicts = []
        for w in windows:
            if w.empty:
                window_feature_dicts.append({f'{p}_var': np.nan for p in parameters})
            else:
                var_map = w.var().to_dict()
                # Ensure keys are parameter_var
                feature_row = {f'{p}_var': float(var_map.get(p, np.nan)) for p in parameters}
                window_feature_dicts.append(feature_row)

        # Build features DataFrame
        df_features = pd.DataFrame(window_feature_dicts)

        labels = self.get_window_labels(df_artifacts_all, timestamps, window_threshold)
        df_features['label'] = labels

        result = {
            'Features': df_features,
            'Residuals': df_residuals_all,
            'Window_timestamps': timestamps,
            'Window_indices': indices
        }
        return result

    @staticmethod
    def get_evaluation(y_actual: List[int], y_pred: List[int]) -> Dict[str, Any]:
        """Return a small dictionary of evaluation metrics used in the original pipeline.
        Expects binary labels with positive class labelled 1.
        """
        report = classification_report(y_actual, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_actual, y_pred)

        # cm shape (2,2) assumed; handle cases where only one class present
        fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
        recall = round(report.get('1', {}).get('recall', 0.0), 5)
        f1 = round(report.get('1', {}).get('f1-score', 0.0), 5)

        return {'FP': fp, 'Recall': recall, 'F1-score': f1}

    def map_window_predictions_to_samples(self, indices: List[Tuple[int, int]], y_pred: List[int], n_samples: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Map window-level predictions back to sample-level by averaging votes for each sample.

        Returns sample_prob (float array) and sample_labels (binary array).
        """
        sample_votes = np.zeros(n_samples, dtype=float)
        sample_counts = np.zeros(n_samples, dtype=float)

        for (start, end), pred in zip(indices, y_pred):
            # ensure indices are within range
            start = max(0, int(start))
            end = min(n_samples, int(end))
            if start >= end:
                continue
            sample_votes[start:end] += float(pred)
            sample_counts[start:end] += 1.0

        # Avoid division by zero for samples never covered by any window
        sample_counts[sample_counts == 0] = 1.0
        sample_prob = sample_votes / sample_counts
        sample_labels = (sample_prob >= threshold).astype(int)

        return sample_prob, sample_labels

    def artifact_detection_model_classic_ML(
        self,
        file_directory_res: str,
        file_directory_art: str,
        test_size: float,
        k: int,
        model,
        parameters: List[str],
        window_size: int,
        step_size: int,
        window_threshold: float,
        sample_threshold: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train-evaluate a classic ML model using k-fold across patients.

        The function returns two DataFrames: window-level and sample-level metrics
        aggregated across Train/CV/Test splits.
        """
        metrics_keys = ['FP', 'Recall', 'F1-score']

        all_window_metrics_train = {k: [] for k in metrics_keys}
        all_window_metrics_cv = {k: [] for k in metrics_keys}
        all_sample_metrics_train = {k: [] for k in metrics_keys}
        all_sample_metrics_cv = {k: [] for k in metrics_keys}

        # Treat files in the residuals directory as patient files
        all_patients = os.listdir(file_directory_res)
        train_patients, test_patients = train_test_split(all_patients, test_size=test_size, shuffle=True, random_state=42)
        train_patients = np.array(train_patients)

        # K-fold on training patients
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for fold, (train_idx, cv_idx) in enumerate(kf.split(train_patients)):
            patients_train = train_patients[train_idx].tolist()
            patients_cv = train_patients[cv_idx].tolist()

            # Build feature sets
            train_data = self.get_features(file_directory_res, file_directory_art, patients_train, parameters, window_size, step_size, window_threshold)
            df_features_train = train_data['Features']
            X_train = df_features_train.drop(columns=['label'])
            y_train = df_features_train['label']

            cv_data = self.get_features(file_directory_res, file_directory_art, patients_cv, parameters, window_size, step_size, window_threshold)
            df_features_cv = cv_data['Features']
            X_cv = df_features_cv.drop(columns=['label'])
            y_cv = df_features_cv['label']

            # Fit model
            model.fit(X_train, y_train)

            # Window-level evaluations
            y_pred_train = model.predict(X_train)
            m_train = self.get_evaluation(y_train, y_pred_train)
            for kk, vv in m_train.items():
                all_window_metrics_train[kk].append(vv)

            y_pred_cv = model.predict(X_cv)
            m_cv = self.get_evaluation(y_cv, y_pred_cv)
            for kk, vv in m_cv.items():
                all_window_metrics_cv[kk].append(vv)

            # Sample-level for train
            df_res_train = train_data['Residuals']
            sample_prob_train, sample_labels_train = self.map_window_predictions_to_samples(train_data['Window_indices'], y_pred_train, len(df_res_train), sample_threshold)
            df_res_train['pred_prob'] = sample_prob_train
            df_res_train['pred_label'] = sample_labels_train
            m_sample_train = self.get_evaluation(df_res_train['true_artifacts'], df_res_train['pred_label'])
            for kk, vv in m_sample_train.items():
                all_sample_metrics_train[kk].append(vv)

            # Sample-level for cv
            df_res_cv = cv_data['Residuals']
            sample_prob_cv, sample_labels_cv = self.map_window_predictions_to_samples(cv_data['Window_indices'], y_pred_cv, len(df_res_cv), sample_threshold)
            df_res_cv['pred_prob'] = sample_prob_cv
            df_res_cv['pred_label'] = sample_labels_cv
            m_sample_cv = self.get_evaluation(df_res_cv['true_artifacts'], df_res_cv['pred_label'])
            for kk, vv in m_sample_cv.items():
                all_sample_metrics_cv[kk].append(vv)

        # Test set
        test_data = self.get_features(file_directory_res, file_directory_art, test_patients, parameters, window_size, step_size, window_threshold)
        df_features_test = test_data['Features']
        X_test = df_features_test.drop(columns=['label'])
        y_test = df_features_test['label']

        y_pred_test = model.predict(X_test)
        avg_window_metrics_test = self.get_evaluation(y_test, y_pred_test)

        df_res_test = test_data['Residuals']
        sample_prob_test, sample_labels_test = self.map_window_predictions_to_samples(test_data['Window_indices'], y_pred_test, len(df_res_test), sample_threshold)
        df_res_test['pred_prob'] = sample_prob_test
        df_res_test['pred_label'] = sample_labels_test
        avg_sample_metrics_test = self.get_evaluation(df_res_test['true_artifacts'], df_res_test['pred_label'])

        # Average metrics across folds
        def avg_dict(d):
            return {k: (sum(v) / len(v) if len(v) > 0 else 0.0) for k, v in d.items()}

        avg_window_metrics_train = avg_dict(all_window_metrics_train)
        avg_window_metrics_cv = avg_dict(all_window_metrics_cv)
        avg_sample_metrics_train = avg_dict(all_sample_metrics_train)
        avg_sample_metrics_cv = avg_dict(all_sample_metrics_cv)

        df_window_metrics = pd.DataFrame({
            'Train': avg_window_metrics_train,
            'CV': avg_window_metrics_cv,
            'Test': avg_window_metrics_test
        }).T.round(3)

        df_sample_metrics = pd.DataFrame({
            'Train': avg_sample_metrics_train,
            'CV': avg_sample_metrics_cv,
            'Test': avg_sample_metrics_test
        }).T.round(3)

        # Ensure FP is integer where appropriate
        if 'FP' in df_window_metrics.columns:
            df_window_metrics['FP'] = df_window_metrics['FP'].astype(int)
        if 'FP' in df_sample_metrics.columns:
            df_sample_metrics['FP'] = df_sample_metrics['FP'].astype(int)

        return df_window_metrics, df_sample_metrics


# End of module

if __name__ == '__main__':
    # Small self-test / usage example (won't run when imported)
    print('the_functions module loaded. Import classes in your notebook:')
    print('from the_functions import DataPreprocessor, ARIMAModeler, ArtifactAnalyzer, ArtifactDetector')
