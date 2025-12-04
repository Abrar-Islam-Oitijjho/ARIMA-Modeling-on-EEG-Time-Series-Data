import os
import datetime
from typing import List, Dict

import pandas as pd


class DataPreprocessor:
    """
    Helpers to load, clean, resample and save time-series patient files.
    Expects CSVs with an Excel-style 'DateTime' numeric serial column or an
    existing datetime column; column names for physiological signals should be
    standard (e.g. 'ICP', 'MAP', 'AMP', 'RAP').
    """

    def __init__(self, directory: str):
        self.directory = directory

    @staticmethod
    def _excel_serial_to_datetime(serial: float) -> datetime.datetime:
        """Convert Excel serial (float/int) to Python datetime.

        Excel uses 1899-12-30 base for serial when files exported from Excel.
        If a value is already datetime-like, callers should convert prior to
        calling this function.
        """
        return datetime.datetime(1899, 12, 30) + datetime.timedelta(days=float(serial))

    def save_new_file(self, df_new: pd.DataFrame, folder_name: str, file_name: str) -> str:
        """Save dataframe to new folder (sibling of self.directory).

        Returns path to saved file.
        """
        parent_path = os.path.dirname(self.directory)
        output_folder = os.path.join(parent_path, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, file_name)
        df_new.to_csv(output_file_path, index=False)
        return output_file_path

    def remove_artifacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove unrealistic physiologic values from data and return cleaned df.

        Limits are heuristic and can be adjusted.
        """
        if df is None or df.shape[0] == 0:
            return df

        cond = (
            (df['ICP'] > 100) |
            (df['ICP'] < -15) |
            (df['MAP'] > 200) |
            (df['MAP'] < 0)
        )
        return df.loc[~cond].copy()

    def change_resolution_and_save(self, file: str, resolution: str) -> pd.DataFrame:
        """Read CSV, parse DateTime, resample to given pandas offset alias (e.g. '1T')
        and save the resampled file to a sibling folder named by resolution.

        Returns the resampled DataFrame with a datetime index.
        """
        file_path = os.path.join(self.directory, file)
        df = pd.read_csv(file_path)

        # If 'DateTime' looks numeric, convert using excel serial; otherwise try parsing
        if pd.api.types.is_numeric_dtype(df['DateTime']):
            df['datetime'] = df['DateTime'].apply(self._excel_serial_to_datetime)
        else:
            df['datetime'] = pd.to_datetime(df['DateTime'])

        df = df.set_index('datetime')
        # Drop the original DateTime numeric/string column to avoid duplication
        df = df.drop(columns=['DateTime'], errors='ignore')
        df = df.dropna()
        df = self.remove_artifacts(df)

        # Resample using pandas' .resample; use mean aggregation
        df_resampled = df.resample(resolution).mean()

        # Save file: keep same filename but into parent/<resolution>/<filename>
        self.save_new_file(df_resampled.reset_index(), resolution, os.path.basename(file))

        return df_resampled

    def rename_cols_and_save(self, file: str, rename_dict: Dict[str, str], out_folder: str = "New") -> str:
        file_path = os.path.join(self.directory, file)
        df = pd.read_csv(file_path)
        df.rename(columns=rename_dict, inplace=True)
        return self.save_new_file(df, out_folder, os.path.basename(file))

    def delete_cols_and_save(self, file: str, col_no: List[int], out_folder: str = "New") -> str:
        file_path = os.path.join(self.directory, file)
        df = pd.read_csv(file_path)
        # Drop by positional indices passed as list
        cols_to_drop = [df.columns[i] for i in col_no if i < len(df.columns)]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        return self.save_new_file(df, out_folder, os.path.basename(file))