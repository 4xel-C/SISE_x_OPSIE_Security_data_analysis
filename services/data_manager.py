from pathlib import Path

import pandas as pd
from pandas import DataFrame, Timestamp

from features import Parser


class DataManager:
    """Singleton class to manage data."""

    _instance = None
    _data_folder = "data"

    def __new__(cls, filename):
        """Singleton pattern to ensure only one instance of the data manager is created"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__(filename)
        return cls._instance

    def __init__(self, filename):
        self.base_dir = Path(__file__).parent.parent

        self.filepath = self.base_dir / self._data_folder / filename

        if not self.filepath.exists():
            raise ValueError(f"The file {self.filepath} does not exist")

        self.parser = Parser()

        self.df_raw = self.load_raw_data(self.filepath)
        self.df = self.parser.generate_aggregated_data(self.df_raw)

    @property
    def time_range(self) -> tuple[Timestamp, Timestamp]:
        """Returns the (min, max) timestamps available in the raw data."""
        return self.df_raw["date"].min(), self.df_raw["date"].max()
    
    def search_ipsrc(self, ipsrc:str|None=None) -> DataFrame:
        """
        Search data by ipsrc, return all data if no ipsrc

        Args:
            ipsrc (str, optional): ipsrc to search. Defaults to None.

        Returns:
            DataFrame: Filtered dataframe
        """
        if not ipsrc:
            return self.df_raw
        
        return self.df_raw[self.df_raw['ipsrc'] == ipsrc]

    def get_filtered_df(self, start: Timestamp, end: Timestamp) -> DataFrame:
        """Return an aggregated DataFrame restricted to the [start, end] time window.

        Args:
            start: Start of the time window (inclusive).
            end: End of the time window (inclusive).

        Returns:
            Aggregated DataFrame with feature engineering applied on the filtered window.
        """
        mask = (self.df_raw["date"] >= start) & (self.df_raw["date"] <= end)
        df_window = self.df_raw[mask]
        return self.parser.generate_aggregated_data(df_window)

    def load_raw_data(self, filepath: Path) -> DataFrame:
        """Load a security log dataframe from a file name.

        Args:
            filepath (Path): Path to the csv file containing the raw security log data.

        Raises:
            ValueError: _description_

        Returns:
            DataFrame: _description_
        """
        df_raw = pd.read_csv(filepath)

        # set the columns
        df_raw.columns = [
            "ipsrc",
            "ipdst",
            "portdst",
            "proto",
            "action",
            "date",
            "regle",
        ]

        df_raw["date"] = pd.to_datetime(df_raw["date"])

        return df_raw
