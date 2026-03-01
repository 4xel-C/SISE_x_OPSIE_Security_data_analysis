from pathlib import Path

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

        filepath = self.base_dir / self._data_folder / filename

        if not filepath.exists():
            raise ValueError(f"The file {filepath} does not exist")

        self.parser = Parser()

        self.df_raw = self.parser.load_raw_data(filepath)
        self.df = self.parser.generate_aggregated_data(self.df_raw)
