import pandas as pd


class parser:
    """Class to parse security logs files as pandas dataframes."""

    def __init__(self, filename):
        self.filename = filename

        self.df_raw = pd.read_csv(self.filename, nrows=1)

    def parse(self): ...
