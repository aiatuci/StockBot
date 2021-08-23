###############################################################################
# A PyTorch dataset for the Huge Stock Market Dataset.                        #
###############################################################################

from datetime import datetime
import math
import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

__all__ = ['HSMD']


class HSMD(Dataset):
    """A PyTorch representation of the Huge Stock Market Dataset.
    
    Tracks daily stock data for 7195 stocks until 2017-11-10, across the 
    following metrics: date, opening price, high, low, closing price, 
        volume, and open interest.

    Requires dataset to be pre-downloaded from Kaggle.

    Dataset Link: https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
    """
    DAYS_IN_WEEK = 7
    DATE_FMT = "%Y-%m-%d"

    def __init__(
        self,
        dataset_dir: str,
        stocks: List[str],
        window_size: Optional[int] = DAYS_IN_WEEK,
        columns: Optional[List[str]] = None,
        transform: Optional[Callable] = None
    ):
        """Load the Huge Stock Market Dataset.

        Args:
            dataset_dir (str): Root directory of the downloaded dataset.
            stocks ([str]): Stock exchange tickers for the series of data to load.
            window_size (int): Days of data in each sliding window.
            columns ([str]): Columns to include from the DataFrame for each stock.
        """
        super(HSMD).__init__(HSMD, self)
        if not os.path.exists(dataset_dir) or not os.path.isdir(dataset_dir):
            raise ValueError(f"Dataset directory {dataset_dir} does not exist.")
        self._dataset_dir = dataset_dir
        self._window_size = window_size
        self._transform = transform

        # TODO: Add support for loading multiple stocks simultaneously
        assert len(stocks) == 1, f"HSMD currently only supports a single stock at a time."
        self._stocks = list(map(lambda s: s.lower(), stocks))
        stock_file = os.path.join(self._dataset_dir, "Stocks", f"{self._stocks[0]}.us.txt")
        if not os.path.exists(stock_file):
            raise ValueError(f"Data file for stock {stocks[0]} not found.")

        date_parser = lambda d: datetime.strptime(d, self.DATE_FMT)
        raw_df = pd.read_csv(stock_file, index_col=0, parse_dates=True, date_parser=date_parser)
        if columns:
            raw_df = raw_df[columns]
        self._df = raw_df
        self.columns = self._df.columns

    def __len__(self):
        """Get the length of the loaded series."""
        return math.ceil(len(self._df) / self._window_size)

    def __getitem__(self, idx):
        """Get a single data point from the loaded series."""
        data = self._df.iloc[idx:idx + self._window_size, :].values
        if self._transform:
            data = self._transform(data)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\'{self._dataset_dir}\', {self._stocks}, window_size={self._window_size})"


if __name__ == '__main__':
    TICKER = 'amzn'
    NUM_DAYS = 5153
    DAY_1_DATA = np.array([1.97, 1.98, 1.71, 1.73, 14700000, 0])
    ALSO_DAY_1_DATA = np.array([1126.1, 1131.75, 1124.06, 1125.35, 2179181, 0])

    dataset = HSMD('./data/HSMD', stocks=[TICKER], window_size=1)
    assert len(dataset) == NUM_DAYS, f"Should have {NUM_DAYS} days of {TICKER} stock data."
    day_1 = dataset[0]
    assert np.sum(np.abs(day_1 - DAY_1_DATA)) < 1e-9, f"Data for index 0 does not match raw values."
    also_day_1 = dataset[-1]
    assert np.sum(np.abs(also_day_1 - ALSO_DAY_1_DATA)) < 1e-9, f"Data for index -1 does not match raw values."
    print(f"Successfully loaded {dataset}, with {len(dataset)} windows.")
