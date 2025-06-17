from abc import ABCMeta, abstractmethod
import os
import pandas as pd
from events import MarketEvent
import numpy as np
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True, parallel=True)
def _process_batch_numba(data_array, start_idx, end_idx):
    """Numba-optimized batch processing"""
    result = np.zeros(end_idx - start_idx)
    for i in prange(end_idx - start_idx):
        result[i] = data_array[start_idx + i]
    return result

class DataHandler(metaclass=ABCMeta):
    @abstractmethod
    def get_latest_bar(self, symbol):
        raise NotImplementedError()

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        raise NotImplementedError()

    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        raise NotImplementedError()

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        raise NotImplementedError()

    @abstractmethod
    def get_latest_bars_values(self, symbol, val_type, N=1):
        raise NotImplementedError()

    @abstractmethod
    def update_bars(self):
        raise NotImplementedError()

class HistoricCSVDataHandler(DataHandler):
    def __init__(self, events, csv_dir, symbol_list):
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self._current_index = 0
        self._batch_size = 1000  # Increased batch size for better performance
        self.datetime_col = 'datetime'
        self.required_cols = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        self._column_mapping = {
            'open': ['open', 'Open', 'OPEN', 'o', 'O'],
            'high': ['high', 'High', 'HIGH', 'h', 'H'],
            'low': ['low', 'Low', 'LOW', 'l', 'L'],
            'close': ['close', 'Close', 'CLOSE', 'c', 'C', 'price', 'Price', 'PRICE'],
            'volume': ['volume', 'Volume', 'VOLUME', 'v', 'V', 'vol', 'Vol', 'VOL']
        }
        self._total_bars = 0
        self._data_cache = {}  # Cache for processed data
        self._open_convert_csv_files()
        self._preload_data()

    def _validate_dataframe(self, df, symbol):
        """Validate the dataframe has required columns"""
        # Check if datetime column exists
        if self.datetime_col not in df.columns and self.datetime_col not in df.index.names:
            raise ValueError(f"Datetime column '{self.datetime_col}' not found in {symbol}")
        
        # Get available columns
        available_cols = set(df.columns)
        
        # Check which required columns are missing
        missing_cols = []
        for std_name, actual_name in self.required_cols.items():
            if actual_name not in available_cols:
                missing_cols.append(actual_name)
                # Remove from required_cols if not present
                del self.required_cols[std_name]
        
        if missing_cols:
            print(f"Warning: Missing columns for {symbol}: {missing_cols}")
            print(f"Available columns: {list(available_cols)}")
            print(f"Continuing with available columns: {list(self.required_cols.values())}")

    def _open_convert_csv_files(self):
        for s in self.symbol_list:
            path = os.path.join(self.csv_dir, f"{s}.csv")
            print(f"\nLoading data from {path}")
            
            try:
                # First read the CSV to check columns
                df = pd.read_csv(path)
                print(f"Available columns: {df.columns.tolist()}")
                
                # Try to identify datetime column
                datetime_candidates = ['datetime', 'date', 'time', 'timestamp', 'Date', 'Time', 'DateTime']
                datetime_col = None
                
                for col in datetime_candidates:
                    if col in df.columns:
                        datetime_col = col
                        break
                
                if datetime_col is None:
                    raise ValueError(f"No datetime column found in {s}.csv. Available columns: {df.columns.tolist()}")
                
                # Set the datetime column
                self.datetime_col = datetime_col
                
                # Convert datetime column
                df[datetime_col] = pd.to_datetime(df[datetime_col])
                df.set_index(datetime_col, inplace=True)
                
                # Map columns to standard names
                mapped_cols = {}
                for std_name, possible_names in self._column_mapping.items():
                    for col in df.columns:
                        if col in possible_names:
                            mapped_cols[std_name] = col
                            break
                
                if not mapped_cols:
                    raise ValueError(f"No matching columns found for {s}. Available columns: {df.columns.tolist()}")
                
                self.required_cols = mapped_cols
                
                # Store the full DataFrame
                self.symbol_data[s] = df
                
                # Initialize latest data
                self.latest_symbol_data[s] = []
                
                print(f"Successfully loaded {len(df)} rows of data")
                print(f"Mapped columns: {mapped_cols}")
                
            except Exception as e:
                print(f"Error loading {s}: {str(e)}")
                continue

    def _preload_data(self):
        """Preload all data into memory with Numba optimization"""
        for symbol in self.symbol_list:
            if symbol not in self.symbol_data:
                continue
                
            df = self.symbol_data[symbol]
            # Convert to numpy arrays for faster access
            data_dict = {'datetime': df.index.values}
            
            # Pre-allocate numpy arrays for better memory management
            for key, col in self.required_cols.items():
                if col in df.columns:
                    data_dict[key] = np.ascontiguousarray(df[col].values, dtype=np.float64)
            
            self.symbol_data[symbol] = data_dict
            self.latest_symbol_data[symbol] = []
            self._total_bars = len(df)
            
            # Pre-compute and cache common calculations
            self._cache_common_calculations(symbol)

    def _cache_common_calculations(self, symbol):
        """Cache common calculations for faster access"""
        if symbol not in self.symbol_data:
            return
            
        data = self.symbol_data[symbol]
        cache = {}
        
        # Cache price changes
        if 'close' in data:
            close_prices = data['close']
            cache['price_changes'] = np.diff(close_prices)
            
        # Cache returns
        if 'close' in data:
            close_prices = data['close']
            cache['returns'] = np.diff(close_prices) / close_prices[:-1]
            
        self._data_cache[symbol] = cache

    def get_latest_bar(self, symbol):
        if not self.latest_symbol_data[symbol]:
            return None
        idx = len(self.latest_symbol_data[symbol]) - 1
        return {
            'datetime': self.symbol_data[symbol]['datetime'][idx],
            **{key: self.symbol_data[symbol][key][idx] 
               for key in self.required_cols.keys() 
               if key in self.symbol_data[symbol]}
        }

    def get_latest_bars(self, symbol, N=1):
        if not self.latest_symbol_data[symbol]:
            return []
        start_idx = max(0, len(self.latest_symbol_data[symbol]) - N)
        end_idx = len(self.latest_symbol_data[symbol])
        return [{
            'datetime': self.symbol_data[symbol]['datetime'][i],
            **{key: self.symbol_data[symbol][key][i] 
               for key in self.required_cols.keys() 
               if key in self.symbol_data[symbol]}
        } for i in range(start_idx, end_idx)]

    def get_latest_bar_datetime(self, symbol):
        if symbol not in self.latest_symbol_data:
            print(f"Warning: Symbol {symbol} not found in data")
            return None
        if not self.latest_symbol_data[symbol]:
            return None
        return self.symbol_data[symbol]['datetime'][len(self.latest_symbol_data[symbol]) - 1]

    def get_latest_bar_value(self, symbol, val_type):
        if not self.latest_symbol_data[symbol]:
            return None
        idx = len(self.latest_symbol_data[symbol]) - 1
        return self.symbol_data[symbol][val_type.lower()][idx]

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """Optimized bar value retrieval"""
        if not self.latest_symbol_data[symbol]:
            return []
            
        start_idx = max(0, len(self.latest_symbol_data[symbol]) - N)
        end_idx = len(self.latest_symbol_data[symbol])
        
        # Use Numba-optimized batch processing
        return _process_batch_numba(
            self.symbol_data[symbol][val_type.lower()],
            start_idx,
            end_idx
        )

    def update_bars(self):
        """Optimized bar updates with batch processing"""
        try:
            if self._current_index >= self._total_bars:
                self.continue_backtest = False
                return

            # Update all symbols in batch
            for symbol in self.symbol_list:
                if len(self.latest_symbol_data[symbol]) < self._total_bars:
                    self.latest_symbol_data[symbol].append(True)

            self._current_index += 1
            
            # Only put market event if we have new data
            if self._current_index % self._batch_size == 0:
                self.events.put(MarketEvent())
                
        except Exception as e:
            print(f"Error updating bars: {e}")
            self.continue_backtest = False

    def get_batch_data(self, symbol, val_type, batch_size=None):
        """Optimized batch data retrieval"""
        if batch_size is None:
            batch_size = self._batch_size
            
        if not self.latest_symbol_data[symbol]:
            return None
            
        start_idx = max(0, len(self.latest_symbol_data[symbol]) - batch_size)
        end_idx = len(self.latest_symbol_data[symbol])
        
        # Use Numba-optimized batch processing
        return _process_batch_numba(
            self.symbol_data[symbol][val_type.lower()],
            start_idx,
            end_idx
        ) 
