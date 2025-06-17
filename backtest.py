import time
from queue import Queue
from datetime import datetime
import yfinance as yf
import os
import pandas as pd
import psutil
import numpy as np
from memory_profiler import profile
import numba
from functools import lru_cache
from collections import deque
import warnings
from typing import Dict, List, Tuple

# Suppress Numba warnings
warnings.filterwarnings('ignore', category=numba.NumbaWarning)

print("Hello World")

@numba.jit(nopython=True, cache=True)
def process_market_data_batch(values: np.ndarray, window_size: int = 20) -> np.ndarray:
    """Process market data in batches using Numba"""
    n = len(values)
    returns = np.zeros(n, dtype=np.float64)
    sma = np.zeros(n, dtype=np.float64)
    std = np.zeros(n, dtype=np.float64)
    
    # Vectorized returns calculation
    returns[1:] = (values[1:] - values[:-1]) / values[:-1]
    
    # Vectorized SMA and std calculation
    for i in range(window_size, n):
        window_data = values[i-window_size:i]
        sma[i] = np.mean(window_data)
        std[i] = np.std(window_data)
    
    return np.column_stack((values, returns, sma, std))

@numba.jit(nopython=True, cache=True)
def calculate_performance_metrics(signals: int, orders: int, fills: int, execution_time: float) -> np.ndarray:
    """Calculate performance metrics using Numba"""
    return np.array([
        signals,
        orders,
        fills,
        execution_time,
        execution_time / max(1, signals)
    ], dtype=np.float64)

class Backtest:
    def __init__(self, csv_dir, symbol_list, initial_capital, heartbeat, start_date, data_handler, execution_handler, portfolio, strategy):
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.portfolio = portfolio
        self.strategy = strategy
        self.events = Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        self.memory_usage = deque(maxlen=1000)
        self.start_time = None
        self.end_time = None
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self._market_data_cache = {}
        self._bar_cache = {}
        self._batch_size = 100  # Process data in batches
        self._initialize_caches()

    def _initialize_caches(self):
        """Initialize caches for better performance"""
        self._market_data_cache = {symbol: deque(maxlen=1000) for symbol in self.symbol_list}
        self._bar_cache = {symbol: {} for symbol in self.symbol_list}

    @lru_cache(maxsize=1024)
    def _get_latest_bar_value(self, symbol: str, field: str) -> float:
        """Cache frequently accessed bar values"""
        if symbol in self._bar_cache and field in self._bar_cache[symbol]:
            return self._bar_cache[symbol][field]
        
        value = self.data_handler.get_latest_bar_value(symbol, field)
        self._bar_cache[symbol][field] = value
        return value

    def _generate_trading_instances(self):
        """Initialize trading components with error handling"""
        try:
            self.data_handler = self.data_handler(self.events, self.csv_dir, self.symbol_list)
            self.strategy = self.strategy(self.data_handler, self.events, self.symbol_list[0])
            self.portfolio = self.portfolio(
                data_handler=self.data_handler,
                events=self.events,
                start_date=self.start_date,
                initial_capital=self.initial_capital
            )
            self.execution_handler = self.execution_handler(self.events, self.data_handler)
        except Exception as e:
            print(f"Error initializing trading instances: {str(e)}")
            raise

    def _track_memory(self):
        """Track memory usage efficiently"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                'timestamp': datetime.now(),
                'rss': memory_info.rss / 1024 / 1024,
                'vms': memory_info.vms / 1024 / 1024
            }
        except Exception:
            return None

    def _process_event_batch(self, events: List) -> None:
        """Process a batch of events efficiently"""
        for event in events:
            if event is None:
                continue

            try:
                if event.type == 'MARKET':
                    self.strategy.calculate_signals(event)
                elif event.type == 'SIGNAL':
                    self.signals += 1
                    order = self.portfolio.generate_order(event)
                    if order:
                        self.events.put(order)
                elif event.type == 'ORDER':
                    self.orders += 1
                    self.execution_handler.execute_order(event)
                elif event.type == 'FILL':
                    self.fills += 1
                    self.portfolio.update_fill(event)
            except Exception as e:
                print(f"Error processing event: {str(e)}")

    def _update_portfolio_batch(self, market_data_batch: Dict[str, List[float]], datetimes: List[datetime]) -> None:
        """Update portfolio with a batch of market data"""
        for i, dt in enumerate(datetimes):
            current_market_data = {symbol: prices[i] for symbol, prices in market_data_batch.items()}
            self.portfolio.update_timeindex(dt, current_market_data)

    def _run_backtest(self):
        """Run the backtest with optimized batch processing"""
        print("\nInitializing backtest...")
        self._generate_trading_instances()
        
        self.start_time = time.time()
        
        try:
            while True:
                if not self.data_handler.continue_backtest:
                    break

                # Process events in batches
                events_batch = []
                while len(events_batch) < self._batch_size:
                    try:
                        event = self.events.get(False)
                        events_batch.append(event)
                    except:
                        break
                
                if events_batch:
                    self._process_event_batch(events_batch)

                # Update market data in batches
                market_data_batch = {symbol: [] for symbol in self.symbol_list}
                datetimes = []
                
                for _ in range(self._batch_size):
                    if not self.data_handler.continue_backtest:
                        break
                        
                    self.data_handler.update_bars()
                    latest_datetime = self.data_handler.get_latest_bar_datetime(self.symbol_list[0])
                    
                    if latest_datetime is None:
                        break
                        
                    datetimes.append(latest_datetime)
                    for symbol in self.symbol_list:
                        market_data_batch[symbol].append(self._get_latest_bar_value(symbol, "Close"))
                
                if market_data_batch[self.symbol_list[0]]:
                    # Process market data in batches
                    for symbol in self.symbol_list:
                        values = np.array(market_data_batch[symbol], dtype=np.float64)
                        processed_data = process_market_data_batch(values)
                        market_data_batch[symbol] = processed_data[:, 0]  # Keep only prices
                    
                    # Update portfolio with batch data
                    self._update_portfolio_batch(market_data_batch, datetimes)
                
                # Track memory usage
                memory_usage = self._track_memory()
                if memory_usage:
                    self.memory_usage.append(memory_usage)
                
                if self.heartbeat > 0:
                    time.sleep(self.heartbeat)

        except Exception as e:
            print(f"Error during backtest: {str(e)}")
            raise
        finally:
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            metrics = calculate_performance_metrics(
                self.signals, self.orders, self.fills, execution_time
            )
            print(f"\nBacktest completed in {metrics[3]:.2f} seconds")
            print(f"Signals: {metrics[0]}, Orders: {metrics[1]}, Fills: {metrics[2]}")

    def simulate_trading(self):
        """Run the trading simulation with error handling"""
        try:
            print("\nStarting backtest simulation...")
            self._run_backtest()
            print("\nBacktest completed, generating performance metrics...")
            self._output_performance()
            self._output_memory_stats()
            self._output_timing_stats()
            print("\nBacktest simulation finished")
        except Exception as e:
            print(f"Error during simulation: {str(e)}")
            raise

    @lru_cache(maxsize=32)
    def _output_performance(self):
        """Output performance metrics with caching"""
        print("\nGenerating performance metrics...")
        self.portfolio.output_summary_stats()

    def _output_memory_stats(self):
        """Output memory statistics efficiently"""
        if not self.memory_usage:
            return

        try:
            memory_df = pd.DataFrame(self.memory_usage)
            print(f"\nPeak Memory Usage: {memory_df['rss'].max():.2f} MB")
        except Exception as e:
            print(f"Error outputting memory stats: {str(e)}")

    def _output_timing_stats(self):
        """Output timing statistics with error handling"""
        if self.start_time is None or self.end_time is None:
            return

        try:
            execution_time = self.end_time - self.start_time
            print(f"Average Time per Signal: {execution_time/max(1, self.signals):.4f} seconds")
        except Exception as e:
            print(f"Error outputting timing stats: {str(e)}")

    @lru_cache(maxsize=128)
    def _open_convert_csv_files(self):
        """Open and convert CSV files with caching"""
        try:
            for s in self.symbol_list:
                path = os.path.join(self.csv_dir, f"{s}.csv")
                df = pd.read_csv(path, index_col="Date", parse_dates=True)
                self.symbol_data[s] = df.itertuples()
                self.latest_symbol_data[s] = []
        except Exception as e:
            print(f"Error opening CSV files: {str(e)}")
            raise

def download_stock_data(symbol, period="max"):
    """Download stock data with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        hist.to_csv(f"{symbol}.csv")
        return hist
    except Exception as e:
        print(f"Error downloading stock data: {str(e)}")
        raise

if __name__ == "__main__":
    from data_handler import HistoricCSVDataHandler
    from execution import SimulatedExecutionHandler
    from portfolio import Portfolio
    from strategy import NumbaStrategy
    
    try:
        print('backe')
        backtest = Backtest(
            csv_dir=".",
            symbol_list=["MSFT"],
            initial_capital=100000.0,
            heartbeat=0.0,
            start_date=datetime(2010, 1, 1),
            data_handler=HistoricCSVDataHandler,
            execution_handler=SimulatedExecutionHandler,
            portfolio=Portfolio,
            strategy=NumbaStrategy
        )
        print('backtest created')
        backtest.simulate_trading() 
        print('backtest finished')
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise