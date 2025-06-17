from abc import ABCMeta, abstractmethod
import numpy as np
import numba
from numba import float64, int64
from events import SignalEvent
from sklearn.preprocessing import StandardScaler

class Strategy(metaclass=ABCMeta):
    @abstractmethod
    def calculate_signals(self, event):
        raise NotImplementedError("Should implement calculate_signals()")

@numba.jit(nopython=True, cache=True)
def calculate_indicators(prices, window):
    """Calculate technical indicators using Numba"""
    n = len(prices)
    returns = np.zeros(n, dtype=np.float64)
    sma = np.zeros(n, dtype=np.float64)
    std = np.zeros(n, dtype=np.float64)
    zscore = np.zeros(n, dtype=np.float64)
    
    # Vectorized returns calculation
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # Vectorized SMA calculation
    for i in range(window, n):
        window_data = prices[i-window:i]
        sma[i] = np.mean(window_data)
        std[i] = np.std(window_data)
        if std[i] != 0:
            zscore[i] = (prices[i] - sma[i]) / std[i]
    
    return returns, sma, std, zscore

@numba.jit(nopython=True, cache=True)
def generate_signals(prices, sma, std, zscore, window):
    """Generate trading signals using Numba"""
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Pre-calculate mean of std for volatility breakout
    std_mean = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        std_mean[i] = np.mean(std[i-window:i])
    
    # Vectorized signal generation
    for i in range(window, n):
        signal_strength = 0.0
        
        # Mean reversion signal based on z-score
        if zscore[i] < -2.0:  # Strong oversold
            signal_strength += 1.0
        elif zscore[i] > 2.0:  # Strong overbought
            signal_strength -= 1.0
            
        # Price relative to SMA
        if prices[i] < sma[i] * 0.97:  # Price significantly below SMA
            signal_strength += 0.5
        elif prices[i] > sma[i] * 1.03:  # Price significantly above SMA
            signal_strength -= 0.5
            
        # Volatility breakout
        if std[i] > std_mean[i] * 1.2:  # High volatility
            if prices[i] > sma[i]:  # Price above SMA in high volatility
                signal_strength -= 0.5
            else:  # Price below SMA in high volatility
                signal_strength += 0.5
                
        signals[i] = signal_strength
    
    return signals

class NumbaStrategy(Strategy):
    def __init__(self, bars, events, symbol, window=20):
        self.bars = bars
        self.events = events
        self.symbol = symbol
        self.window = window
        self.bought = False
        self.historical_data = []
        self.min_samples_for_trading = window * 2
        self.position_size = 100  # Base position size
        self._last_signal = 0.0
        self._signal_threshold = 0.5

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            # Get latest prices
            close_prices = self.bars.get_latest_bars_values(self.symbol, 'Close', N=self.window + 1)
            
            if len(close_prices) < self.window + 1:
                return
                
            # Store historical data
            self.historical_data.extend(close_prices)
            
            # Only start trading after we have enough data
            if len(self.historical_data) < self.min_samples_for_trading:
                return
                
            # Convert to numpy array
            prices = np.array(self.historical_data, dtype=np.float64)
            
            # Calculate technical indicators using Numba
            returns, sma, std, zscore = calculate_indicators(prices, self.window)
            
            # Generate signals using Numba
            signals = generate_signals(prices, sma, std, zscore, self.window)
            
            # Get the latest signal
            latest_signal = signals[-1]
            
            # Only generate new signal if it crosses threshold
            if abs(latest_signal - self._last_signal) >= self._signal_threshold:
                # Calculate position size based on signal strength
                position_size = int(self.position_size * abs(latest_signal))
                
                # Generate trading signals
                if not self.bought and latest_signal > self._signal_threshold:  # Strong buy signal
                    signal = SignalEvent(self.symbol, None, 'LONG', abs(latest_signal))
                    self.events.put(signal)
                    self.bought = True
                    self._last_signal = latest_signal
                    
                elif self.bought and latest_signal < -self._signal_threshold:  # Strong sell signal
                    signal = SignalEvent(self.symbol, None, 'EXIT', abs(latest_signal))
                    self.events.put(signal)
                    self.bought = False
                    self._last_signal = latest_signal

class MeanReversionStrategy(Strategy):
    def __init__(self, bars, events, symbol, window=20):
        self.bars = bars
        self.events = events
        self.symbol = symbol
        self.window = window
        self.bought = False

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            close_prices = self.bars.get_latest_bars_values(self.symbol, 'Close', N=self.window)

            if len(close_prices) < self.window:
                return

            avg_close = np.mean(close_prices)
            latest_close = close_prices[-1]

            if not self.bought and latest_close < 0.97 * avg_close:
                signal = SignalEvent(self.symbol, None, 'LONG', 1.0)
                self.events.put(signal)
                self.bought = True

            elif self.bought and latest_close > avg_close:
                signal = SignalEvent(self.symbol, None, 'EXIT', 1.0)
                self.events.put(signal)
                self.bought = False 