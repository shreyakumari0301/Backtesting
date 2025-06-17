from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime
from queue import Queue
from events import OrderEvent, FillEvent
import numba

@numba.jit(nopython=True, cache=True)
def calculate_returns(prices):
    """Calculate returns using numba for speed"""
    returns = np.zeros_like(prices, dtype=np.float64)
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    return returns

@numba.jit(nopython=True, cache=True)
def calculate_equity_curve(returns):
    """Calculate equity curve using numba for speed"""
    return np.cumprod(1.0 + returns)

@numba.jit(nopython=True, cache=True)
def calculate_drawdowns(total_values):
    """Calculate drawdowns using numba for speed"""
    n = len(total_values)
    cummax = np.zeros(n, dtype=np.float64)
    drawdown = np.zeros(n, dtype=np.float64)
    
    # Calculate cumulative maximum
    cummax[0] = total_values[0]
    for i in range(1, n):
        cummax[i] = max(cummax[i-1], total_values[i])
    
    # Calculate drawdowns
    drawdown = (total_values - cummax) / cummax
    max_dd = abs(np.min(drawdown))
    
    return drawdown, max_dd

@dataclass
class Portfolio:
    data_handler: object
    events: Queue
    start_date: datetime
    initial_capital: float = 100000.0
    current_positions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    current_holdings: Dict[str, float] = field(default_factory=lambda: {
        "cash": 100000.0,
        "commission": 0.0,
        "total": 100000.0
    })
    all_positions: List[Dict[str, int]] = field(default_factory=list)
    all_holdings: List[Dict[str, Union[datetime, float]]] = field(default_factory=list)
    equity_curve: Optional[pd.DataFrame] = None
    _position_cache: Dict[str, int] = field(default_factory=dict)
    _market_value_cache: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.current_holdings["cash"] = self.initial_capital
        self.current_holdings["total"] = self.initial_capital
        self._position_cache = defaultdict(int)
        self._market_value_cache = {}

    def update_timeindex(self, latest_datetime: datetime, market_data: Dict[str, float]) -> None:
        """Update portfolio holdings with latest market data"""
        try:
            # Update position cache
            for symbol in market_data:
                self._position_cache[symbol] = self.current_positions.get(symbol, 0)
            
            # Vectorized market value calculation
            positions_array = np.array([self._position_cache[symbol] for symbol in market_data.keys()], dtype=np.float64)
            prices_array = np.array(list(market_data.values()), dtype=np.float64)
            market_value = np.sum(positions_array * prices_array)
            
            # Update holdings
            new_holdings = {
                "datetime": latest_datetime,
                "cash": self.current_holdings["cash"],
                "commission": self.current_holdings["commission"],
                "total": self.current_holdings["cash"] + market_value
            }
            
            # Append to historical data
            self.all_positions.append(dict(self._position_cache))
            self.all_holdings.append(new_holdings)
            
            # Update current holdings
            self.current_holdings["total"] = new_holdings["total"]
            
        except Exception as e:
            print(f"Error in update_timeindex: {e}")

    def generate_order(self, signal) -> Optional[OrderEvent]:
        """Generate order from signal"""
        try:
            order = None
            symbol = signal.symbol
            direction = signal.signal_type
            quantity = int(100)  # Fixed quantity for simplicity

            if direction == 'LONG':
                order = OrderEvent(symbol, 'BUY', quantity, 'BUY')
            elif direction == 'EXIT':
                order = OrderEvent(symbol, 'SELL', quantity, 'SELL')

            return order
        except Exception as e:
            print(f"Error generating order: {e}")
            return None

    def update_fill(self, fill: FillEvent) -> None:
        """Update portfolio with fill information"""
        try:
            # Update positions
            if fill.direction == 'BUY':
                self.current_positions[fill.symbol] += fill.quantity
            else:  # SELL
                self.current_positions[fill.symbol] -= fill.quantity

            # Update cash and commission
            fill_cost = fill.fill_cost * fill.quantity
            self.current_holdings["cash"] -= fill_cost
            self.current_holdings["commission"] += fill.commission
            self.current_holdings["total"] -= fill.commission
            
            # Update caches
            self._position_cache[fill.symbol] = self.current_positions[fill.symbol]
            
        except Exception as e:
            print(f"Error updating fill: {e}")

    def create_equity_curve(self) -> pd.DataFrame:
        """Create equity curve DataFrame from holdings"""
        try:
            if not self.all_holdings:
                print("No holdings data available")
                return pd.DataFrame()
                
            df = pd.DataFrame(self.all_holdings)
            
            if 'datetime' not in df.columns:
                print("Error: 'datetime' column not found in holdings data")
                return pd.DataFrame()
                
            df.set_index("datetime", inplace=True)
            
            # Use numba-accelerated functions for calculations
            total_values = df["total"].values
            returns = calculate_returns(total_values)
            equity_curve = calculate_equity_curve(returns)
            
            df["returns"] = returns
            df["equity_curve"] = equity_curve
            
            return df
        except Exception as e:
            print(f"Error creating equity curve: {str(e)}")
            return pd.DataFrame()

    def calculate_sharpe_ratio(self, returns: np.ndarray, periods: int = 252, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio using vectorized operations"""
        try:
            if len(returns) < 2:
                return 0.0
                
            # Remove NaN values using boolean indexing
            returns = returns[~np.isnan(returns)]
            if len(returns) < 2:
                return 0.0
                
            # Vectorized calculations
            daily_rf = (1 + risk_free_rate) ** (1/252) - 1
            excess_returns = returns - daily_rf
            return_mean = np.mean(excess_returns)
            return_std = np.std(returns)
            
            if return_std == 0:
                return 0.0
                
            sharpe = np.sqrt(periods) * return_mean / return_std
            return sharpe if not np.isnan(sharpe) else 0.0
            
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_drawdowns(self, equity_curve: pd.Series) -> tuple[pd.Series, float, int]:
        """Calculate drawdowns"""
        try:
            hwm = [0]
            drawdown = [0]
            duration = [0]

            for t in range(1, len(equity_curve)):
                hwm.append(max(hwm[t-1], equity_curve.iloc[t]))
                drawdown.append((hwm[t] - equity_curve.iloc[t]))
                if drawdown[t] == 0:
                    duration.append(0)
                else:
                    duration.append(duration[t-1] + 1)

            return pd.Series(drawdown, index=equity_curve.index), max(drawdown), max(duration)
        except Exception as e:
            print(f"Error calculating drawdowns: {e}")
            return pd.Series(), 0.0, 0

    def output_summary_stats(self) -> None:
        """Output performance metrics"""
        try:
            print("\nCalculating performance metrics...")
            
            if not self.all_holdings:
                print("No holdings data available")
                return
                
            # Convert holdings to DataFrame
            df = pd.DataFrame(self.all_holdings)
            if 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
            
            # Use numba-accelerated functions for calculations
            total_values = df["total"].values
            returns = calculate_returns(total_values)
            equity_curve = calculate_equity_curve(returns)
            
            df["returns"] = returns
            df["equity_curve"] = equity_curve
            
            # Vectorized calculations for metrics
            total_return = (total_values[-1] / total_values[0]) - 1.0
            if len(df) > 0 and total_return > -1:
                annualized_return = (1.0 + total_return) ** (252.0 / len(df)) - 1.0
            else:
                annualized_return = total_return
                
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
            
            # Calculate drawdowns using numba
            drawdown, max_dd = calculate_drawdowns(total_values)
            
            # Print performance summary
            print("\nPerformance Summary:")
            print("=" * 50)
            print(f"Total Return         : {total_return*100:.2f}%")
            print(f"Annualized Return    : {annualized_return*100:.2f}%")
            print(f"Volatility           : {volatility*100:.2f}%")
            print(f"Sharpe Ratio         : {sharpe_ratio:.2f}")
            print(f"Max Drawdown         : {max_dd*100:.2f}%")
            print(f"Final Portfolio Value: ${total_values[-1]:,.2f}")
            print("=" * 50)
            
            # Plot equity curve
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                
                # Convert index to datetime properly
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index)
                    except:
                        df.index = range(len(df))
                
                plt.plot(df.index, df["equity_curve"], label="Equity Curve")
                plt.title("Portfolio Equity Curve")
                plt.xlabel("Date" if isinstance(df.index, pd.DatetimeIndex) else "Trading Days")
                plt.ylabel("Portfolio Value")
                plt.grid(True)
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('equity_curve.png')
                plt.close()
                print("\nEquity curve saved as 'equity_curve.png'")
            except Exception as e:
                print(f"Error plotting equity curve: {e}")
                print(f"DataFrame index type: {type(df.index)}")
                print(f"DataFrame index: {df.index}")
            
        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")
            if 'df' in locals():
                print(f"DataFrame info: {df.info()}")
                print(f"DataFrame head:\n{df.head()}")
                print(f"DataFrame shape: {df.shape}")
                print(f"DataFrame columns: {df.columns.tolist()}")

    def _calculate_max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive True values in a boolean series"""
        try:
            return (series.astype(int).groupby((~series).astype(int).cumsum()).cumsum().max())
        except Exception as e:
            print(f"Error calculating consecutive values: {e}")
            return 0

    def plot_equity_curve(self, save_path='equity_curve.png'):
        """Plot and save equity curve independently"""
        try:
            if not self.all_holdings:
                print("No holdings data available")
                return
                
            # Convert holdings to DataFrame
            df = pd.DataFrame(self.all_holdings)
            if 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
            
            # Use numba-accelerated functions
            total_values = df["total"].values
            returns = calculate_returns(total_values)
            equity_curve = calculate_equity_curve(returns)
            
            df["returns"] = returns
            df["equity_curve"] = equity_curve
            
            # Plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    df.index = range(len(df))
            
            plt.plot(df.index, df["equity_curve"], label="Equity Curve")
            plt.title("Portfolio Equity Curve")
            plt.xlabel("Date" if isinstance(df.index, pd.DatetimeIndex) else "Trading Days")
            plt.ylabel("Portfolio Value")
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"\nEquity curve saved as '{save_path}'")
            
        except Exception as e:
            print(f"Error plotting equity curve: {e}")

if __name__ == "__main__":
    # Example usage:
    from data_handler import HistoricCSVDataHandler
    from events import Queue
    from datetime import datetime
    
    # Initialize components
    events = Queue()
    data_handler = HistoricCSVDataHandler(events, ".", ["MSFT"])
    portfolio = Portfolio(data_handler, events, datetime(2010, 1, 1))
    
    # Run backtest
    while data_handler.continue_backtest:
        data_handler.update_bars()
        # Process events if needed
    
    # Plot equity curve
    portfolio.plot_equity_curve()