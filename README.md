# Stock Backtesting System

A modular backtesting system for stock trading strategies.

## Structure

- `events.py`: Event classes for handling trading signals, orders, and fills
- `data_handler.py`: Data handling and market data feed
- `strategy.py`: Trading strategy implementations
- `portfolio.py`: Portfolio management and position tracking
- `execution.py`: Order execution handling
- `backtest.py`: Main backtesting engine

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the backtest with default settings:
```bash
python backtest.py
```

2. To use a different stock, modify the `symbol` variable in `backtest.py`:
```python
symbol = "AAPL"  # Change to any stock symbol
```

## Features

- Modular design for easy extension
- Mean reversion strategy implementation
- Simulated order execution
- Portfolio tracking and performance metrics
- Automatic data download using yfinance

## Customization

- Add new strategies by extending the `Strategy` class in `strategy.py`
- Modify portfolio parameters in `portfolio.py`
- Adjust execution settings in `execution.py` 