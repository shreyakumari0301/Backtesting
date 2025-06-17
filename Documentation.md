# 📘 **Event-Driven Backtesting Engine**

## 🧩 **Overview**

This project implements a high-performance, event-driven backtesting engine for algorithmic trading. Initially developed using **PyTorch** for prototype accuracy and GPU support, the project was later migrated to **NumPy** to improve portability and flexibility. Finally, **Numba** was integrated to optimize computational efficiency, bringing execution time for large datasets down from minutes to seconds.

The framework replicates the structure of real-world trading systems, handling asynchronous events such as market updates, trading signals, order placements, and order fills. It is **modular and extensible**, making it easy to test various trading strategies in a controlled environment.

---

## 🎯 **Project Motivation and Background**

Backtesting is essential in quantitative finance to validate the viability of trading strategies before deployment. 

This engine was designed to:

* Be **minimal yet realistic**
* Allow full control over **data batching and signal generation**
* Support **rapid prototyping** and speed up research iterations

---

## 🛠️ **Development Summary**

* Built the initial prototype using **PyTorch** for signal modeling
* Converted the core computation logic to **NumPy** for better compatibility
* Optimized key components with **Numba JIT**
* Implemented the **event-driven architecture**
* Developed core modules: `DataHandler`, `Strategy`, `Portfolio`, `ExecutionHandler`
* Finalized testing, profiling, and GitHub deployment

---

## 🧱 **Core Components & Architecture**

### 📦 `Event`

* Base class for all event types (using `@dataclass`)
* Subclasses: `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`

### 📊 `DataHandler`

* Reads and batches historical data
* Feeds `MarketEvent` at each step
* Uses `np.ascontiguousarray` for cache optimization

### 🧠 `Strategy`

* Listens to `MarketEvent`
* Emits `SignalEvent` based on trading logic

### 💼 `Portfolio`

* Listens to `SignalEvent`
* Emits `OrderEvent`
* Updates holdings upon `FillEvent`

### 🏦 `ExecutionHandler`

* Listens to `OrderEvent`
* Simulates trade fill and emits `FillEvent`

---

## ⚙️ **Optimization Details**

### 🔄 From:

* PyTorch with high memory usage and CPU bottlenecks

### 🚀 To:

* Efficient NumPy array operations
* `np.ascontiguousarray` for memory layout optimization
* Numba-accelerated loops for real-time simulation

### 📈 Results:

* Speedup: **\~15x** for 1M+ data points
* Memory usage: **60% reduction**
* Execution time: **< 10 seconds** for 1M points

---

## 🔁 **Flow of Execution**

### ▶️ Entry Point:

```bash
python backtest.py
```

### 🛎️ Initialization:

* Load event queue
* Instantiate `DataHandler`, `Portfolio`, `Strategy`, `ExecutionHandler`

### 🔄 Main Loop:

```python
while data_handler.continue_backtest:
    data_handler.update_bars()

    while not events.empty():
        event = events.get()
        if event.type == 'MARKET':
            strategy.calculate_signals()
        elif event.type == 'SIGNAL':
            portfolio.generate_order()
        elif event.type == 'ORDER':
            execution_handler.execute_order()
        elif event.type == 'FILL':
            portfolio.update_from_fill()
```

### 🔗 Component Interaction Flow:

```
MarketEvent → SignalEvent → OrderEvent → FillEvent → Portfolio Update
```

---

## ✅ **Summary**

This modular, efficient engine can be scaled to test multiple strategies over massive datasets, supporting realistic simulation through its strict event-based processing pipeline.

---
