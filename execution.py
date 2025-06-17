from abc import ABCMeta, abstractmethod
from events import FillEvent

class ExecutionHandler(metaclass=ABCMeta):
    @abstractmethod
    def execute_order(self, event):
        raise NotImplementedError("Should implement execute_order()")

class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, events, data_handler):
        self.events = events
        self.data_handler = data_handler

    def execute_order(self, event):
        if event.type == 'ORDER':
            fill_event = FillEvent(
                timeindex=self.data_handler.get_latest_bar_datetime(event.symbol),
                symbol=event.symbol,
                exchange='ARCA',
                quantity=event.quantity,
                direction=event.direction,
                fill_cost=self.data_handler.get_latest_bar_value(event.symbol, "Close")
            )
            self.events.put(fill_event)
            # print(f"Order executed: {event.order_type} {event.quantity} {event.symbol} at {fill_event.fill_cost}")

    def _create_fill_from_order(self, order):
        symbol = order.symbol
        quantity = order.quantity
        direction = order.direction
        price = self.data_handler.get_latest_bar_value(symbol, "Close")
        commission = self.calculate_commission(quantity, price)
        
        return FillEvent(
            self.data_handler.get_latest_bar_datetime(symbol),
            symbol,
            "ARCA",
            quantity,
            direction,
            price,
            commission
        )

    def calculate_commission(self, quantity, price):
        return max(1.3, 0.013 * quantity) 