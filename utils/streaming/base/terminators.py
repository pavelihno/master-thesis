from abc import ABC, abstractmethod
from typing import Any

from pybeamline.bevent import BEvent


class TraceTerminator(ABC):
    """Determines when a trace has reached its terminal state."""

    @abstractmethod
    def is_terminal(self, event: BEvent) -> bool:
        """Return True if the event marks the end of the trace."""
        pass


class EventNameTerminator(TraceTerminator):
    """Trace ends when specific events occur."""

    def __init__(self, end_events: set[str] | None = None):
        self.end_events = end_events or set()

    def is_terminal(self, event: BEvent) -> bool:
        return event.get_event_name() in self.end_events


class AttributeValueTerminator(TraceTerminator):
    """Trace ends when a specific attribute has a certain value."""

    def __init__(self, attribute_name: str, terminal_values: set[Any]):
        self.attribute_name = attribute_name
        self.terminal_values = terminal_values

    def is_terminal(self, event: BEvent) -> bool:
        attributes = event.event_attributes

        attribute_value = attributes.get(self.attribute_name, None)

        return attribute_value in self.terminal_values


class OracleTerminator(AttributeValueTerminator):
    """Trace ends when an attribute 'is_terminal' is set to True."""

    def __init__(self):
        super().__init__(attribute_name='is_terminal', terminal_values={True})
