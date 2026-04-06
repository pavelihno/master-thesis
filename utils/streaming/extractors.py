from abc import ABC, abstractmethod

from pybeamline.bevent import BEvent


class OutcomeExtractor(ABC):
    @abstractmethod
    def extract(self, trace_id: str, event: BEvent) -> int | None:
        pass


class BinaryOutcomeExtractor(OutcomeExtractor):
    def __init__(self, positive_outcomes: set[str], negative_outcomes: set[str]):
        super().__init__()

        self.positive = positive_outcomes
        self.negative = negative_outcomes

    def extract(self, trace_id: str, event: BEvent) -> int | None:
        event_name = event.get_event_name()

        if event_name in self.positive:
            return 1
        if event_name in self.negative:
            return 0
        return None


class MultiClassOutcomeExtractor(OutcomeExtractor):
    def __init__(self, outcome_mapping: dict[str, int]):
        super().__init__()

        self.outcome_mapping = outcome_mapping

    def extract(self, trace_id: str, event: BEvent) -> int | None:
        event_name = event.get_event_name()
        return self.outcome_mapping.get(event_name, None)
