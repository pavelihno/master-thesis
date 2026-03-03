from river import base


class NoDriftDetector(base.DriftDetector):
    """A no-op drift detector that never signals drift."""

    def update(self, x: int | float) -> None:
        pass
