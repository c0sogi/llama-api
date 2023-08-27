from threading import Event
from typing import Optional

from ..mixins.completion import CompletionStatus


class InterruptMixin:
    """A mixin class for interrupting(aborting) a job."""

    _interrupt_signal: Optional[Event] = None

    @property
    def is_interrupted(self) -> bool:
        """Check whether the job is interrupted or not."""
        return (
            self.interrupt_signal is not None
            and self.interrupt_signal.is_set()
        )

    @property
    def raise_for_interruption(self) -> None:
        """Raise an InterruptedError if the job is interrupted."""
        if self.is_interrupted:
            raise InterruptedError

    @property
    def interrupt_signal(self) -> Optional[Event]:
        """Get the interrupt signal."""
        return self._interrupt_signal

    @interrupt_signal.setter
    def interrupt_signal(self, value: Optional[Event]) -> None:
        """Set the interrupt signal."""
        self._interrupt_signal = value

    def check_interruption(self, status: CompletionStatus) -> bool:
        """Check whether the job is interrupted or not.
        If the job is interrupted, set the status to "interrupted"
        and return True. Otherwise, return False."""
        if self.is_interrupted:
            status.state = "interrupted"
            return True
        return False
