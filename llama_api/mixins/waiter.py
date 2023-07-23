from threading import Event
from typing import Optional


class WaiterMixin:
    _is_available: Optional[Event] = None

    @property
    def is_available(self) -> bool:
        """Check if the model is available."""
        if self._is_available is None:
            self._is_available = Event()
            self._is_available.set()
        return self._is_available.is_set()

    def wait_until_available(self) -> None:
        """Wait until the model is available."""
        if self._is_available is None:
            self._is_available = Event()
            self._is_available.set()
        self._is_available.wait()

    def set_availability(self, availablity: bool) -> None:
        """Set the model availability."""
        if self._is_available is None:
            self._is_available = Event()
        if availablity:
            self._is_available.set()
        else:
            self._is_available.clear()
