from threading import Lock
from typing import Optional


class LockMixin:
    _lock: Optional[Lock] = None

    @property
    def lock(self) -> Lock:
        """Get the lock."""
        if self._lock is None:
            self._lock = Lock()
        return self._lock

    def acquire_lock(self) -> None:
        """Acquire the lock."""
        self.lock.acquire()

    def release_lock(self) -> None:
        """Release the lock."""
        self.lock.release()
