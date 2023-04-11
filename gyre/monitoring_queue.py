import threading
from queue import Queue


class MonitoringQueue(Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._waiting = {}

    def get(self, block=True, timeout=None):
        thread_id = threading.get_ident()
        self._waiting[thread_id] = True

        try:
            return super().get(block, timeout)
        finally:
            # Not re-entrant. Probably fine, super().get can't call code
            del self._waiting[thread_id]

    def wait_count(self):
        return len(self._waiting)
