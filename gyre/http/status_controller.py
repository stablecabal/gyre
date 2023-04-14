import json
import os
import threading

from gyre.constants import IS_DEV
from gyre.http.json_api_controller import JSONAPIController
from gyre.logging import store_handler, tqdm_in_progress
from gyre.manager import EngineManager

if IS_DEV:
    cache = lambda f: f

else:
    from functools import cache


@cache
def html_template():
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "status_controller.html",
        ),
        "r",
    ) as f:
        return f.read()


class StatusController(JSONAPIController):
    preferred_return_type = "text/html"
    return_types = {"application/json", "text/html"}

    def __init__(self):
        super().__init__()
        self._manager: EngineManager | None = None

    def set_manager(self, manager):
        self._manager = manager

    def encode_text_html(self, request, data):
        request.setHeader("content-type", "text/html")
        return html_template().replace("$DATA$", json.dumps(data))

    def handle_GET(self, request, _):
        progress_bars = [bar.format_dict for bar in tqdm_in_progress]

        pipeline_bars = {
            bar.get("pipeline_unique_id"): bar
            for bar in progress_bars
            if bar.get("pipeline_unique_id")
        }

        regular_bars = [
            bar for bar in progress_bars if not bar.get("pipeline_unique_id")
        ]

        if self._manager is None:
            status = "Pending"
            slots = []
            queue_depth = 0

        else:
            status = self._manager.status
            slots = [
                {"device": str(slot.device), "pipeline": None, "progress": None}
                | (
                    {
                        "pipeline": slot.pipeline.id,
                        "progress": pipeline_bars.get(id(slot.pipeline)),
                    }
                    if slot.pipeline is not None
                    else {}
                )
                for slot in self._manager._device_slots
            ]
            queue_depth = self._manager._device_queue.wait_count()

        return {
            "status": status,
            "slots": slots,
            "queue_depth": queue_depth,
            "active_threads": threading.active_count(),
            "logs": store_handler.logs,
            "progress": regular_bars,
        }
