import json
import os

from gyre.http.json_api_controller import JSONAPIController
from gyre.logging import store_handler

with open(
    os.path.join(
        os.path.dirname(__file__),
        "status_controller.html",
    ),
    "r",
) as f:
    html_source = f.read()


class StatusController(JSONAPIController):
    preferred_return_type = "text/html"
    return_types = {"application/json", "text/html"}

    def __init__(self):
        super().__init__()
        self._manager = None

    def set_manager(self, manager):
        self._manager = manager

    def encode_text_html(self, request, data):
        request.setHeader("content-type", "text/html")
        return html_source.replace("$DATA$", json.dumps(data))

    def handle_GET(self, request, _):
        return {
            "status": self._manager.status if self._manager else "pending",
            "logs": store_handler.logs,
        }
