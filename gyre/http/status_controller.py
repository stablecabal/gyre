import chevron

from gyre.http.json_api_controller import (
    JSONAPIController,
    UnsupportedMediaTypeResource,
)

template = """
<html>
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/milligram/1.4.1/milligram.css">
        <style type="text/css">
            body { padding: 20px; }
            h1 { line-height: 64px; }
            h1 img { height: 64px; vertical-align: bottom; margin-right: 8px; }
            .log { max-height: 200px; overflow: auto; background: #EEE; font-family: mono; }
            .log > div { white-space: pre; }
        </style>
    </head>
    <body>
        <h1><img src='https://gyre.ai/img/gyrelogo-256.png' alt='Gyre.ai logo'/>Gyre server status</h1>
        <h4>Status overview: {{manager_status}}</h4>
        <h4>Log:</h4>
        <div class="log">
            {{#log}}
                <div>{{.}}</div>
            {{/log}}
        </div>
    </body>
</html>
"""


class StatusController(JSONAPIController):
    preferred_return_type = "text/html"
    return_types = {"application/json", "text/html"}

    def __init__(self):
        super().__init__()
        self._manager = None

    def set_manager(self, manager):
        self._manager = manager

    def handle_GET(self, request, _):
        data = {
            "manager_status": self._manager.status if self._manager else "pending",
            "log": self._manager.log if self._manager else "",
        }

        accept_header = request.getHeader("accept")
        if accept_header == "text/html":
            return chevron.render(template, data)

        else:
            return data
