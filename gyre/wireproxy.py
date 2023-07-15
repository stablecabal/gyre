import logging
import os
import shutil
import tempfile
import time
from twisted.internet import protocol, reactor
from twisted.internet.error import ProcessDone

BASE_TEMPLATE = """
[Interface]
Address = {ip}/32
PrivateKey = {key}

[Peer]
PublicKey = {endpointkey}
EndPoint = {endpoint}
PersistentKeepalive = 10
"""

PORT_TEMPLATE = """
[TCPServerTunnel]
ListenPort = {external_port}
Target = localhost:{internal_port}
"""


class Wireproxy:
    class Protocol(protocol.ProcessProtocol):
        def __init__(self, cfgpath):
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

            self.cfgpath = cfgpath
            self.up = False

        def connectionMade(self):
            self.transport.closeStdin()

        def outReceived(self, data: bytes):
            # Try to remove the config file ASAP
            if self.cfgpath:
                os.unlink(self.cfgpath)
                self.cfgpath = None

            str = data.decode("utf-8")
            str = str.strip()

            for line in str.splitlines():
                if line.startswith("DEBUG: "):
                    self.logger.debug(line[len("DEBUG: ") :])
                else:
                    self.logger.info(line)

                if "Received handshake response" in line:
                    self.up = True
                    self.logger.info("Wireproxy connection made")

        def errReceived(self, err: bytes):
            str = err.decode("utf-8")
            str = str.strip()
            self.logger.error(str)

        def processExited(self, status):
            if isinstance(status.value, ProcessDone):
                self.logger.info("Wireproxy exited cleanly")
            else:
                self.logger.error("Wireproxy exited with errror")

    def __init__(
        self,
        ip,
        key,
        endpoint,
        endpointkey,
        ports={5000: 5000, 50051: 50051},
    ):
        self.ip = ip
        self.key = key
        self.endpointkey = endpointkey
        self.endpoint = endpoint
        self.ports = ports
        self.proc = None

    def _write_cfg(self):
        cfgfd, cfgpath = tempfile.mkstemp(".cfg", text=True)

        with os.fdopen(cfgfd, "w") as cfg:
            cfg.write(
                BASE_TEMPLATE.format(
                    ip=self.ip,
                    key=self.key,
                    endpointkey=self.endpointkey,
                    endpoint=self.endpoint,
                )
            )
            for external_port, internal_port in self.ports.items():
                cfg.write(
                    PORT_TEMPLATE.format(
                        external_port=external_port,
                        internal_port=internal_port,
                    )
                )

        return cfgpath

    def start(self):
        wireproxy_path = shutil.which("wireproxy")

        if not wireproxy_path:
            raise NotImplementedError(
                "You need an wireproxy in your path to use it. Download from https://github.com/pufferffish/wireproxy/releases."
            )

        cfgpath = self._write_cfg()
        print(cfgpath)

        self.proc = reactor.spawnProcess(
            Wireproxy.Protocol(cfgpath),
            executable=wireproxy_path,
            args=["wireproxy", "-c", cfgpath],
            env=os.environ,
        )

    def stop(self, grace=10):
        if not self.proc:
            return

        self.proc.signalProcess("TERM")

        for _ in range(grace):
            if not self.proc.pid:
                return
            time.sleep(1)

        print("Hard killing wireproxy")
        self.proc.signalProcess("KILL")
