from twisted.internet import ssl
from twisted.protocols.tls import TLSMemoryBIOFactory
from twisted.web.proxy import ProxyClient, ReverseProxyResource
from urllib.parse import quote as urlquote


class StripHeadersProxyClient(ProxyClient):
    stripped_headers = {b"cookie", b"set-cookie"}

    def __init__(self, command, rest, version, headers, data, father):
        for stripped_header in self.stripped_headers:
            if stripped_header in headers:
                del headers[stripped_header]

        super().__init__(command, rest, version, headers, data, father)

    def handleHeader(self, key, value):
        if key.lower() not in self.stripped_headers:
            super().handleHeader(key, value)


class HTTPReverseProxyResource(ReverseProxyResource, object):
    def getChild(self, path, request):
        """
        Ensure that implementation of C{proxyClientFactoryClass} is honored
        down the resource chain.
        """
        return HTTPReverseProxyResource(
            self.host,
            self.port,
            self.path + b"/" + urlquote(path, safe=b"@").encode("utf-8"),
            self.reactor,
        )


class HTTPSReverseProxyResource(ReverseProxyResource, object):
    def proxyClientFactoryClass(self, *args, **kwargs):
        """
        Make all connections using HTTPS.
        """
        proxyClientFactory = super().proxyClientFactoryClass(*args, **kwargs)
        proxyClientFactory.protocol = StripHeadersProxyClient

        return TLSMemoryBIOFactory(
            ssl.optionsForClientTLS(self.host), True, proxyClientFactory
        )

    def getChild(self, path, request):
        """
        Ensure that implementation of C{proxyClientFactoryClass} is honored
        down the resource chain.
        """
        return HTTPSReverseProxyResource(
            self.host,
            self.port,
            self.path + b"/" + urlquote(path, safe=b"@").encode("utf-8"),
            self.reactor,
        )

    def render(self, request):
        # Only allow GET
        if request.method != b"GET":
            request.setResponseCode(405)
            return b"Method Not Allowed"

        return super().render(request)
