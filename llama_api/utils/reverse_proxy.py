from os import environ
from typing import List, Mapping, Optional, Set, Tuple, Union

from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

RawHeaderKeys = Union[List[bytes], Tuple[bytes, ...], Set[bytes]]


class ReverseProxy:
    """Reverse proxy for the OpenAI API.
    Set `OPENAI_API_KEY` in the environment to set the Authorization header,
    or the client will have to set the Authorization header manually."""

    def __init__(self, *args, **kwargs):
        self._client = None
        self._URL = None
        self._args = args
        self._kwargs = kwargs

    @property
    def client(self):
        if not self._client:
            self._install_httpx_if_needed()
            import httpx

            self._client = httpx.AsyncClient(*self._args, **self._kwargs)
        return self._client

    @staticmethod
    def _install_httpx_if_needed():
        try:
            import httpx  # noqa: F401
        except ImportError:
            from .dependency import install_package

            install_package("httpx")

    def _get_url(self, base_url: str, path: str, query: Optional[str]):
        if not self._URL:
            self._install_httpx_if_needed()
            import httpx

            self._URL = httpx.URL
        return self._URL(
            base_url, path=path, query=(query or "").encode("utf-8")
        )

    async def get_reverse_proxy_response(
        self,
        request: Request,
        base_url: str,
        excluded_headers: Optional[RawHeaderKeys] = None,
        included_headers: Optional[RawHeaderKeys] = None,
        additional_headers: Optional[Mapping[bytes, bytes]] = None,
    ) -> StreamingResponse:
        """Get the response from the reverse proxy.
        This function is used to proxy the OpenAI API.
        The excluded_headers and included_headers are used to
        filter the headers from the request to the reverse proxy.
        The additional_headers are added to the request to the reverse proxy.
        """
        headers = {
            name: value
            for name, value in request.headers.raw
            if name not in (excluded_headers or ())
            and (included_headers is None or name in included_headers)
        }
        if additional_headers:
            headers.update(additional_headers)
        rp_req = self.client.build_request(
            request.method,
            self._get_url(
                base_url, path=request.url.path, query=request.url.query
            ),
            headers=self.client._merge_headers(headers),
            content=request.stream(),
        )
        rp_resp = await self.client.send(rp_req, stream=True)
        return StreamingResponse(
            rp_resp.aiter_raw(),
            status_code=rp_resp.status_code,
            headers=rp_resp.headers,
            background=BackgroundTask(rp_resp.aclose),
        )


def get_openai_authorization_header() -> Optional[Mapping[bytes, bytes]]:
    """Get the OpenAI API key from the environment or CLI arguments.
    Return None if the API key is not set.
    This function is used to set the Authorization header
    for the reverse proxy."""
    openai_api_key = environ.get("OPENAI_API_KEY")
    print(f"OpenAI API key: {openai_api_key}")
    return (
        {b"Authorization": f"Bearer {openai_api_key}".encode("utf-8")}
        if openai_api_key
        else None
    )


if __name__ == "__main__":
    from functools import partial

    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()
    rp = ReverseProxy(headers=get_openai_authorization_header())
    app.post("/v1/chat/completions")(
        partial(
            rp.get_reverse_proxy_response,
            base_url="https://api.openai.com",
            excluded_headers=(b"host", b"content-length"),
        )
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
