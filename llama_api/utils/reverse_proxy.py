from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask


class ReverseProxy:
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
        self, request: Request, base_url: str
    ) -> StreamingResponse:
        rp_req = self.client.build_request(
            request.method,
            self._get_url(
                base_url, path=request.url.path, query=request.url.query
            ),
            headers=[
                (name, value)
                for name, value in request.headers.raw
                if name not in (b"host", b"content-length")
            ],
            content=request.stream(),
        )
        rp_resp = await self.client.send(rp_req, stream=True)
        return StreamingResponse(
            rp_resp.aiter_raw(),
            status_code=rp_resp.status_code,
            headers=rp_resp.headers,
            background=BackgroundTask(rp_resp.aclose),
        )


if __name__ == "__main__":
    from functools import partial

    import uvicorn

    app = FastAPI()
    rp = ReverseProxy()
    app.post("/v1/chat/completions")(
        partial(
            rp.get_reverse_proxy_response, base_url="https://api.openai.com"
        )
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
