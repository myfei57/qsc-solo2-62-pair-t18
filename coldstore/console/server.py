"""WSGI application that serves the control console."""

from __future__ import annotations

from typing import Callable, Iterable

from coldstore.store import Store

from . import handlers
from .http import Request, RequestError, Response, Router, build_request, error_response
from .metrics import Metrics
from .routes import register_routes
from .runtime import Runtime
from .snapshot import build_coordinator

# POST routes that only read state; everything else non-GET is a mutation
# and must pass the snapshot freeze gate and advance the state version.
READ_ONLY_POSTS = frozenset(
    {
        "/room/lookup",
        "/comp/lookup",
        "/fan/lookup",
        "/audit/filter",
        "/quota/check",
        "/quota/remaining",
        "/load/estimate",
        "/schedule/plan",
        "/defrost/select",
    }
)


def _is_mutation(request: Request) -> bool:
    return request.method != "GET" and request.path not in READ_ONLY_POSTS


class Server:
    """Owns the runtime state and dispatches console requests."""

    def __init__(self, store: Store) -> None:
        self.runtime = Runtime(store)
        self.router = Router()
        self.metrics = Metrics()
        self.snapshots = build_coordinator(self.runtime)
        register_routes(self.router, handlers)

    def respond(self, request: Request) -> Response:
        response = self._route(request)
        self.metrics.record(request.method, request.path, response.status)
        return response

    def _route(self, request: Request) -> Response:
        handler = self.router.match(request.method, request.path)
        if handler is None:
            if self.router.allows(request.path):
                return error_response(
                    405, f"method {request.method} not allowed for {request.path}"
                )
            return error_response(404, f"no route for {request.path}")
        if _is_mutation(request):
            with self.snapshots.mutation():
                response = self._invoke(handler, request)
                # 4xx means the request was rejected before committing;
                # anything else may have changed state, so bump the version.
                if response.status < 400 or response.status >= 500:
                    self.snapshots.note_mutation()
            return response
        return self._invoke(handler, request)

    def _invoke(self, handler, request: Request) -> Response:
        try:
            return handler(self, request)
        except RequestError as exc:
            return error_response(exc.status, exc.message)
        except ValueError as exc:
            return error_response(400, str(exc))
        except Exception as exc:  # noqa: BLE001 - last resort for a live console
            return error_response(500, str(exc))

    def dispatch(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> Response:
        """Call a route directly, used by the test suite and CLI probes."""

        request = Request(method=method.upper(), path=path, query={}, payload=dict(payload or {}))
        return self.respond(request)

    def __call__(
        self, environ: dict[str, object], start_response: Callable[..., object]
    ) -> Iterable[bytes]:
        try:
            response = self.respond(build_request(environ))
        except RequestError as exc:
            response = error_response(exc.status, exc.message)
        status, headers = response.wsgi()
        start_response(status, headers)
        return [response.body]


def create_application(store: Store) -> Server:
    """Build a ready to serve console for the supplied store."""

    return Server(store)
