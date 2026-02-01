"""
HTTP Request Logging Middleware

This module defines a Starlette `BaseHTTPMiddleware` that logs inbound HTTP
requests and their outcomes. It enriches log records with request context
including request ID, method, path, client IP, user agent, status, and
duration. It uses Loguru bindings to attach structured context to each log.
"""
from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


def _client_ip(request: Request) -> str:
    """
    Resolve the client IP address from headers or the connection scope.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        str: The best-effort client IP string, possibly empty.
    """
    xfwd = request.headers.get("x-forwarded-for")
    if xfwd:
        return xfwd.split(",")[0].strip()
    return request.client.host if request.client else ""


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs request start, completion, and unhandled exceptions."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process a request, logging start, completion with status and duration, and exceptions.

        Args:
            request (Request): The incoming FastAPI/Starlette request.
            call_next (Callable): The next handler in the ASGI chain.

        Returns:
            Response: The response returned by the downstream application.

        Raises:
            Exception: Re-raises any unhandled exceptions after logging.
        """
        rid = str(uuid.uuid4())
        method = request.method
        path = request.url.path
        client_ip = _client_ip(request)
        user_agent = request.headers.get("user-agent", "")

        # Bind request context
        ctx_logger = logger.bind(request_id=rid, method=method, path=path, client_ip=client_ip, user_agent=user_agent)

        request.state.request_id = rid

        start = time.perf_counter()
        ctx_logger.info("HTTP request started")

        # To avoid blocking the stream, we read the body here and replace the stream.
        body = await request.body()
        
        async def receive():
            return {"type": "http.request", "body": body}

        request = Request(request.scope, receive)
        
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000.0
            ctx_logger.bind(status=response.status_code, duration_ms=round(duration_ms, 2)).info("HTTP request completed")
            return response
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000.0
            # Log exception with traceback
            ctx_logger.bind(duration_ms=round(duration_ms, 2)).exception("Unhandled exception while processing request")
            raise


