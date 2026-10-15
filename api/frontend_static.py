from __future__ import annotations

import re

from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


_CLIENT_ROUTE = re.compile(r"^(?:library|notes|reader/session/[A-Za-z0-9_-]+)$")


class FrontendStaticFiles(StaticFiles):
    """Serve the React entry for known browser routes, preserving other 404s."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as error:
            if error.status_code != 404 or not _CLIENT_ROUTE.fullmatch(path.replace("\\", "/")):
                raise
            return await super().get_response("index.html", scope)
