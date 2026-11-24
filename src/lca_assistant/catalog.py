"""Load the illustrative resource catalogue."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Resource


def load_resources(path: str | Path) -> list[Resource]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Resource.model_validate(item) for item in payload]
