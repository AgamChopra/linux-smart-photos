from __future__ import annotations

import json
from pathlib import Path

from .models import LibraryState


class JsonLibraryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> LibraryState:
        if not self.path.exists():
            return LibraryState()

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return LibraryState.from_dict(payload)

    def save(self, state: LibraryState) -> None:
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(state.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self.path)
