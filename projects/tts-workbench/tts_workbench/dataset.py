"""Read processed dialogue manifests without modifying the source material."""

import base64
import hashlib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from tts_workbench.config import RunConfig


class DialogueLine(BaseModel):
    """Describe the subset of the curated dialogue schema used for requests."""

    line_id: str
    unit_id: str
    line_index: int
    speaker: str
    speaker_id: str
    text_zh: str
    text_en_tts: str
    audio_path: str
    audio_sha256: str
    reference_duration_ms: float
    reference_text_needs_review: bool = False


class DataUnit(BaseModel):
    """Locate a unit's complete, ordered line list."""

    unit_id: str
    title: str
    lines_path: str
    line_count: int


class DataManifest(BaseModel):
    """Read the portable dataset entry point."""

    dataset_id: str
    line_count: int
    units: list[DataUnit]


class DataCatalog(BaseModel):
    """Expose the portable data catalog without filtering audio or text content."""

    dataset_id: str
    fingerprint: str
    line_count: int
    review_count: int
    units: list[DataUnit]
    lines: list[DialogueLine]


class Dataset:
    """Index line IDs and constrain all reads to the configured dataset root."""

    def __init__(self, root: Path) -> None:
        """Validate counts and identities once before the workbench starts."""
        self.root = root.resolve()
        manifest_bytes = (self.root / "manifest.json").read_bytes()
        self.manifest = DataManifest.model_validate_json(manifest_bytes)
        self.fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
        self.lines: dict[str, DialogueLine] = {}
        for unit in self.manifest.units:
            rows = (
                self.resolve(unit.lines_path).read_text(encoding="utf-8").splitlines()
            )
            entries = [
                DialogueLine.model_validate_json(row) for row in rows if row.strip()
            ]
            if len(entries) != unit.line_count:
                raise ValueError(f"Line count mismatch for {unit.unit_id}")
            else:
                pass
            for line in sorted(entries, key=lambda entry: entry.line_index):
                if line.line_id in self.lines or line.unit_id != unit.unit_id:
                    raise ValueError(f"Invalid or duplicate line ID: {line.line_id}")
                else:
                    pass
                self.lines[line.line_id] = line
        if len(self.lines) != self.manifest.line_count:
            raise ValueError("Dataset total line count does not match its manifest")
        else:
            pass

    def resolve(self, relative_path: str) -> Path:
        """Prevent a manifest path or symlink from escaping the dataset root."""
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Dataset paths must stay inside the configured directory")
        else:
            pass
        return path

    def select(self, config: RunConfig) -> list[DialogueLine]:
        """Keep source order while applying unit, speaker, and line filters."""
        known_units = {unit.unit_id for unit in self.manifest.units}
        known_speakers = {line.speaker_id for line in self.lines.values()}
        if set(config.unit_ids) - known_units:
            raise ValueError("Unknown unit selected")
        elif set(config.speaker_ids) - known_speakers:
            raise ValueError("Unknown speaker selected")
        elif set(config.line_ids) - self.lines.keys():
            raise ValueError("Unknown dialogue line selected")
        else:
            pass
        selected = [
            line
            for line in self.lines.values()
            if (not config.unit_ids or line.unit_id in config.unit_ids)
            and (not config.speaker_ids or line.speaker_id in config.speaker_ids)
            and (not config.line_ids or line.line_id in config.line_ids)
        ]
        if not selected:
            raise ValueError("No dialogue matches the selected filters")
        elif config.mode == "single" and len(selected) != 1:
            raise ValueError("Single-request mode requires exactly one dialogue line")
        else:
            pass
        return selected

    @lru_cache(maxsize=512)
    def reference_uri(self, line_id: str) -> str:
        """Send original bytes without checksum, duration, or content filtering."""
        line = self.lines[line_id]
        audio_bytes = self.resolve(line.audio_path).read_bytes()
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        return f"data:audio/wav;base64,{encoded}"

    def public_manifest(self) -> DataCatalog:
        """Expose portable metadata, leaving source-machine provenance private."""
        return DataCatalog(
            dataset_id=self.manifest.dataset_id,
            fingerprint=self.fingerprint,
            line_count=len(self.lines),
            review_count=sum(
                line.reference_text_needs_review for line in self.lines.values()
            ),
            units=self.manifest.units,
            lines=list(self.lines.values()),
        )
