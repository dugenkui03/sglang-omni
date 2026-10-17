"""Validate deployment paths and the public test-task contract."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass(frozen=True)
class Settings:
    """Hold deployment configuration without depending on the model project."""

    api_base: str
    data_directory: Path
    results_directory: Path
    log_file: Path | None = None
    password: str | None = None
    api_key: str | None = None
    audio_quota_bytes: int = 1024**3

    def validate(self) -> None:
        """Reject ambiguous endpoints and result paths that overlap input data."""
        endpoint = urlsplit(self.api_base)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            raise ValueError("The TTS API base must be an absolute HTTP(S) URL")
        elif (
            endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError(
                "The TTS URL must not contain credentials, a query, or a fragment"
            )
        else:
            pass
        data_directory = self.data_directory.resolve()
        results_directory = self.results_directory.resolve()
        if not (data_directory / "manifest.json").is_file():
            raise ValueError("The dataset directory must contain manifest.json")
        elif (
            results_directory == data_directory
            or data_directory in results_directory.parents
        ):
            raise ValueError("Results must be stored outside the read-only dataset")
        elif self.audio_quota_bytes <= 0:
            raise ValueError("The audio quota must be positive")
        else:
            pass


class RunConfig(BaseModel):
    """Define one immutable test workload; omitted filters select all lines."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(default="", max_length=100)
    model: str = Field(default="qwen3-tts-1.7b-base", min_length=1, max_length=300)
    mode: Literal["duration", "once", "single"] = "duration"
    sample_order: Literal["sequential", "random"] = "sequential"
    repeat_probability: float = Field(default=0.2, ge=0, le=1)
    language: Literal["Chinese", "English"] = "English"
    unit_ids: list[str] = Field(default_factory=list, max_length=100)
    speaker_ids: list[str] = Field(default_factory=list, max_length=1000)
    line_ids: list[str] = Field(default_factory=list, max_length=10000)
    requests_per_minute: float = Field(default=10, gt=0, le=60000)
    duration_seconds: float = Field(default=600, gt=0, le=604800)
    request_timeout_seconds: float = Field(default=120, gt=0, le=3600)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_concurrency(cls, value: object) -> object:
        """Read historical tasks and old browser submissions without restoring a cap."""
        if isinstance(value, dict):
            return {
                key: item
                for key, item in value.items()
                if key != "max_concurrent_requests"
            }
        else:
            return value
