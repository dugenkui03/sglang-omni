"""Provide the workbench's independent process entry point."""

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from tts_workbench.app import create_app
from tts_workbench.config import Settings


def main() -> None:
    """Start one workbench process with explicit deployment paths and endpoint."""
    parser = argparse.ArgumentParser(
        description="Independent TTS load testing and observation workbench"
    )
    parser.add_argument(
        "--api-base", required=True, help="TTS HTTP base URL, without /v1/audio/speech"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing the dialogue manifest.json",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Workbench-owned result directory",
    )
    parser.add_argument(
        "--log-file", type=Path, help="Read-only TTS stdout/stderr log file"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Workbench listen address")
    parser.add_argument("--port", type=int, default=8081, help="Workbench HTTP port")
    parser.add_argument(
        "--password-env",
        default="TTS_WORKBENCH_PASSWORD",
        help="Environment variable holding the workbench password",
    )
    parser.add_argument(
        "--api-key-env",
        default="TTS_API_KEY",
        help="Optional TTS bearer-key environment variable",
    )
    parser.add_argument(
        "--audio-quota-mib",
        type=int,
        default=1024,
        help="Maximum sampled audio MiB per task",
    )
    parser.add_argument(
        "--allow-unauthenticated-local",
        action="store_true",
        help="Allow password-free loopback-only local development",
    )
    arguments = parser.parse_args()
    password = os.environ.get(arguments.password_env)
    if not password and not (
        arguments.allow_unauthenticated_local
        and arguments.host in {"127.0.0.1", "localhost", "::1"}
    ):
        parser.error(
            f"Set {arguments.password_env}; public listeners always require a password"
        )
    else:
        pass
    settings = Settings(
        api_base=arguments.api_base,
        data_directory=arguments.data_dir.resolve(),
        results_directory=arguments.results_dir.resolve(),
        log_file=arguments.log_file.resolve() if arguments.log_file else None,
        password=password,
        api_key=os.environ.get(arguments.api_key_env),
        audio_quota_bytes=arguments.audio_quota_mib * 1024**2,
    )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        app = create_app(settings)
    except (ValueError, OSError) as exception:
        parser.error(str(exception))
    uvicorn.run(
        app, host=arguments.host, port=arguments.port, workers=1, log_level="info"
    )
