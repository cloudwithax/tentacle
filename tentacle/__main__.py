"""entry point for tentacle."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from tentacle.config import Config
from tentacle.service import TentacleService


def setup_logging() -> None:
    """configure structured logging."""
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # quiet noisy libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tentacle",
        description="automated lidarr track downloader via tidal",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="path to config.yml",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single cycle then exit",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config_path = args.config
    if config_path:
        from pathlib import Path
        config_path = str(Path(config_path).resolve())
        if not Path(config_path).exists():
            import os
            if os.environ.get("TENTACLE_LIDARR_API_KEY"):
                logging.info("config file missing, regenerating from env vars: %s", config_path)
            else:
                logging.error("config file not found: %s", config_path)
                sys.exit(1)

    try:
        config = Config.load(config_path)
    except ValueError as e:
        logging.error("config error: %s", e)
        sys.exit(1)

    if config_path and not Path(config_path).exists():
        config.save(config_path)

    service = TentacleService(config)
    asyncio.run(service.run(once=args.once))


if __name__ == "__main__":
    main()
