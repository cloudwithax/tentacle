"""configuration loading from yaml + env vars."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import yaml

log = logging.getLogger(__name__)


@dataclass
class LidarrConfig:
    url: str = "http://localhost:8686"
    api_key: str = ""


@dataclass
class DownloadConfig:
    path: str = "/music"
    quality: str = "LOSSLESS"


@dataclass
class ServiceConfig:
    scan_interval_minutes: int = 15
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    max_concurrent_downloads: int = 25


@dataclass
class MatchingConfig:
    min_confidence: float = 0.80
    prefer_explicit: bool = True
    source_priority: list[str] = field(default_factory=lambda: ["tidal"])


@dataclass
class Config:
    lidarr: LidarrConfig = field(default_factory=LidarrConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Self:
        """load config from yaml file, then overlay env vars."""
        cfg = cls()

        if path and Path(path).exists():
            log.info("loading config", extra={"path": str(path)})
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            cfg = cls._from_dict(raw)

        cfg._apply_env_overrides()
        cfg._validate()
        return cfg

    @classmethod
    def _from_dict(cls, d: dict) -> Self:
        lidarr_d = d.get("lidarr", {})
        download_d = d.get("download", {})
        service_d = d.get("service", {})
        matching_d = d.get("matching", {})

        return cls(
            lidarr=LidarrConfig(
                url=lidarr_d.get("url", "http://localhost:8686"),
                api_key=lidarr_d.get("api_key", ""),
            ),
            download=DownloadConfig(
                path=download_d.get("path", "/music"),
                quality=download_d.get("quality", "LOSSLESS"),
            ),
            service=ServiceConfig(
                scan_interval_minutes=int(service_d.get("scan_interval_minutes", 15)),
                max_retries=int(service_d.get("max_retries", 3)),
                rate_limit_delay=float(service_d.get("rate_limit_delay", 1.0)),
                max_concurrent_downloads=int(
                    service_d.get("max_concurrent_downloads", 25)
                ),
            ),
            matching=MatchingConfig(
                min_confidence=float(matching_d.get("min_confidence", 0.80)),
                prefer_explicit=bool(matching_d.get("prefer_explicit", True)),
                source_priority=matching_d.get("source_priority", ["tidal"]),
            ),
        )

    def _apply_env_overrides(self) -> None:
        """overlay TENTACLE_* env vars onto config."""
        env_map: list[tuple[str, object, str, type]] = [
            ("TENTACLE_LIDARR_URL", self.lidarr, "url", str),
            ("TENTACLE_LIDARR_API_KEY", self.lidarr, "api_key", str),
            ("TENTACLE_DOWNLOAD_PATH", self.download, "path", str),
            ("TENTACLE_DOWNLOAD_QUALITY", self.download, "quality", str),
            (
                "TENTACLE_SERVICE_SCAN_INTERVAL_MINUTES",
                self.service,
                "scan_interval_minutes",
                int,
            ),
            ("TENTACLE_SERVICE_MAX_RETRIES", self.service, "max_retries", int),
            (
                "TENTACLE_SERVICE_RATE_LIMIT_DELAY",
                self.service,
                "rate_limit_delay",
                float,
            ),
            (
                "TENTACLE_SERVICE_MAX_CONCURRENT_DOWNLOADS",
                self.service,
                "max_concurrent_downloads",
                int,
            ),
            (
                "TENTACLE_MATCHING_MIN_CONFIDENCE",
                self.matching,
                "min_confidence",
                float,
            ),
            (
                "TENTACLE_MATCHING_PREFER_EXPLICIT",
                self.matching,
                "prefer_explicit",
                lambda v: v.lower() in ("1", "true", "yes"),
            ),
        ]
        for env_key, obj, attr, typ in env_map:
            val = os.environ.get(env_key)
            if val is not None:
                log.info("env override", extra={"key": env_key})
                setattr(obj, attr, typ(val))

    def _validate(self) -> None:
        if not self.lidarr.api_key:
            raise ValueError(
                "lidarr api_key is required (config or TENTACLE_LIDARR_API_KEY)"
            )
        valid_qualities = {"HI_RES_LOSSLESS", "LOSSLESS", "HIGH", "LOW"}
        if self.download.quality not in valid_qualities:
            raise ValueError(
                f"quality must be one of {valid_qualities}, got {self.download.quality!r}"
            )

        valid_sources = {"tidal"}
        for src in self.matching.source_priority:
            if src not in valid_sources:
                raise ValueError(
                    f"source_priority contains invalid source {src!r}, must be one of {valid_sources}"
                )

        # resolve and validate download path
        dl = Path(self.download.path).resolve()
        self.download.path = str(dl)
        if not dl.exists():
            log.warning(
                "download path does not exist, creating", extra={"path": str(dl)}
            )
            try:
                dl.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ValueError(f"cannot create download path {dl}: {e}") from e
        if not dl.is_dir():
            raise ValueError(f"download path is not a directory: {dl}")
