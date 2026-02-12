"""main service loop - orchestrates the fetch/match/download cycle."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from pathlib import Path
from typing import Any

from tentacle.config import Config
from tentacle.downloader import (
    build_track_path,
    download_from_source,
    get_extension_for_quality,
    _fetch_cover_art,
)
from tentacle.lidarr import LidarrClient
from tentacle.matcher import UnifiedTrack, match_album_tracks
from tentacle.qobuz import QobuzClient
from tentacle.tidal import TidalClient

log = logging.getLogger(__name__)

STATE_FILE = "tentacle_state.json"


class _TokenBucket:
    """async token bucket rate limiter — controls req/s without blocking the loop."""

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate  # tokens per second
        self._burst = burst  # max tokens
        self._tokens = float(burst)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """wait until a token is available."""
        import time

        async with self._lock:
            now = time.monotonic()
            if self._last:
                elapsed = now - self._last
                self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last = now

            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
                self._last = time.monotonic()
            else:
                self._tokens -= 1.0


class TentacleService:
    """main service that ties lidarr, tidal, matching, and downloading together."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.lidarr = LidarrClient(config.lidarr)
        self.tidal = TidalClient(
            max_retries=config.service.max_retries,
            rate_limit_delay=config.service.rate_limit_delay,
        )
        self.qobuz = QobuzClient(
            max_retries=config.service.max_retries,
            rate_limit_delay=config.service.rate_limit_delay,
        )
        self._shutdown = asyncio.Event()
        self._state_path = Path(config.download.path) / STATE_FILE
        self._attempted: set[str] = set()
        # concurrency controls
        self._dl_semaphore = asyncio.Semaphore(config.service.max_concurrent_downloads)
        self._rate_limiters: dict[str, _TokenBucket] = {
            "tidal": _TokenBucket(rate=5.0, burst=8),  # 5 req/s, burst of 8
            "qobuz": _TokenBucket(rate=3.0, burst=5),  # 3 req/s, burst of 5
        }

    def _load_state(self) -> None:
        """load previously attempted track ids from state file."""
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text())
                self._attempted = set(data.get("attempted", []))
                log.info(
                    "loaded state", extra={"attempted_count": len(self._attempted)}
                )
            except (json.JSONDecodeError, OSError) as e:
                log.warning("failed to load state", extra={"error": str(e)})

    def _save_state(self) -> None:
        """persist attempted track ids."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(
                    {
                        "attempted": list(self._attempted),
                    }
                )
            )
        except OSError as e:
            log.warning("failed to save state", extra={"error": str(e)})

    def _track_key(self, album_id: int, track_id: int) -> str:
        return f"{album_id}:{track_id}"

    async def run(self, once: bool = False) -> None:
        """run the service loop."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        self._load_state()

        log.info(
            "tentacle service starting",
            extra={
                "lidarr_url": self.config.lidarr.url,
                "quality": self.config.download.quality,
                "interval_minutes": self.config.service.scan_interval_minutes,
                "once": once,
                "source_priority": self.config.matching.source_priority,
                "prefer_explicit": self.config.matching.prefer_explicit,
            },
        )

        # rank mirrors by latency (pacman-style)
        log.info("ranking mirrors...")
        await asyncio.gather(self.tidal.rank(), self.qobuz.rank())

        try:
            while not self._shutdown.is_set():
                try:
                    await self._cycle()
                except Exception as e:
                    log.error("cycle failed", extra={"error": str(e)})

                if once:
                    log.info("single run complete, exiting")
                    break

                log.info(
                    "sleeping",
                    extra={"minutes": self.config.service.scan_interval_minutes},
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(),
                        timeout=self.config.service.scan_interval_minutes * 60,
                    )
                except asyncio.TimeoutError:
                    pass  # normal wakeup
        finally:
            self._save_state()
            await self.lidarr.close()
            await self.tidal.close()
            await self.qobuz.close()
            log.info("tentacle service stopped")

    def _handle_shutdown(self) -> None:
        log.info("shutdown signal received")
        self._shutdown.set()

    async def _cycle(self) -> None:
        """single fetch/match/download cycle."""
        log.info("starting cycle")

        wanted_albums = await self.lidarr.get_all_wanted_missing()
        if not wanted_albums:
            log.info("no wanted/missing albums found")
            return

        log.info("found wanted albums", extra={"count": len(wanted_albums)})

        for album_data in wanted_albums:
            if self._shutdown.is_set():
                break
            await self._process_album(album_data)

    async def _process_album(self, album_data: dict[str, Any]) -> None:
        """process a single wanted album."""
        album_id: int = album_data.get("id", 0)
        album_title: str = album_data.get("title", "Unknown Album")
        artist_id: int = 0
        artist_name: str = "Unknown Artist"
        artist_path: str = ""

        # extract artist info — lidarr includes the artist's disk path
        artist_obj = album_data.get("artist", {})
        if artist_obj:
            artist_id = artist_obj.get("id", 0)
            artist_name = artist_obj.get(
                "artistName", artist_obj.get("name", "Unknown Artist")
            )
            artist_path = artist_obj.get("path", "")

        # fallback: build artist path from download root if lidarr didn't provide one
        if not artist_path:
            artist_path = str(Path(self.config.download.path) / artist_name)
            log.warning(
                "no artist path from lidarr, using fallback",
                extra={"path": artist_path},
            )

        # extract release year from album data
        release_year = ""
        release_date = album_data.get("releaseDate", "")
        if release_date and len(release_date) >= 4:
            release_year = release_date[:4]

        log.info(
            "processing album",
            extra={
                "album_id": album_id,
                "album": album_title,
                "artist": artist_name,
            },
        )

        # get tracks for this album
        try:
            tracks = await self.lidarr.get_tracks(album_id)
        except Exception as e:
            log.error(
                "failed to get tracks", extra={"album_id": album_id, "error": str(e)}
            )
            return

        if not tracks:
            log.info("no tracks found for album", extra={"album_id": album_id})
            return

        # filter out already-attempted and already-downloaded tracks
        pending_tracks = []
        for t in tracks:
            track_id = t.get("id", 0)
            has_file = t.get("hasFile", False)
            key = self._track_key(album_id, track_id)

            if has_file:
                continue
            if key in self._attempted:
                log.debug("skipping already attempted", extra={"track_id": track_id})
                continue
            pending_tracks.append(t)

        if not pending_tracks:
            log.info("no pending tracks for album", extra={"album": album_title})
            return

        log.info(
            "matching tracks",
            extra={
                "album": album_title,
                "pending": len(pending_tracks),
                "total": len(tracks),
            },
        )

        # match tracks
        matches = await match_album_tracks(
            self.tidal,
            artist_name,
            album_title,
            pending_tracks,
            min_confidence=self.config.matching.min_confidence,
            qobuz=self.qobuz,
            prefer_explicit=self.config.matching.prefer_explicit,
            source_priority=self.config.matching.source_priority,
        )

        if not matches:
            log.info("no matches found for album", extra={"album": album_title})
            # mark all as attempted
            for t in pending_tracks:
                self._attempted.add(self._track_key(album_id, t.get("id", 0)))
            self._save_state()
            return

        log.info(
            "matched tracks",
            extra={"matched": len(matches), "total": len(pending_tracks)},
        )

        # build download jobs
        jobs: list[tuple[int, UnifiedTrack, Path]] = []
        ext = get_extension_for_quality(self.config.download.quality)
        for lidarr_track_id, match_result in matches.items():
            matched_track = match_result.track
            dest = build_track_path(
                artist_path,
                album_title,
                matched_track.track_number,
                matched_track.title,
                ext,
                release_year=release_year,
            )
            if dest.exists():
                log.info("file already exists, skipping", extra={"path": str(dest)})
                self._attempted.add(self._track_key(album_id, lidarr_track_id))
                continue
            jobs.append((lidarr_track_id, matched_track, dest))

        if not jobs:
            log.info("all matched tracks already exist", extra={"album": album_title})
            self._save_state()
            return

        log.info(
            "downloading tracks",
            extra={
                "count": len(jobs),
                "max_concurrent": self.config.service.max_concurrent_downloads,
            },
        )

        # fan out downloads with semaphore + rate limiting
        results = await asyncio.gather(
            *(
                self._download_worker(
                    album_id,
                    lidarr_track_id,
                    matched_track,
                    dest,
                    artist_name,
                    album_title,
                    release_year,
                    len(tracks),
                )
                for lidarr_track_id, matched_track, dest in jobs
            ),
            return_exceptions=True,
        )

        downloaded = sum(1 for r in results if r is True)
        failed = sum(1 for r in results if r is not True)
        log.info(
            "batch complete",
            extra={
                "downloaded": downloaded,
                "failed": failed,
                "album": album_title,
            },
        )

        self._save_state()

        # trigger lidarr rescan if we downloaded anything
        if downloaded > 0 and artist_id:
            log.info(
                "triggering lidarr rescan",
                extra={
                    "artist_id": artist_id,
                    "downloaded": downloaded,
                },
            )
            try:
                await self.lidarr.rescan_artist(artist_id)
                await asyncio.sleep(2)
                await self.lidarr.rename_artist_files(artist_id)
            except Exception as e:
                log.error("lidarr rescan/rename failed", extra={"error": str(e)})

    async def _download_worker(
        self,
        album_id: int,
        lidarr_track_id: int,
        track: UnifiedTrack,
        dest: Path,
        artist_name: str,
        album_title: str,
        release_year: str,
        track_total: int,
    ) -> bool:
        """single download worker — acquires semaphore + respects rate limits."""
        if self._shutdown.is_set():
            return False

        key = self._track_key(album_id, lidarr_track_id)
        self._attempted.add(key)

        # wait for rate limiter before acquiring semaphore slot
        limiter = self._rate_limiters.get(track.source)
        if limiter:
            await limiter.acquire()

        async with self._dl_semaphore:
            if self._shutdown.is_set():
                return False

            try:
                metadata: dict[str, Any] = {
                    "title": track.title,
                    "artist": track.artist,
                    "album": track.album or album_title,
                    "album_artist": artist_name,
                    "track_number": track.track_number,
                    "track_total": track_total,
                    "disc_number": track.disc_number,
                    "date": release_year,
                }

                metadata = await self._enrich_metadata(track, metadata)

                await download_from_source(
                    track,
                    dest,
                    tidal=self.tidal,
                    qobuz=self.qobuz,
                    quality=self.config.download.quality,
                    metadata=metadata,
                    max_retries=self.config.service.max_retries,
                )

                log.info(
                    "downloaded",
                    extra={
                        "title": track.title,
                        "source": track.source,
                        "has_lyrics": bool(metadata.get("lyrics")),
                        "has_cover": bool(metadata.get("cover_data")),
                    },
                )
                return True

            except Exception as e:
                log.error(
                    "download failed",
                    extra={
                        "track": track.title,
                        "source": track.source,
                        "error": str(e),
                    },
                )
                return False

    async def _enrich_metadata(
        self,
        track: UnifiedTrack,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """fetch lyrics, cover art, and rich metadata from the track's source."""

        if track.source == "tidal":
            metadata = await self._enrich_from_tidal(track, metadata)
        elif track.source == "qobuz":
            metadata = await self._enrich_from_qobuz(track, metadata)

        return metadata

    async def _enrich_from_tidal(
        self,
        track: UnifiedTrack,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """pull lyrics, cover art, ISRC, copyright, replay gain from tidal."""

        # fetch track info for ISRC, copyright, replay gain, cover id
        info = await self.tidal.get_track_info(track.id)
        if info:
            # the response may have nested track data or be flat
            track_data = info.get("track", info)

            if "isrc" in track_data:
                metadata["isrc"] = track_data["isrc"]
            if "copyright" in track_data:
                metadata["copyright"] = track_data["copyright"]

            # replay gain from track info
            metadata.update(_extract_replay_gain(info))

            # cover art from album.cover UUID
            album_obj = track_data.get("album", {})
            if isinstance(album_obj, dict):
                cover_id = album_obj.get("cover", "")
                if cover_id:
                    cover_url = TidalClient.get_cover_url(cover_id)
                    cover_data = await _fetch_cover_art(cover_url)
                    if cover_data:
                        metadata["cover_data"] = cover_data

        # fetch lyrics (synced + plain)
        lyrics_data = await self.tidal.get_lyrics(track.id)
        if lyrics_data:
            if lyrics_data.get("lyrics"):
                metadata["lyrics"] = lyrics_data["lyrics"]
            if lyrics_data.get("subtitles"):
                metadata["subtitles"] = lyrics_data["subtitles"]

        return metadata

    async def _enrich_from_qobuz(
        self,
        track: UnifiedTrack,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """pull rich metadata and cover art from qobuz (no lyrics endpoint available)."""

        # try to get album data for richer metadata + cover
        if track.album_id:
            try:
                album_data = await self.qobuz.get_album(str(track.album_id))
                if album_data:
                    tracks_list = album_data.get("tracks", album_data.get("items", []))
                    # find our track in the album for per-track metadata
                    for t in tracks_list if isinstance(tracks_list, list) else []:
                        t_obj = t.get("item", t) if isinstance(t, dict) else t
                        if isinstance(t_obj, dict) and t_obj.get("id") == track.id:
                            rich = QobuzClient.extract_rich_metadata(t_obj)
                            metadata.update(rich)
                            break
                    else:
                        # still grab album-level metadata
                        rich = QobuzClient.extract_rich_metadata({"album": album_data})
                        metadata.update(rich)

                    # cover art
                    cover_url = metadata.get("cover_url", "")
                    if not cover_url:
                        img = album_data.get("image", {})
                        if isinstance(img, dict):
                            for sz in ("mega", "extralarge", "large", "medium"):
                                if sz in img and img[sz]:
                                    cover_url = img[sz]
                                    break
                    if cover_url:
                        cover_data = await _fetch_cover_art(cover_url)
                        if cover_data:
                            metadata["cover_data"] = cover_data
                        metadata.pop("cover_url", None)
            except Exception as e:
                log.debug("failed to enrich from qobuz album", extra={"error": str(e)})

        return metadata


def _extract_replay_gain(info: dict[str, Any]) -> dict[str, Any]:
    """extract replay gain values from tidal track info response."""
    rg: dict[str, Any] = {}

    for key, meta_key in (
        ("trackReplayGain", "replay_gain"),
        ("trackPeakAmplitude", "replay_peak"),
        ("albumReplayGain", "replay_gain_album"),
        ("albumPeakAmplitude", "replay_peak_album"),
    ):
        val = info.get(key)
        if val is not None:
            try:
                rg[meta_key] = float(val)
            except (ValueError, TypeError):
                pass

    return rg
