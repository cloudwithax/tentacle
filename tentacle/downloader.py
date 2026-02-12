"""download manager for tidal and qobuz tracks with metadata embedding."""

from __future__ import annotations

import asyncio
import io
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from mutagen.flac import FLAC, Picture
from mutagen.mp4 import MP4

from tentacle.matcher import UnifiedTrack
from tentacle.qobuz import QobuzClient
from tentacle.tidal import StreamInfo, TidalClient

log = logging.getLogger(__name__)


async def _fetch_cover_art(url: str) -> bytes | None:
    """download cover art from URL."""
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        log.debug("failed to fetch cover art", extra={"url": url, "error": str(e)})
        return None


def _save_lrc(audio_path: Path, subtitles: str) -> None:
    """save synced lyrics as .lrc sidecar file next to the audio file."""
    if not subtitles or not subtitles.strip():
        return
    lrc_path = audio_path.with_suffix(".lrc")
    try:
        lrc_path.write_text(subtitles, encoding="utf-8")
        log.debug("saved lrc", extra={"path": str(lrc_path)})
    except OSError as e:
        log.warning("failed to save lrc", extra={"error": str(e)})


def sanitize_filename(name: str) -> str:
    """sanitize a string for use as a filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = name.strip(". ")
    return name or "Unknown"


def get_extension(stream: StreamInfo) -> str:
    """determine file extension from stream info."""
    codec = stream.codec.upper()
    if "FLAC" in codec:
        return ".flac"
    if "AAC" in codec or "MP4" in codec or "M4A" in codec:
        return ".m4a"
    if stream.mime_type and "flac" in stream.mime_type.lower():
        return ".flac"
    if stream.mime_type and "mp4" in stream.mime_type.lower():
        return ".m4a"
    # default to flac for lossless
    return ".flac"


def build_track_path(
    artist_path: str,
    album_title: str,
    track_number: int,
    track_title: str,
    extension: str,
    release_year: str | int = "",
) -> Path:
    """build the full file path following lidarr's default naming convention.

    lidarr standard: {Artist Path}/{Album Title} ({Release Year})/{TrackNumber} {Track Title}.ext
    artist_path comes directly from lidarr's artist.path field.
    """
    if release_year:
        album_dir = sanitize_filename(f"{album_title} ({release_year})")
    else:
        album_dir = sanitize_filename(album_title)
    filename = f"{track_number:02d} {sanitize_filename(track_title)}{extension}"
    return Path(artist_path) / album_dir / filename


async def download_track(
    tidal: TidalClient,
    track: UnifiedTrack,
    stream: StreamInfo,
    dest_path: Path,
    metadata: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> Path:
    """download a track from its stream info to dest_path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("downloading track", extra={
        "track_id": track.id,
        "title": track.title,
        "artist": track.artist,
        "dest": str(dest_path),
        "is_dash": stream.is_dash,
    })

    if stream.is_dash:
        audio_data = await _download_dash(tidal, stream, max_retries)
    else:
        audio_data = await _download_direct(stream.url, max_retries)

    dest_path.write_bytes(audio_data)
    log.info("track downloaded", extra={"path": str(dest_path), "size_mb": f"{len(audio_data) / 1024 / 1024:.1f}"})

    # embed metadata
    if metadata:
        _embed_metadata(dest_path, metadata)

    return dest_path


async def download_qobuz_track(
    qobuz: QobuzClient,
    track: UnifiedTrack,
    dest_path: Path,
    quality: str = "LOSSLESS",
    metadata: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> Path:
    """download a track from qobuz using direct stream URL."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("downloading from qobuz", extra={
        "track_id": track.id, "title": track.title, "dest": str(dest_path),
    })

    url = await qobuz.get_stream_url(track.id, quality=quality)
    audio_data = await _download_direct(url, max_retries)

    dest_path.write_bytes(audio_data)
    log.info("track downloaded", extra={"path": str(dest_path), "size_mb": f"{len(audio_data) / 1024 / 1024:.1f}"})

    if metadata:
        _embed_metadata(dest_path, metadata)

    return dest_path


async def download_from_source(
    track: UnifiedTrack,
    dest_path: Path,
    tidal: TidalClient | None = None,
    qobuz: QobuzClient | None = None,
    quality: str = "LOSSLESS",
    metadata: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> Path:
    """download a track from the appropriate source."""
    if track.source == "qobuz" and qobuz:
        return await download_qobuz_track(qobuz, track, dest_path, quality, metadata, max_retries)
    elif track.source == "tidal" and tidal:
        stream = await tidal.get_track_stream(track.id, quality=quality)
        return await download_track(tidal, track, stream, dest_path, metadata, max_retries)
    else:
        raise ValueError(f"no client available for source={track.source}")


def get_extension_for_quality(quality: str) -> str:
    """determine extension based on quality level."""
    if quality in ("HI_RES_LOSSLESS", "LOSSLESS"):
        return ".flac"
    return ".m4a"


async def _download_direct(url: str, max_retries: int) -> bytes:
    """download a direct URL with retry."""
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
        except (httpx.HTTPError, httpx.StreamError) as e:
            log.warning("download attempt failed", extra={
                "attempt": attempt + 1, "error": str(e),
            })
            last_exc = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    raise ConnectionError(f"failed to download after {max_retries} attempts") from last_exc


async def _download_dash(tidal: TidalClient, stream: StreamInfo, max_retries: int) -> bytes:
    """download DASH segments and concatenate."""
    segment_urls = TidalClient.parse_dash_manifest(stream.manifest_raw)

    if not segment_urls:
        raise ValueError("no segments found in DASH manifest")

    log.info("downloading DASH segments", extra={"segment_count": len(segment_urls)})

    buf = io.BytesIO()

    for i, seg_url in enumerate(segment_urls):
        data = await _download_direct(seg_url, max_retries)
        buf.write(data)

        if (i + 1) % 10 == 0:
            log.debug("dash progress", extra={"segment": i + 1, "total": len(segment_urls)})

    return buf.getvalue()


def _embed_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """embed metadata tags into audio file."""
    ext = path.suffix.lower()

    try:
        if ext == ".flac":
            _tag_flac(path, metadata)
        elif ext in (".m4a", ".mp4"):
            _tag_m4a(path, metadata)
        else:
            log.warning("unknown format for tagging", extra={"extension": ext})
    except Exception as e:
        log.error("failed to embed metadata", extra={"path": str(path), "error": str(e)})

    # save synced lyrics as .lrc sidecar
    subtitles = metadata.get("subtitles", "")
    if subtitles:
        _save_lrc(path, subtitles)


def _tag_flac(path: Path, meta: dict[str, Any]) -> None:
    """tag a FLAC file with full metadata."""
    audio = FLAC(str(path))

    tag_map = {
        "title": "title",
        "artist": "artist",
        "album": "album",
        "album_artist": "albumartist",
        "genre": "genre",
        "date": "date",
        "year": "date",
        "isrc": "isrc",
        "copyright": "copyright",
        "label": "organization",
        "composer": "composer",
        "performers": "performer",
        "lyrics": "lyrics",
        "comment": "comment",
        "upc": "upc",
    }

    for meta_key, vorbis_key in tag_map.items():
        val = meta.get(meta_key)
        if val:
            if isinstance(val, list):
                audio[vorbis_key] = [str(v) for v in val]
            else:
                audio[vorbis_key] = str(val)

    # track number
    if "track_number" in meta:
        total = meta.get("track_total", "")
        if total:
            audio["tracknumber"] = f"{meta['track_number']}/{total}"
        else:
            audio["tracknumber"] = str(meta["track_number"])

    # disc number
    if "disc_number" in meta:
        disc_total = meta.get("disc_total", "")
        if disc_total:
            audio["discnumber"] = f"{meta['disc_number']}/{disc_total}"
        else:
            audio["discnumber"] = str(meta["disc_number"])

    # replay gain
    if "replay_gain" in meta:
        audio["replaygain_track_gain"] = f"{meta['replay_gain']:.2f} dB"
    if "replay_peak" in meta:
        audio["replaygain_track_peak"] = f"{meta['replay_peak']:.6f}"
    if "replay_gain_album" in meta:
        audio["replaygain_album_gain"] = f"{meta['replay_gain_album']:.2f} dB"
    if "replay_peak_album" in meta:
        audio["replaygain_album_peak"] = f"{meta['replay_peak_album']:.6f}"

    # cover art
    cover_data = meta.get("cover_data")
    if cover_data and isinstance(cover_data, bytes):
        pic = Picture()
        pic.type = 3  # front cover
        pic.mime = "image/jpeg"
        if cover_data[:4] == b'\x89PNG':
            pic.mime = "image/png"
        pic.data = cover_data
        audio.add_picture(pic)

    audio.save()
    log.debug("tagged flac", extra={"path": str(path)})


def _tag_m4a(path: Path, meta: dict[str, Any]) -> None:
    """tag an M4A/MP4 file with full metadata."""
    audio = MP4(str(path))

    tag_map = {
        "title": "\xa9nam",
        "artist": "\xa9ART",
        "album": "\xa9alb",
        "album_artist": "aART",
        "genre": "\xa9gen",
        "date": "\xa9day",
        "year": "\xa9day",
        "copyright": "cprt",
        "composer": "\xa9wrt",
        "comment": "\xa9cmt",
        "lyrics": "\xa9lyr",
    }

    for meta_key, mp4_key in tag_map.items():
        val = meta.get(meta_key)
        if val:
            if isinstance(val, list):
                audio[mp4_key] = [str(v) for v in val]
            else:
                audio[mp4_key] = [str(val)]

    # track number
    if "track_number" in meta:
        total = meta.get("track_total", 0)
        audio["trkn"] = [(int(meta["track_number"]), int(total) if total else 0)]

    # disc number
    if "disc_number" in meta:
        disc_total = meta.get("disc_total", 0)
        audio["disk"] = [(int(meta["disc_number"]), int(disc_total) if disc_total else 0)]

    # cover art
    cover_data = meta.get("cover_data")
    if cover_data and isinstance(cover_data, bytes):
        from mutagen.mp4 import MP4Cover
        fmt = MP4Cover.FORMAT_JPEG
        if cover_data[:4] == b'\x89PNG':
            fmt = MP4Cover.FORMAT_PNG
        audio["covr"] = [MP4Cover(cover_data, imageformat=fmt)]

    audio.save()
    log.debug("tagged m4a", extra={"path": str(path)})
