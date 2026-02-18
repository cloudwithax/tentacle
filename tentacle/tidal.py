"""tidal hifi-api client with ranked instance selection and failover."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import httpx

from tentacle.ranker import RankedMirrorPool, rank_mirrors

log = logging.getLogger(__name__)

INSTANCES: list[tuple[str, int]] = [
    ("https://triton.squid.wtf", 15),
    ("https://hifi-one.spotisaver.net", 15),
    ("https://hifi-two.spotisaver.net", 15),
    ("https://tidal.kinoplus.online", 15),
    ("https://tidal-api.binimum.org", 10),
    ("https://hund.qqdl.site", 15),
    ("https://katze.qqdl.site", 15),
    ("https://maus.qqdl.site", 15),
    ("https://vogel.qqdl.site", 15),
    ("https://wolf.qqdl.site", 15),
    ("https://arran.monochrome.tf", 15),
    ("https://monochrome-api.samidy.com", 15),
    ("https://api.monochrome.tf", 15),
]

QUALITY_FALLBACK_ORDER = ["HI_RES_LOSSLESS", "LOSSLESS", "HIGH", "LOW"]


@dataclass
class StreamInfo:
    """parsed stream info from tidal api."""

    url: str
    codec: str
    mime_type: str
    is_dash: bool
    manifest_raw: str


@dataclass
class TidalTrack:
    """a tidal track result."""

    id: int
    title: str
    artist: str
    album: str
    duration: int  # seconds
    track_number: int
    disc_number: int
    album_id: int
    explicit: bool = False


class TidalClient:
    """async client for hifi-api tidal instances."""

    def __init__(self, max_retries: int = 3, rate_limit_delay: float = 1.0) -> None:
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self._pool: RankedMirrorPool | None = None
        self._client: httpx.AsyncClient | None = None

    async def rank(self) -> None:
        """rank all tidal mirrors by latency. call once on startup."""
        urls = [url for url, _ in INSTANCES]
        results = await rank_mirrors(
            urls,
            probe_path="/search/?s=test",
            timeout=8.0,
            rounds=2,
        )
        self._pool = RankedMirrorPool(results)
        log.info("tidal mirrors ranked", extra={"pool": repr(self._pool)})

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _pick_instance(self) -> str:
        """pick next mirror from ranked pool, falling back to first defined if unranked."""
        if self._pool:
            return self._pool.pick()
        return INSTANCES[0][0]

    def _demote_instance(self, url: str) -> None:
        """demote a failing instance in the ranked pool."""
        if self._pool:
            self._pool.demote(url)

    async def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """make a request, trying every alive instance before giving up."""
        last_exc: Exception | None = None
        tried: set[str] = set()
        pool_size = len(self._pool) if self._pool else len(INSTANCES)
        max_attempts = max(self.max_retries, pool_size)

        for attempt in range(max_attempts):
            instance = self._pick_instance()

            # if we've cycled through every instance, stop
            if instance in tried and len(tried) >= pool_size:
                break
            tried.add(instance)

            url = f"{instance}{path}"

            try:
                client = await self._get_client()
                resp = await client.get(url, params=params)

                if resp.status_code == 429:
                    delay = self.rate_limit_delay * (2 ** min(attempt, 4))
                    log.warning(
                        "rate limited, backing off",
                        extra={
                            "instance": instance,
                            "delay": delay,
                            "attempt": attempt + 1,
                        },
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code == 401:
                    log.warning("401 from instance", extra={"instance": instance})
                    self._demote_instance(instance)
                    continue

                resp.raise_for_status()
                body = resp.json()
                # unwrap the {version, data: {...}} envelope if present
                if isinstance(body, dict) and "data" in body and "version" in body:
                    body = body["data"]
                if self._pool:
                    self._pool.record_success(instance)
                return body

            except httpx.HTTPStatusError as e:
                log.error(
                    "http error from instance",
                    extra={
                        "instance": instance,
                        "status": e.response.status_code,
                        "attempt": attempt + 1,
                    },
                )
                self._demote_instance(instance)
                last_exc = e

            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError) as e:
                log.error(
                    "connection error",
                    extra={
                        "instance": instance,
                        "error": str(e),
                        "attempt": attempt + 1,
                    },
                )
                self._demote_instance(instance)
                last_exc = e

        raise ConnectionError(
            f"all {len(tried)}/{pool_size} instances exhausted"
        ) from last_exc

    @staticmethod
    def _extract_items(data: Any, kind: str = "tracks") -> list[dict[str, Any]]:
        """extract items list from already-unwrapped response data.

        handles flat ({items: [...]}) and categorized
        ({tracks: {items: [...]}, albums: {items: []}}) shapes.
        """
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        if "items" in data:
            return data["items"]
        bucket = data.get(kind, {})
        if isinstance(bucket, dict) and "items" in bucket:
            return bucket["items"]
        return []

    async def search_tracks(self, query: str) -> list[dict[str, Any]]:
        """search for tracks by query string."""
        log.info("searching tracks", extra={"query": query})
        data = await self._request("/search/", params={"s": query})
        return self._extract_items(data)

    async def search_albums(self, query: str) -> list[dict[str, Any]]:
        """search for albums."""
        log.info("searching albums", extra={"query": query})
        data = await self._request("/search/", params={"al": query})
        return self._extract_items(data, kind="albums")

    async def search_artists(self, query: str) -> list[dict[str, Any]]:
        """search for artists."""
        data = await self._request("/search/", params={"a": query})
        return self._extract_items(data, kind="artists")

    async def get_track_stream(
        self, track_id: int, quality: str = "LOSSLESS"
    ) -> StreamInfo:
        """get stream info for a track, with quality fallback."""
        start_idx = (
            QUALITY_FALLBACK_ORDER.index(quality)
            if quality in QUALITY_FALLBACK_ORDER
            else 1
        )

        for q in QUALITY_FALLBACK_ORDER[start_idx:]:
            try:
                data = await self._request(
                    "/track/", params={"id": track_id, "quality": q}
                )
                return self._parse_stream_info(data)
            except (ConnectionError, KeyError, ValueError) as e:
                if q == "LOW":
                    raise
                log.warning("quality fallback", extra={"from": q, "error": str(e)})

        raise ValueError(f"could not get stream for track {track_id}")

    async def get_album(self, album_id: int) -> dict[str, Any]:
        """get album info with tracks."""
        return await self._request("/album/", params={"id": album_id})

    async def get_lyrics(self, track_id: int) -> dict[str, Any] | None:
        """fetch lyrics for a track. returns dict with 'lyrics' (plain text) and 'subtitles' (synced LRC) fields."""
        try:
            data = await self._request("/lyrics/", params={"id": track_id})
            if isinstance(data, list):
                data = data[0] if data else {}
            return data
        except Exception as e:
            log.debug(
                "no lyrics available", extra={"track_id": track_id, "error": str(e)}
            )
            return None

    async def get_track_info(self, track_id: int) -> dict[str, Any] | None:
        """fetch full track metadata including ISRC, copyright, replay gain etc."""
        try:
            return await self._request(
                "/track/", params={"id": track_id, "quality": "LOSSLESS"}
            )
        except Exception as e:
            log.debug(
                "failed to get track info",
                extra={"track_id": track_id, "error": str(e)},
            )
            return None

    @staticmethod
    def get_cover_url(cover_id: str, size: int = 1280) -> str:
        """build tidal cover art URL from cover UUID."""
        if not cover_id:
            return ""
        return f"https://resources.tidal.com/images/{cover_id.replace('-', '/')}/{size}x{size}.jpg"

    @staticmethod
    def _parse_stream_info(data: dict[str, Any]) -> StreamInfo:
        """parse stream info from api response, handling both DASH and direct URL manifests."""
        manifest_b64 = data.get("manifest", "")
        mime_type = data.get("manifestMimeType", data.get("mimeType", ""))
        codec = data.get("codec", data.get("audioQuality", ""))

        if not manifest_b64:
            raise ValueError("no manifest in track response")

        manifest_raw = base64.b64decode(manifest_b64).decode("utf-8", errors="replace")
        is_dash = (
            "mpd" in mime_type.lower()
            or manifest_raw.strip().startswith("<?xml")
            or "<MPD" in manifest_raw
        )

        if is_dash:
            url = ""  # DASH needs segment downloading
        else:
            try:
                manifest_data = json.loads(manifest_raw)
                url = manifest_data.get("url", manifest_data.get("urls", [""])[0])
            except json.JSONDecodeError:
                url = manifest_raw.strip()

        return StreamInfo(
            url=url,
            codec=codec,
            mime_type=mime_type,
            is_dash=is_dash,
            manifest_raw=manifest_raw,
        )

    @staticmethod
    def parse_dash_manifest(manifest_xml: str) -> list[str]:
        """parse DASH MPD manifest and extract segment URLs."""
        root = ET.fromstring(manifest_xml)
        ns = {"mpd": "urn:mpeg:dash:schema:mpd:2011"}
        urls: list[str] = []

        # try to find SegmentTimeline-based segments
        for adaptation_set in root.iter():
            if (
                adaptation_set.tag.endswith("AdaptationSet")
                or adaptation_set.tag == "AdaptationSet"
            ):
                for rep in adaptation_set:
                    if (
                        rep.tag.endswith("Representation")
                        or rep.tag == "Representation"
                    ):
                        for seg_tmpl in rep:
                            if (
                                seg_tmpl.tag.endswith("SegmentTemplate")
                                or seg_tmpl.tag == "SegmentTemplate"
                            ):
                                init = seg_tmpl.get("initialization", "")
                                media = seg_tmpl.get("media", "")
                                if init:
                                    urls.append(init)
                                for timeline in seg_tmpl:
                                    if timeline.tag.endswith("SegmentTimeline"):
                                        t = 0
                                        for s in timeline:
                                            st = int(s.get("t", t))
                                            d = int(s.get("d", 0))
                                            r = int(s.get("r", 0))
                                            for i in range(r + 1):
                                                seg_url = media.replace(
                                                    "$Number$", str(len(urls))
                                                ).replace("$Time$", str(st))
                                                urls.append(seg_url)
                                                st += d
                                                t = st

        # fallback: look for BaseURL
        if not urls:
            for base_url in root.iter():
                if base_url.tag.endswith("BaseURL") or base_url.tag == "BaseURL":
                    if base_url.text and base_url.text.startswith("http"):
                        urls.append(base_url.text.strip())

        return urls

    @staticmethod
    def parse_tidal_track(data: dict[str, Any]) -> TidalTrack:
        """parse a tidal track from api response."""
        artist_name = ""
        if "artist" in data:
            artist_obj = data["artist"]
            artist_name = (
                artist_obj.get("name", "")
                if isinstance(artist_obj, dict)
                else str(artist_obj)
            )
        elif "artists" in data and data["artists"]:
            first = data["artists"][0]
            artist_name = (
                first.get("name", "") if isinstance(first, dict) else str(first)
            )

        album_name = ""
        if "album" in data:
            album_obj = data["album"]
            album_name = (
                album_obj.get("title", "")
                if isinstance(album_obj, dict)
                else str(album_obj)
            )

        return TidalTrack(
            id=int(data.get("id", 0)),
            title=data.get("title", ""),
            artist=artist_name,
            album=album_name,
            duration=int(data.get("duration", 0)),
            track_number=int(data.get("trackNumber", 1)),
            disc_number=int(data.get("volumeNumber", data.get("discNumber", 1))),
            album_id=(
                int(data.get("album", {}).get("id", 0))
                if isinstance(data.get("album"), dict)
                else 0
            ),
            explicit=bool(data.get("explicit", False)),
        )
