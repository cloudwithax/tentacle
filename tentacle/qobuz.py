"""qobuz api client with ranked instance selection and failover."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from tentacle.ranker import RankedMirrorPool, rank_mirrors

log = logging.getLogger(__name__)

INSTANCES: list[tuple[str, int]] = [
    ("https://qobuz.squid.wtf", 15),
]

QOBUZ_QUALITY_MAP = {
    "HI_RES_LOSSLESS": "27",
    "LOSSLESS": "7",
    "HIGH": "6",
    "LOW": "5",
}

QUALITY_FALLBACK_ORDER = ["HI_RES_LOSSLESS", "LOSSLESS", "HIGH", "LOW"]


@dataclass
class QobuzTrack:
    """a qobuz track result."""
    id: int
    title: str
    artist: str
    album: str
    duration: int  # seconds
    track_number: int
    disc_number: int
    album_id: str


class QobuzClient:
    """async client for qobuz api instances."""

    def __init__(
        self,
        max_retries: int = 3,
        rate_limit_delay: float = 1.0,
        country: str | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.country = country
        self._pool: RankedMirrorPool | None = None
        self._client: httpx.AsyncClient | None = None

    async def rank(self) -> None:
        """rank all qobuz mirrors by latency. call once on startup."""
        urls = [url for url, _ in INSTANCES]
        results = await rank_mirrors(
            urls,
            probe_path="/api/get-countries",
            timeout=8.0,
            rounds=2,
        )
        self._pool = RankedMirrorPool(results)
        log.info("qobuz mirrors ranked", extra={"pool": repr(self._pool)})

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _pick_instance(self) -> str:
        """pick next mirror from ranked pool."""
        if self._pool:
            return self._pool.pick()
        return INSTANCES[0][0]

    def _demote_instance(self, url: str) -> None:
        """demote a failing instance in the ranked pool."""
        if self._pool:
            self._pool.demote(url)

    async def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """make a request with retry, failover, and rate limit handling."""
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            instance = self._pick_instance()
            url = f"{instance}/api{path}"

            headers: dict[str, str] = {}
            if self.country:
                headers["Token-Country"] = self.country

            try:
                client = await self._get_client()
                resp = await client.get(url, params=params, headers=headers)

                if resp.status_code == 429:
                    delay = self.rate_limit_delay * (2 ** attempt)
                    log.warning("rate limited, backing off", extra={
                        "instance": instance, "delay": delay, "attempt": attempt + 1,
                    })
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code == 401:
                    log.warning("401 from instance", extra={"instance": instance})
                    self._demote_instance(instance)
                    continue

                resp.raise_for_status()
                body = resp.json()

                if not body.get("success", False):
                    log.error("api returned success=false", extra={
                        "instance": instance, "path": path,
                    })
                    raise ValueError(f"qobuz api error on {path}")

                return body.get("data", body)

            except httpx.HTTPStatusError as e:
                log.error("http error from instance", extra={
                    "instance": instance, "status": e.response.status_code, "attempt": attempt + 1,
                })
                self._demote_instance(instance)
                last_exc = e

            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError) as e:
                log.error("connection error", extra={
                    "instance": instance, "error": str(e), "attempt": attempt + 1,
                })
                self._demote_instance(instance)
                last_exc = e

            if attempt < self.max_retries - 1:
                delay = self.rate_limit_delay * (2 ** attempt)
                await asyncio.sleep(delay)

        raise ConnectionError(f"all {self.max_retries} attempts failed") from last_exc

    async def search_tracks(self, query: str) -> list[dict[str, Any]]:
        """search for tracks by query string."""
        log.info("searching tracks", extra={"query": query})
        data = await self._request("/get-music", params={"q": query, "offset": 0})
        try:
            return data["tracks"]["items"]
        except (KeyError, TypeError):
            return []

    async def search_albums(self, query: str) -> list[dict[str, Any]]:
        """search for albums."""
        log.info("searching albums", extra={"query": query})
        data = await self._request("/get-music", params={"q": query, "offset": 0})
        try:
            return data["albums"]["items"]
        except (KeyError, TypeError):
            return []

    async def search_artists(self, query: str) -> list[dict[str, Any]]:
        """search for artists."""
        log.info("searching artists", extra={"query": query})
        data = await self._request("/get-music", params={"q": query, "offset": 0})
        try:
            return data["artists"]["items"]
        except (KeyError, TypeError):
            return []

    async def get_album(self, album_id: str) -> dict[str, Any]:
        """get album info with tracks."""
        log.info("fetching album", extra={"album_id": album_id})
        return await self._request("/get-album", params={"album_id": album_id})

    async def get_artist(self, artist_id: str) -> dict[str, Any]:
        """get artist info."""
        log.info("fetching artist", extra={"artist_id": artist_id})
        return await self._request("/get-artist", params={"artist_id": artist_id})

    async def get_stream_url(self, track_id: int, quality: str = "LOSSLESS") -> str:
        """get direct download/stream url for a track, with quality fallback."""
        q_value = QOBUZ_QUALITY_MAP.get(quality)
        if q_value is None:
            # maybe they passed the raw numeric value
            if quality in QOBUZ_QUALITY_MAP.values():
                q_value = quality
            else:
                q_value = QOBUZ_QUALITY_MAP["LOSSLESS"]

        start_idx = (
            QUALITY_FALLBACK_ORDER.index(quality)
            if quality in QUALITY_FALLBACK_ORDER
            else 1
        )

        for q in QUALITY_FALLBACK_ORDER[start_idx:]:
            try:
                data = await self._request(
                    "/download-music",
                    params={"track_id": track_id, "quality": QOBUZ_QUALITY_MAP[q]},
                )
                url = data.get("url", "")
                if url:
                    log.info("got stream url", extra={
                        "track_id": track_id, "quality": q,
                    })
                    return url
                log.warning("empty url in response", extra={
                    "track_id": track_id, "quality": q,
                })
            except (ConnectionError, KeyError, ValueError) as e:
                if q == "LOW":
                    raise
                log.warning("quality fallback", extra={"from": q, "error": str(e)})

        raise ValueError(f"could not get stream url for track {track_id}")

    async def get_countries(self) -> Any:
        """get available countries for geo-restriction."""
        return await self._request("/get-countries")

    @staticmethod
    def extract_rich_metadata(track_data: dict[str, Any]) -> dict[str, Any]:
        """extract rich metadata fields from a qobuz track response."""
        meta: dict[str, Any] = {}

        # ISRC
        if "isrc" in track_data:
            meta["isrc"] = track_data["isrc"]

        # copyright
        if "copyright" in track_data:
            meta["copyright"] = track_data["copyright"]

        # composer
        if "composer" in track_data:
            comp = track_data["composer"]
            meta["composer"] = comp.get("name", "") if isinstance(comp, dict) else str(comp)

        # performers string
        if "performers" in track_data:
            meta["performers"] = track_data["performers"]

        # explicit
        if "parental_warning" in track_data:
            meta["explicit"] = track_data["parental_warning"]

        # audio quality info
        if "maximum_bit_depth" in track_data:
            meta["bit_depth"] = track_data["maximum_bit_depth"]
        if "maximum_sampling_rate" in track_data:
            meta["sample_rate"] = track_data["maximum_sampling_rate"]

        # album-level metadata
        album = track_data.get("album", {})
        if isinstance(album, dict):
            if "label" in album:
                label = album["label"]
                meta["label"] = label.get("name", "") if isinstance(label, dict) else str(label)
            if "genres_list" in album:
                meta["genre"] = album["genres_list"]
            if "image" in album:
                img = album["image"]
                if isinstance(img, dict):
                    # prefer largest available
                    for size in ("mega", "extralarge", "large", "medium", "small"):
                        if size in img and img[size]:
                            meta["cover_url"] = img[size]
                            break
            if "copyright" in album and "copyright" not in meta:
                meta["copyright"] = album["copyright"]
            if "upc" in album:
                meta["upc"] = album["upc"]

        return meta

    @staticmethod
    def parse_qobuz_track(data: dict[str, Any]) -> QobuzTrack:
        """parse a qobuz track from api response."""
        artist_name = ""
        if "performer" in data:
            perf = data["performer"]
            artist_name = perf.get("name", "") if isinstance(perf, dict) else str(perf)
        elif "artist" in data:
            artist_obj = data["artist"]
            artist_name = artist_obj.get("name", "") if isinstance(artist_obj, dict) else str(artist_obj)

        album_name = ""
        album_id = ""
        if "album" in data:
            album_obj = data["album"]
            if isinstance(album_obj, dict):
                album_name = album_obj.get("title", "")
                album_id = str(album_obj.get("id", ""))
            else:
                album_name = str(album_obj)

        return QobuzTrack(
            id=int(data.get("id", 0)),
            title=data.get("title", ""),
            artist=artist_name,
            album=album_name,
            duration=int(data.get("duration", 0)),
            track_number=int(data.get("track_number", 1)),
            disc_number=int(data.get("media_number", data.get("disc_number", 1))),
            album_id=album_id,
        )
