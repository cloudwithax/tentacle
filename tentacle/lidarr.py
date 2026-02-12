"""lidarr api v1 client."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from tentacle.config import LidarrConfig

log = logging.getLogger(__name__)


class LidarrClient:
    """async client for lidarr api v1."""

    def __init__(self, config: LidarrConfig) -> None:
        self.base_url = config.url.rstrip("/")
        self.headers = {"X-Api-Key": config.api_key}
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        client = await self._get_client()
        resp = await client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        client = await self._get_client()
        resp = await client.post(path, json=json)
        resp.raise_for_status()
        return resp.json()

    async def _put(self, path: str, json: Any = None) -> Any:
        client = await self._get_client()
        resp = await client.put(path, json=json)
        resp.raise_for_status()
        return resp.json()

    async def get_wanted_missing(
        self, page: int = 1, page_size: int = 50
    ) -> dict[str, Any]:
        """fetch wanted/missing albums."""
        log.info(
            "fetching wanted/missing", extra={"page": page, "page_size": page_size}
        )
        return await self._get(
            "/api/v1/wanted/missing",
            params={
                "page": page,
                "pageSize": page_size,
                "sortKey": "albums.title",
                "sortDirection": "ascending",
            },
        )

    async def get_all_wanted_missing(self) -> list[dict[str, Any]]:
        """fetch all wanted/missing records across pages."""
        all_records: list[dict[str, Any]] = []
        page = 1
        while True:
            data = await self.get_wanted_missing(page=page, page_size=100)
            records = data.get("records", [])
            all_records.extend(records)
            total = data.get("totalRecords", 0)
            if len(all_records) >= total or not records:
                break
            page += 1
        log.info("fetched all wanted/missing", extra={"count": len(all_records)})
        return all_records

    async def get_album(self, album_id: int) -> dict[str, Any]:
        """get album details by id."""
        return await self._get(f"/api/v1/album/{album_id}")

    async def get_tracks(self, album_id: int) -> list[dict[str, Any]]:
        """get tracks for an album."""
        return await self._get("/api/v1/track", params={"albumId": album_id})

    async def get_artist(self, artist_id: int) -> dict[str, Any]:
        """get artist info by id."""
        return await self._get(f"/api/v1/artist/{artist_id}")

    async def rescan_artist(
        self, artist_id: int, artist_path: str | None = None
    ) -> dict[str, Any]:
        """trigger a rescan for an artist via RescanFolders command."""
        log.info(
            "triggering rescan", extra={"artist_id": artist_id, "path": artist_path}
        )
        payload: dict[str, Any] = {
            "name": "RescanFolders",
            "artistIds": [artist_id],
            "filter": "Matched",
            "addNewArtists": False,
        }
        if artist_path:
            payload["folders"] = [artist_path]
        return await self._post("/api/v1/command", json=payload)

    async def rename_artist_files(self, artist_id: int) -> dict[str, Any]:
        """trigger rename for an artist's files."""
        log.info("triggering rename", extra={"artist_id": artist_id})
        return await self._post(
            "/api/v1/command", json={"name": "RenameArtist", "artistIds": [artist_id]}
        )

    async def update_track_file(self, track_file: dict[str, Any]) -> dict[str, Any]:
        """update track file info."""
        return await self._put("/api/v1/trackfile", json=track_file)
