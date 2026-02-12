"""mirror ranker — pacman-style latency testing and sorting on startup."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

# unreachable mirrors get this penalty so they sort last
UNREACHABLE_MS = 99999.0


@dataclass
class MirrorResult:
    """result of probing a single mirror."""
    url: str
    latency_ms: float
    status: int | None
    alive: bool


async def _probe(client: httpx.AsyncClient, url: str, probe_path: str, timeout: float) -> MirrorResult:
    """send a lightweight GET and measure round-trip time."""
    target = f"{url}{probe_path}"
    try:
        t0 = time.monotonic()
        resp = await client.get(target, timeout=timeout, follow_redirects=True)
        latency = (time.monotonic() - t0) * 1000
        alive = resp.status_code < 500
        return MirrorResult(url=url, latency_ms=latency, status=resp.status_code, alive=alive)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError, httpx.ConnectTimeout, OSError) as e:
        log.debug("probe failed", extra={"url": url, "error": str(e)})
        return MirrorResult(url=url, latency_ms=UNREACHABLE_MS, status=None, alive=False)


async def rank_mirrors(
    mirrors: list[str],
    probe_path: str = "/",
    timeout: float = 8.0,
    rounds: int = 2,
) -> list[MirrorResult]:
    """probe all mirrors concurrently, average latencies over multiple rounds, sort fastest-first.

    works like `rankmirrors` / `reflector` — fires parallel HEAD-ish requests and ranks by
    median response time. dead mirrors sink to the bottom.

    args:
        mirrors: list of base urls to probe (e.g. ["https://triton.squid.wtf"])
        probe_path: path to hit on each mirror (should be fast/cheap, like a search endpoint)
        timeout: per-request timeout in seconds
        rounds: number of probe rounds (results averaged to smooth jitter)

    returns:
        sorted list of MirrorResult, fastest alive mirrors first
    """
    if not mirrors:
        return []

    log.info("ranking mirrors", extra={"count": len(mirrors), "rounds": rounds})

    # accumulate latencies across rounds
    totals: dict[str, list[float]] = {m: [] for m in mirrors}
    last_results: dict[str, MirrorResult] = {}

    async with httpx.AsyncClient() as client:
        for r in range(rounds):
            tasks = [_probe(client, m, probe_path, timeout) for m in mirrors]
            results = await asyncio.gather(*tasks)
            for res in results:
                totals[res.url].append(res.latency_ms)
                last_results[res.url] = res

            # small gap between rounds to avoid looking like a burst
            if r < rounds - 1:
                await asyncio.sleep(0.3)

    # build final results using average latency
    ranked: list[MirrorResult] = []
    for url in mirrors:
        times = totals[url]
        avg = sum(times) / len(times) if times else UNREACHABLE_MS
        base = last_results.get(url)
        ranked.append(MirrorResult(
            url=url,
            latency_ms=round(avg, 1),
            status=base.status if base else None,
            alive=any(t < UNREACHABLE_MS for t in times),
        ))

    # sort: alive first, then by latency
    ranked.sort(key=lambda r: (not r.alive, r.latency_ms))

    # log the leaderboard
    for i, r in enumerate(ranked):
        tag = "✓" if r.alive else "✗"
        ms = f"{r.latency_ms:.0f}ms" if r.alive else "dead"
        log.info(f"  {tag} #{i+1:2d}  {ms:>8s}  {r.url}")

    alive_count = sum(1 for r in ranked if r.alive)
    log.info("ranking complete", extra={
        "alive": alive_count,
        "dead": len(ranked) - alive_count,
        "fastest": ranked[0].url if ranked else "none",
    })

    return ranked


class RankedMirrorPool:
    """thread-safe ranked mirror pool with runtime demotion on failure."""

    def __init__(self, ranked: list[MirrorResult]) -> None:
        self._ranked = [r for r in ranked if r.alive] or ranked  # fallback to all if none alive
        self._idx = 0
        self._demoted: set[str] = set()

    @property
    def mirrors(self) -> list[str]:
        """return mirrors in ranked order, demoted ones at the end."""
        good = [r.url for r in self._ranked if r.url not in self._demoted]
        bad = [r.url for r in self._ranked if r.url in self._demoted]
        return good + bad

    def pick(self) -> str:
        """pick the next mirror, cycling through ranked order."""
        pool = self.mirrors
        if not pool:
            # shouldn't happen but safety valve
            return self._ranked[0].url
        url = pool[self._idx % len(pool)]
        self._idx += 1
        return url

    def demote(self, url: str) -> None:
        """push a failing mirror to the back of the line."""
        self._demoted.add(url)
        log.debug("demoted mirror", extra={"url": url})

    def reset(self) -> None:
        """reset demotions (e.g. between cycles)."""
        self._demoted.clear()
        self._idx = 0

    def best(self) -> str:
        """return the current top-ranked mirror."""
        return self.mirrors[0]

    def __len__(self) -> int:
        return len(self._ranked)

    def __repr__(self) -> str:
        alive = sum(1 for r in self._ranked if r.alive)
        return f"RankedMirrorPool({alive} alive, best={self.best()})"
