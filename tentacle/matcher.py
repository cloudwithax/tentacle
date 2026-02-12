"""track matching between lidarr wants and tidal/qobuz search results."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from thefuzz import fuzz

from tentacle.qobuz import QobuzClient, QobuzTrack
from tentacle.tidal import TidalClient, TidalTrack

log = logging.getLogger(__name__)

# patterns to strip from titles for normalization
STRIP_PATTERNS = [
    r"\s*\(feat\.?[^)]*\)",
    r"\s*\[feat\.?[^\]]*\]",
    r"\s*ft\.?\s+.*$",
    r"\s*\(with\s+[^)]*\)",
    r"\s*\[with\s+[^\]]*\]",
    r"\s*\(bonus\s*track\)",
    r"\s*\[bonus\s*track\]",
    r"\s*\(deluxe[^)]*\)",
    r"\s*\[deluxe[^\]]*\]",
    r"\s*\(remaster(ed)?\s*\d*\)",
    r"\s*\[remaster(ed)?\s*\d*\]",
    r"\s*-\s*remaster(ed)?\s*\d*$",
]


@dataclass
class UnifiedTrack:
    """source-agnostic track for matching."""

    id: int
    title: str
    artist: str
    album: str
    duration: int
    track_number: int
    disc_number: int
    album_id: str | int
    source: str  # "tidal" or "qobuz"
    explicit: bool = False


def from_tidal(t: TidalTrack) -> UnifiedTrack:
    return UnifiedTrack(
        id=t.id,
        title=t.title,
        artist=t.artist,
        album=t.album,
        duration=t.duration,
        track_number=t.track_number,
        disc_number=t.disc_number,
        album_id=t.album_id,
        source="tidal",
        explicit=t.explicit,
    )


def from_qobuz(t: QobuzTrack) -> UnifiedTrack:
    return UnifiedTrack(
        id=t.id,
        title=t.title,
        artist=t.artist,
        album=t.album,
        duration=t.duration,
        track_number=t.track_number,
        disc_number=t.disc_number,
        album_id=t.album_id,
        source="qobuz",
        explicit=t.explicit,
    )


@dataclass
class MatchResult:
    """result of matching a lidarr track to a source track."""

    track: UnifiedTrack
    confidence: float
    match_reasons: list[str]


def normalize(s: str) -> str:
    """normalize a string for comparison."""
    s = s.lower().strip()
    for pattern in STRIP_PATTERNS:
        s = re.sub(pattern, "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_similarity(a: str, b: str) -> float:
    """compute title similarity using multiple fuzzy methods."""
    na, nb = normalize(a), normalize(b)

    if na == nb:
        return 1.0

    ratio = fuzz.ratio(na, nb) / 100.0
    partial = fuzz.partial_ratio(na, nb) / 100.0
    token_sort = fuzz.token_sort_ratio(na, nb) / 100.0

    return max(ratio, partial * 0.95, token_sort * 0.95)


def artist_similarity(a: str, b: str) -> float:
    """compute artist similarity."""
    na, nb = normalize(a), normalize(b)

    if na == nb:
        return 1.0

    # handle "artist1 & artist2" vs "artist1, artist2"
    a_parts = set(re.split(r"[,&/]", na))
    b_parts = set(re.split(r"[,&/]", nb))
    a_parts = {p.strip() for p in a_parts if p.strip()}
    b_parts = {p.strip() for p in b_parts if p.strip()}

    if a_parts & b_parts:
        overlap = len(a_parts & b_parts) / max(len(a_parts), len(b_parts))
        return max(overlap, 0.8)

    return fuzz.token_sort_ratio(na, nb) / 100.0


def duration_similarity(a: int, b: int) -> float:
    """compare durations in seconds. returns 1.0 for exact, drops off for differences."""
    if a == 0 or b == 0:
        return 0.5  # can't compare, neutral
    diff = abs(a - b)
    if diff <= 2:
        return 1.0
    if diff <= 5:
        return 0.9
    if diff <= 10:
        return 0.7
    if diff <= 30:
        return 0.4
    return 0.1


def score_match(
    wanted_title: str,
    wanted_artist: str,
    wanted_duration: int,
    candidate: UnifiedTrack,
    prefer_explicit: bool = True,
    source_priority: list[str] | None = None,
) -> MatchResult:
    """score how well a candidate track matches a wanted track."""
    reasons: list[str] = []

    t_sim = title_similarity(wanted_title, candidate.title)
    a_sim = artist_similarity(wanted_artist, candidate.artist)
    d_sim = duration_similarity(wanted_duration, candidate.duration)

    reasons.append(f"title={t_sim:.2f}")
    reasons.append(f"artist={a_sim:.2f}")
    reasons.append(f"duration={d_sim:.2f}")

    # weighted combination
    confidence = (t_sim * 0.45) + (a_sim * 0.35) + (d_sim * 0.20)

    # penalties
    if t_sim < 0.5:
        confidence *= 0.6
        reasons.append("low_title_match")
    if a_sim < 0.4:
        confidence *= 0.5
        reasons.append("low_artist_match")

    # boost explicit versions when preferred
    if prefer_explicit and candidate.explicit:
        confidence = min(confidence * 1.05, 1.0)
        reasons.append("explicit_boost")

    # small source priority tiebreaker (up to +0.02 for primary source)
    if source_priority:
        try:
            idx = source_priority.index(candidate.source)
            # primary source gets +0.02, secondary +0.01, etc.
            bonus = max(0.0, 0.02 - idx * 0.01)
            confidence = min(confidence + bonus, 1.0)
            if idx == 0:
                reasons.append(f"primary_source={candidate.source}")
        except ValueError:
            pass

    return MatchResult(
        track=candidate,
        confidence=confidence,
        match_reasons=reasons,
    )


async def find_best_match(
    tidal: TidalClient,
    artist_name: str,
    track_title: str,
    album_title: str,
    duration: int = 0,
    min_confidence: float = 0.80,
    qobuz: QobuzClient | None = None,
    prefer_explicit: bool = True,
    source_priority: list[str] | None = None,
) -> MatchResult | None:
    """search tidal (and optionally qobuz) and find the best matching track."""
    if source_priority is None:
        source_priority = ["tidal", "qobuz"]

    queries = [
        f"{artist_name} {track_title}",
        f"{track_title} {artist_name}",
    ]

    all_candidates: list[UnifiedTrack] = []

    # search sources in priority order
    sources: list[tuple[str, Any]] = []
    for src in source_priority:
        if src == "tidal":
            sources.append(("tidal", tidal))
        elif src == "qobuz" and qobuz is not None:
            sources.append(("qobuz", qobuz))

    for src_name, client in sources:
        for query in queries:
            try:
                results = await client.search_tracks(query)
                for r in results[:10]:
                    try:
                        if src_name == "tidal":
                            track = TidalClient.parse_tidal_track(r)
                            if track.id > 0:
                                all_candidates.append(from_tidal(track))
                        else:
                            track = QobuzClient.parse_qobuz_track(r)
                            if track.id > 0:
                                all_candidates.append(from_qobuz(track))
                    except (KeyError, TypeError, ValueError):
                        continue
            except Exception as e:
                log.warning(
                    f"{src_name} search failed", extra={"query": query, "error": str(e)}
                )
                continue

    if not all_candidates:
        log.info(
            "no candidates found", extra={"artist": artist_name, "title": track_title}
        )
        return None

    # deduplicate by (source, id)
    seen: set[tuple[str, int]] = set()
    unique: list[UnifiedTrack] = []
    for c in all_candidates:
        key = (c.source, c.id)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # score all candidates
    scored = [
        score_match(
            track_title,
            artist_name,
            duration,
            c,
            prefer_explicit=prefer_explicit,
            source_priority=source_priority,
        )
        for c in unique
    ]
    scored.sort(key=lambda m: m.confidence, reverse=True)

    best = scored[0]

    log.info(
        "best match",
        extra={
            "wanted_title": track_title,
            "wanted_artist": artist_name,
            "matched_title": best.track.title,
            "matched_artist": best.track.artist,
            "matched_source": best.track.source,
            "confidence": f"{best.confidence:.3f}",
            "reasons": ", ".join(best.match_reasons),
        },
    )

    if best.confidence >= min_confidence:
        return best

    log.info(
        "best match below threshold",
        extra={
            "confidence": f"{best.confidence:.3f}",
            "threshold": f"{min_confidence:.2f}",
        },
    )
    return None


async def match_album_tracks(
    tidal: TidalClient,
    artist_name: str,
    album_title: str,
    lidarr_tracks: list[dict[str, Any]],
    min_confidence: float = 0.80,
    qobuz: QobuzClient | None = None,
    prefer_explicit: bool = True,
    source_priority: list[str] | None = None,
) -> dict[int, MatchResult]:
    """try to match all tracks in an album. first tries album search, then per-track search.

    returns dict of lidarr_track_id -> MatchResult.
    """
    if source_priority is None:
        source_priority = ["tidal", "qobuz"]

    matches: dict[int, MatchResult] = {}

    # try album search first
    album_match = await _try_album_match(
        tidal,
        artist_name,
        album_title,
        lidarr_tracks,
        min_confidence,
        qobuz=qobuz,
        prefer_explicit=prefer_explicit,
        source_priority=source_priority,
    )
    if album_match:
        matches.update(album_match)

    # fill in any unmatched tracks with individual searches
    for lt in lidarr_tracks:
        lt_id = lt.get("id", 0)
        if lt_id in matches:
            continue

        title = lt.get("title", "")
        duration = lt.get("duration", 0)
        # lidarr duration is in ms sometimes, convert
        if duration > 10000:
            duration = duration // 1000

        result = await find_best_match(
            tidal,
            artist_name,
            title,
            album_title,
            duration=duration,
            min_confidence=min_confidence,
            qobuz=qobuz,
            prefer_explicit=prefer_explicit,
            source_priority=source_priority,
        )
        if result:
            matches[lt_id] = result

    return matches


async def _try_album_match(
    tidal: TidalClient,
    artist_name: str,
    album_title: str,
    lidarr_tracks: list[dict[str, Any]],
    min_confidence: float,
    qobuz: QobuzClient | None = None,
    prefer_explicit: bool = True,
    source_priority: list[str] | None = None,
) -> dict[int, MatchResult] | None:
    """try to find and match a whole album on tidal and/or qobuz.

    searches sources in priority order (default: tidal first, qobuz fallback).
    prefers explicit versions when prefer_explicit is True.
    """
    if source_priority is None:
        source_priority = ["tidal", "qobuz"]

    all_album_tracks: list[UnifiedTrack] = []

    # search each source in priority order
    for src in source_priority:
        if src == "tidal":
            await _search_tidal_album(tidal, artist_name, album_title, all_album_tracks)
        elif src == "qobuz" and qobuz is not None:
            await _search_qobuz_album(qobuz, artist_name, album_title, all_album_tracks)

    if not all_album_tracks:
        return None

    # deduplicate by (source, id)
    seen: set[tuple[str, int]] = set()
    unique_tracks: list[UnifiedTrack] = []
    for c in all_album_tracks:
        key = (c.source, c.id)
        if key not in seen:
            seen.add(key)
            unique_tracks.append(c)

    # match lidarr tracks to source tracks
    matches: dict[int, MatchResult] = {}
    used: set[tuple[str, int]] = set()

    for lt in lidarr_tracks:
        lt_id = lt.get("id", 0)
        lt_title = lt.get("title", "")
        lt_duration = lt.get("duration", 0)
        if lt_duration > 10000:
            lt_duration = lt_duration // 1000

        best_match: MatchResult | None = None
        for tt in unique_tracks:
            if (tt.source, tt.id) in used:
                continue
            result = score_match(
                lt_title,
                artist_name,
                lt_duration,
                tt,
                prefer_explicit=prefer_explicit,
                source_priority=source_priority,
            )
            if best_match is None or result.confidence > best_match.confidence:
                best_match = result

        if best_match and best_match.confidence >= min_confidence:
            matches[lt_id] = best_match
            used.add((best_match.track.source, best_match.track.id))

    return matches if matches else None


async def _search_tidal_album(
    tidal: TidalClient,
    artist_name: str,
    album_title: str,
    out: list[UnifiedTrack],
) -> None:
    """search tidal for an album and append parsed tracks to out."""
    try:
        results = await tidal.search_albums(f"{artist_name} {album_title}")
    except Exception as e:
        log.warning("tidal album search failed", extra={"error": str(e)})
        return

    best_album: dict[str, Any] | None = None
    best_score = 0.0

    for album in (results or [])[:5]:
        a_title = album.get("title", "")
        a_artist = ""
        if "artist" in album:
            art = album["artist"]
            a_artist = art.get("name", "") if isinstance(art, dict) else str(art)
        elif "artists" in album and album["artists"]:
            first = album["artists"][0]
            a_artist = first.get("name", "") if isinstance(first, dict) else str(first)

        t_sim = title_similarity(album_title, a_title)
        a_sim = artist_similarity(artist_name, a_artist)
        score = t_sim * 0.6 + a_sim * 0.4

        if score > best_score:
            best_score = score
            best_album = album

    if best_album and best_score >= 0.7:
        album_id = best_album.get("id")
        if album_id:
            log.info(
                "found tidal album match",
                extra={
                    "album": best_album.get("title", ""),
                    "score": f"{best_score:.3f}",
                },
            )
            try:
                album_data = await tidal.get_album(int(album_id))
                tidal_tracks_raw = album_data.get("tracks", album_data.get("items", []))
                for t in tidal_tracks_raw:
                    try:
                        out.append(from_tidal(TidalClient.parse_tidal_track(t)))
                    except (KeyError, TypeError, ValueError):
                        continue
            except Exception as e:
                log.warning(
                    "failed to fetch tidal album",
                    extra={"album_id": album_id, "error": str(e)},
                )


async def _search_qobuz_album(
    qobuz: QobuzClient,
    artist_name: str,
    album_title: str,
    out: list[UnifiedTrack],
) -> None:
    """search qobuz for an album and append parsed tracks to out."""
    try:
        qobuz_results = await qobuz.search_albums(f"{artist_name} {album_title}")
    except Exception as e:
        log.warning("qobuz album search failed", extra={"error": str(e)})
        return

    best_q_album: dict[str, Any] | None = None
    best_q_score = 0.0

    for album in (qobuz_results or [])[:5]:
        a_title = album.get("title", "")
        a_artist = ""
        if "artist" in album:
            art = album["artist"]
            a_artist = art.get("name", "") if isinstance(art, dict) else str(art)
        elif "artists" in album and album["artists"]:
            first = album["artists"][0]
            a_artist = first.get("name", "") if isinstance(first, dict) else str(first)

        t_sim = title_similarity(album_title, a_title)
        a_sim = artist_similarity(artist_name, a_artist)
        score = t_sim * 0.6 + a_sim * 0.4

        if score > best_q_score:
            best_q_score = score
            best_q_album = album

    if best_q_album and best_q_score >= 0.7:
        q_album_id = best_q_album.get("id")
        if q_album_id:
            log.info(
                "found qobuz album match",
                extra={
                    "album": best_q_album.get("title", ""),
                    "score": f"{best_q_score:.3f}",
                },
            )
            try:
                album_data = await qobuz.get_album(str(q_album_id))
                qobuz_tracks_raw = album_data.get("tracks", {})
                if isinstance(qobuz_tracks_raw, dict):
                    qobuz_tracks_raw = qobuz_tracks_raw.get("items", [])
                for t in qobuz_tracks_raw:
                    try:
                        out.append(from_qobuz(QobuzClient.parse_qobuz_track(t)))
                    except (KeyError, TypeError, ValueError):
                        continue
            except Exception as e:
                log.warning(
                    "failed to fetch qobuz album",
                    extra={"album_id": q_album_id, "error": str(e)},
                )
