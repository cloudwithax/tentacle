# tentacle

automated service for downloading wanted lidarr tracks from tidal via public apis.

polls lidarr for wanted/missing albums, fuzzy-matches tracks against tidal (13 hifi-api instances), picks the best match, downloads to the correct artist/album directory structure, embeds metadata, and triggers a lidarr rescan.

## setup

```bash
pip install .
cp config.example.yml config.yml
# edit config.yml with your lidarr details
tentacle --config config.yml
```

### single run (no loop)

```bash
tentacle --config config.yml --once
```

## docker

```bash
cp config.example.yml config.yml
# edit config.yml
docker compose up -d
```

### environment variables

all config values can be overridden via env vars:

| variable | description | default |
|---|---|---|
| `TENTACLE_LIDARR_URL` | lidarr base url | `http://localhost:8686` |
| `TENTACLE_LIDARR_API_KEY` | lidarr api key | (required) |
| `TENTACLE_DOWNLOAD_PATH` | root download path | `/music` |
| `TENTACLE_DOWNLOAD_QUALITY` | audio quality | `LOSSLESS` |
| `TENTACLE_SERVICE_SCAN_INTERVAL_MINUTES` | poll interval | `15` |
| `TENTACLE_SERVICE_MAX_RETRIES` | retry count | `3` |
| `TENTACLE_SERVICE_RATE_LIMIT_DELAY` | delay between requests | `1.0` |
| `TENTACLE_SERVICE_MAX_CONCURRENT_DOWNLOADS` | parallel download workers | `25` |
| `TENTACLE_MATCHING_MIN_CONFIDENCE` | match threshold | `0.80` |

## config reference

```yaml
lidarr:
  url: "http://localhost:8686"
  api_key: "your-api-key-here"

download:
  path: "/music"
  quality: "LOSSLESS"  # HI_RES_LOSSLESS, LOSSLESS, HIGH, LOW

service:
  scan_interval_minutes: 15
  max_retries: 3
  rate_limit_delay: 1.0
  max_concurrent_downloads: 25  # parallel download workers

matching:
  min_confidence: 0.80
```

## how it works

1. polls lidarr's wanted/missing endpoint
2. for each wanted album, fetches track list
3. searches **tidal** (13 hifi-api instances) for matching tracks using fuzzy matching
4. picks the best match by confidence score
5. downloads matched tracks (direct FLAC or DASH manifests)
6. embeds metadata (title, artist, album, track/disc number) via mutagen
7. triggers lidarr rescan + rename
8. tracks attempted downloads in a state file to avoid re-processing

uses weighted random selection across api instances with automatic failover. handles rate limiting with exponential backoff and quality fallback (HI_RES_LOSSLESS → LOSSLESS → HIGH → LOW).

### sources

| source | instances | quality | format |
|--------|-----------|---------|--------|
| tidal | 13 hifi-api instances | up to 24-bit/192kHz | FLAC (direct or DASH) |
