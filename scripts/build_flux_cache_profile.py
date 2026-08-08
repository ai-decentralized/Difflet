#!/usr/bin/env python3
"""CLI compatibility wrapper for offline FLUX cache-profile construction."""

from difflet.offline.cache_profile.builder import main


if __name__ == "__main__":
    raise SystemExit(main())
