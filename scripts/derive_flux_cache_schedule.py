#!/usr/bin/env python3
"""CLI compatibility wrapper for deterministic cache-schedule derivation."""

from difflet.offline.cache_profile.derivation import main


if __name__ == "__main__":
    raise SystemExit(main())
