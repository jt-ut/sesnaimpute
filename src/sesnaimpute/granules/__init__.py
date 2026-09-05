"""The granule map: the single join between granules, and its accessor.

Stores, once per source, its sightline row, its hpx512 pixel, its tile, and
its region; `granules.access.per_source` is the single code path that maps
any product at any granule to the source level through it.
"""
