"""Geospatial Utility Module for Geohashes and Urban Zoning.

Provides geohash decoding, Haversine distance, and Core vs Non-Core classification.
Used across Zone Management, ML Engine (Rain Paradox), and Dynamic Heatmaps.
"""

from __future__ import annotations

import math
from typing import Tuple

try:
    import geohash
    HAS_GEOHASH_LIB = True
except ImportError:
    HAS_GEOHASH_LIB = False

# City Center Coordinate (Bangalore Central: Vidhana Soudha / MG Road)
CITY_CENTER_LAT = 12.9716
CITY_CENTER_LON = 77.5946


def decode_geohash(gh: str) -> Tuple[float, float]:
    """Decodes geohash to (lat, lon). Uses pygeohash or custom base32 fallback."""
    if HAS_GEOHASH_LIB:
        try:
            lat, lon = geohash.decode(gh)
            return float(lat), float(lon)
        except Exception:
            pass

    # Simple base32 bounding box fallback
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    is_even = True

    for char in gh.lower():
        cd = base32.find(char)
        if cd == -1:
            break
        for mask in [16, 8, 4, 2, 1]:
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if cd & mask:
                    lon_interval[0] = mid
                else:
                    lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if cd & mask:
                    lat_interval[0] = mid
                else:
                    lat_interval[1] = mid
            is_even = not is_even

    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    return lat, lon


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points in km."""
    lat1, lon1 = float(lat1), float(lon1)
    lat2, lon2 = float(lat2), float(lon2)
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def classify_zone_tier(lat: float, lon: float) -> str:
    """Classifies urban zone based on radius from Central Hub.
    - Core City: < 6.0 km
    - Mid-Ring: 6.0 - 14.0 km
    - Periphery: > 14.0 km
    """
    lat, lon = float(lat), float(lon)
    dist = haversine_km(lat, lon, CITY_CENTER_LAT, CITY_CENTER_LON)
    if dist < 6.0:
        return "Core City"
    elif dist < 14.0:
        return "Mid-Ring"
    return "Periphery"
