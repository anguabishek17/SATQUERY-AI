"""
Geocoding API Endpoint using OpenStreetMap Nominatim.
Provides geographic place lookup -> (latitude, longitude, bounding box).
"""

from fastapi import APIRouter, HTTPException, Query
import requests

router = APIRouter(prefix="/api/geocode", tags=["Geocoding"])

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "SatQuery-AI/2.0 (satquery-ai-geocoder)"}


@router.get("")
def geocode_location(query: str = Query(..., description="Place name to search")):
    """Searches Nominatim for location query and returns lat, lon, boundingbox, display_name."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        params = {
            "q": query,
            "format": "json",
            "limit": 5,
        }
        res = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=8)
        if res.status_code != 200:
            raise HTTPException(status_code=502, detail="Geocoding service error.")

        data = res.json()
        if not data:
            return {"success": False, "query": query, "results": []}

        results = []
        for item in data:
            lat = float(item["lat"])
            lon = float(item["lon"])
            bbox_str = item.get("boundingbox", [])
            bbox = [float(b) for b in bbox_str] if bbox_str else [lat-0.05, lat+0.05, lon-0.05, lon+0.05]

            results.append({
                "display_name": item.get("display_name", query),
                "lat": lat,
                "lon": lon,
                "boundingbox": bbox,
            })

        return {
            "success": True,
            "query": query,
            "results": results,
            "primary": results[0],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geocoding failed: {e}")
