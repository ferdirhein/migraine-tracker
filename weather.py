'''
What APIs we used to fetch hPa data 
Endpoints that we used:
  - Geocoding: https://geocoding-api.open-meteo.com/v1/search
  - Forecast: https://api.open-meteo.com/v1/forecast
  - Archive: https://archive-api.open-meteo.com/v1/archive
'''

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Optional

# mkae a dataclass for location with information required for fetching the barometric pressure
@dataclass(frozen=True) #make the datapoints immutable as it is assumed to be fixed for each user
class Location:
    '''Resolved location (city + country code + coordinates).'''

    city: str
    country_code: Optional[str]
    latitude: float
    longitude: float


def _http_get_json(url: str, *, timeout_s: float = 10.0) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "health-tracker-migraine-cli/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_location(city: str, country_code: Optional[str] = None) -> Optional[Location]:
    '''
    convert a (city, country_code) into latitude/longitude using Open-Meteo geocoding
    Returns None if no match is found or network error occurs
    '''

    params = {
        "name": city,
        "count": 1,
        "format": "json",
        "language": "en",
    }
    if country_code:
        params["countryCode"] = country_code.upper()

    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(params)
    try:
        payload = _http_get_json(url)
    except Exception:
        return None

    results = payload.get("results") or []
    if not results:
        return None

    best = results[0]
    try:
        lat = float(best["latitude"])
        lon = float(best["longitude"])
    except Exception:
        return None

    resolved_city = str(best.get("name") or city)
    resolved_cc = str(best.get("country_code") or country_code or "").upper() or None
    return Location(city=resolved_city, country_code=resolved_cc, latitude=lat, longitude=lon)


def fetch_barometric_pressure_hpa(
    latitude: float,
    longitude: float,
    day: str,
    *,
    timezone: str = "auto",
) -> Optional[float]:
    '''
    important step: find hPa score for given location and day
    --> Returns daily mean of hourly values
    - Uses "past_days" (up to 92) when day is recent
    - otherwise, try Archive API.
    - also, an issue that occured: use surface_pressure if pressure_msl is missing !
    '''

    # iso format required for the date
    try:
        target = date.fromisoformat(day)
    except ValueError:
        return None

    today = date.today()
    delta_days = (today - target).days

    #calculate  mean of barometric pressure of the day
    def _mean_from_payload(payload: dict, hourly_var: str) -> Optional[float]:
        hourly = payload.get("hourly") or {}
        values = hourly.get(hourly_var)
        if not values:
            return None
        cleaned = []
        for v in values:
            if v is None:
                continue
            try:
                cleaned.append(float(v))
            except Exception:
                continue
        if not cleaned:
            return None
        return sum(cleaned) / len(cleaned)

    # either taking the forecast or archive
    def _query_forecast(hourly_var: str) -> Optional[float]:
        params = {
            "latitude": f"{latitude:.6f}",
            "longitude": f"{longitude:.6f}",
            "hourly": hourly_var,
            "start_date": day,
            "end_date": day,
            "timezone": timezone,
        }

        # forecast API: use `past_days` for recent past (they only keep the past 92 days !)
        if delta_days >= 1:
            params["past_days"] = str(min(92, delta_days))
        elif delta_days < 0:
            # Future date: there are limited but some into the future.
            params["forecast_days"] = str(min(16, abs(delta_days) + 1))

        url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
        try:
            payload = _http_get_json(url)
        except Exception:
            return None
        return _mean_from_payload(payload, hourly_var) #return the calculated mean here

    # take wither pressure msl or (because here we encountered some issues) the surface pressure
    def _query_archive(hourly_var: str) -> Optional[float]:
        params = {
            "latitude": f"{latitude:.6f}",
            "longitude": f"{longitude:.6f}",
            "hourly": hourly_var,
            "start_date": day,
            "end_date": day,
            "timezone": timezone,
        }
        url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
        try:
            payload = _http_get_json(url)
        except Exception:
            return None
        return _mean_from_payload(payload, hourly_var)

    # forecast(for today and also recent past up to 92 days in the past)
    out = _query_forecast("pressure_msl") or _query_forecast("surface_pressure")
    if out is not None:
        return out

    # archive(for older past dates)
    return _query_archive("pressure_msl") or _query_archive("surface_pressure")
