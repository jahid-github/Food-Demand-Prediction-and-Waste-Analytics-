from __future__ import annotations

import requests

from .constants import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_TIMEZONE


def get_tomorrow_temperature(
    lat: float = DEFAULT_LATITUDE,
    lon: float = DEFAULT_LONGITUDE,
    timezone: str = DEFAULT_TIMEZONE,
    timeout: int = 10,
) -> float:
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "timezone": timezone,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    daily = payload.get("daily", {})
    temperatures = daily.get("temperature_2m_max", [])
    if len(temperatures) < 2:
        raise ValueError("Open-Meteo response did not include tomorrow's temperature.")

    return float(temperatures[1])
