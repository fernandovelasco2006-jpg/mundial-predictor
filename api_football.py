"""
Módulo de integración con Free API Live Football Data (RapidAPI)
================================================================
Conecta la app con datos en tiempo real del Mundial 2026:
- Resultados automáticos de partidos ya jugados
- Estadísticas de córners por equipo
- Forma reciente actualizada automáticamente
- Livescores en tiempo real

Uso:
    from api_football import actualizar_resultados, obtener_corners
"""

import requests
import streamlit as st
from datetime import datetime, timedelta
import json

# ── Configuración ─────────────────────────────────────────────────────────────
RAPIDAPI_HOST = "free-api-live-football-data.p.rapidapi.com"
BASE_URL = f"https://{RAPIDAPI_HOST}"

# ID del Mundial 2026 en esta API (lo detectamos automáticamente)
WORLD_CUP_LEAGUE_ID = None  # se busca al iniciar

# Nombres de equipos en la API vs nombres en nuestra app
# (la API usa nombres en inglés/estándar)
NOMBRES_API = {
    "Mexico": ["Mexico", "México"],
    "Sudafrica": ["South Africa"],
    "Corea del Sur": ["Korea Republic", "South Korea"],
    "Chequia": ["Czech Republic", "Czechia"],
    "Canada": ["Canada", "Canadá"],
    "Bosnia y Herzegovina": ["Bosnia & Herzegovina", "Bosnia and Herzegovina"],
    "Catar": ["Qatar"],
    "Suiza": ["Switzerland"],
    "Brasil": ["Brazil"],
    "Marruecos": ["Morocco"],
    "Haiti": ["Haiti"],
    "Escocia": ["Scotland"],
    "Estados Unidos": ["United States", "USA"],
    "Paraguay": ["Paraguay"],
    "Australia": ["Australia"],
    "Turquia": ["Turkey", "Türkiye"],
    "Alemania": ["Germany"],
    "Curazao": ["Curaçao", "Curacao"],
    "Costa de Marfil": ["Ivory Coast", "Côte d'Ivoire"],
    "Ecuador": ["Ecuador"],
    "Paises Bajos": ["Netherlands"],
    "Japon": ["Japan"],
    "Suecia": ["Sweden"],
    "Tunez": ["Tunisia"],
    "Belgica": ["Belgium"],
    "Egipto": ["Egypt"],
    "Iran": ["IR Iran", "Iran"],
    "Nueva Zelanda": ["New Zealand"],
    "Espana": ["Spain"],
    "Cabo Verde": ["Cape Verde"],
    "Arabia Saudi": ["Saudi Arabia"],
    "Arabia Saudita": ["Saudi Arabia"],
    "Uruguay": ["Uruguay"],
    "Francia": ["France"],
    "Senegal": ["Senegal"],
    "Irak": ["Iraq"],
    "Noruega": ["Norway"],
    "Argentina": ["Argentina"],
    "Algeria": ["Algeria"],
    "Argelia": ["Algeria"],
    "Austria": ["Austria"],
    "Jordania": ["Jordan"],
    "Portugal": ["Portugal"],
    "RD Congo": ["DR Congo", "Congo DR"],
    "Uzbekistan": ["Uzbekistan"],
    "Colombia": ["Colombia"],
    "Inglaterra": ["England"],
    "Croacia": ["Croatia"],
    "Ghana": ["Ghana"],
    "Panama": ["Panama"],
}

# Invertir el mapa para buscar rápido
API_A_APP = {}
for nombre_app, nombres_api in NOMBRES_API.items():
    for n in nombres_api:
        API_A_APP[n.lower()] = nombre_app


def _headers(api_key: str) -> dict:
    return {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": RAPIDAPI_HOST,
        "Content-Type": "application/json"
    }


def _get(endpoint: str, params: dict, api_key: str) -> dict | None:
    """Llamada GET con manejo de errores y cache de Streamlit."""
    try:
        r = requests.get(
            f"{BASE_URL}/{endpoint}",
            headers=_headers(api_key),
            params=params,
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            return None  # límite de requests alcanzado — modo silencioso
        else:
            return None  # cualquier error → fallback silencioso
    except Exception as e:
        st.warning(f"Error conectando a la API: {e}")
        return None


@st.cache_data(ttl=300)  # cache 5 minutos
def buscar_liga_mundial(api_key: str) -> int | None:
    """Busca el ID de la liga del Mundial 2026."""
    data = _get("football-get-all-leagues", {}, api_key)
    if not data:
        return None

    respuesta = data.get("response", data) if isinstance(data, dict) else data
    if isinstance(respuesta, list):
        for liga in respuesta:
            nombre = str(liga.get("name", "")).lower()
            if "world cup" in nombre or "mundial" in nombre or "fifa" in nombre:
                league_id = liga.get("id") or liga.get("league_id")
                if league_id:
                    return int(league_id)
    return None


@st.cache_data(ttl=300)  # cache 5 minutos — refresca cada 5 min
def obtener_fixtures_mundial(api_key: str, league_id: int = None) -> list:
    """
    Obtiene todos los fixtures del Mundial 2026 con resultados.
    Retorna lista de (equipo_a, equipo_b, goles_a, goles_b, estado)
    """
    if not league_id:
        league_id = buscar_liga_mundial(api_key)
    if not league_id:
        return []

    data = _get(
        "football-get-fixtures-by-league",
        {"leagueId": league_id, "season": 2026},
        api_key
    )
    if not data:
        return []

    fixtures = []
    respuesta = data.get("response", []) if isinstance(data, dict) else []

    for f in respuesta:
        try:
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            status = f.get("fixture", {}).get("status", {}).get("short", "")

            home_name = teams.get("home", {}).get("name", "")
            away_name = teams.get("away", {}).get("name", "")

            home_app = API_A_APP.get(home_name.lower(), home_name)
            away_app = API_A_APP.get(away_name.lower(), away_name)

            goles_home = goals.get("home")
            goles_away = goals.get("away")

            fixtures.append({
                "equipo_a": home_app,
                "equipo_b": away_app,
                "goles_a": goles_home,
                "goles_b": goles_away,
                "estado": status,  # FT=terminado, NS=por jugar, LIVE=en vivo
                "fixture_id": f.get("fixture", {}).get("id")
            })
        except Exception:
            continue

    return fixtures


@st.cache_data(ttl=600)  # cache 10 minutos
def obtener_estadisticas_fixture(fixture_id: int, api_key: str) -> dict:
    """
    Obtiene estadísticas de un partido específico:
    córners, tarjetas, posesión, tiros a puerta.
    """
    data = _get(
        "football-get-fixture-by-id",
        {"fixtureId": fixture_id},
        api_key
    )
    if not data:
        return {}

    stats = {}
    respuesta = data.get("response", [{}])[0] if isinstance(data, dict) else {}
    estadisticas = respuesta.get("statistics", [])

    for equipo_stats in estadisticas:
        nombre = equipo_stats.get("team", {}).get("name", "")
        nombre_app = API_A_APP.get(nombre.lower(), nombre)
        stats[nombre_app] = {}

        for stat in equipo_stats.get("statistics", []):
            tipo = stat.get("type", "").lower()
            valor = stat.get("value", 0) or 0
            if "corner" in tipo:
                stats[nombre_app]["corners"] = int(valor)
            elif "yellow" in tipo:
                stats[nombre_app]["amarillas"] = int(valor)
            elif "red" in tipo:
                stats[nombre_app]["rojas"] = int(valor)
            elif "shots on" in tipo:
                stats[nombre_app]["tiros_puerta"] = int(valor)
            elif "possession" in tipo:
                stats[nombre_app]["posesion"] = str(valor)

    return stats


@st.cache_data(ttl=1800)  # cache 30 minutos
def calcular_corners_promedio(equipo: str, api_key: str, league_id: int = None) -> float:
    """
    Calcula el promedio de córners del equipo en el Mundial 2026
    basado en los partidos ya jugados.
    Devuelve el promedio o 5.0 como default si no hay datos.
    """
    fixtures = obtener_fixtures_mundial(api_key, league_id)
    corners_total = 0
    partidos = 0

    for f in fixtures:
        if f["estado"] != "FT":
            continue
        if equipo not in (f["equipo_a"], f["equipo_b"]):
            continue

        stats = obtener_estadisticas_fixture(f["fixture_id"], api_key)
        if equipo in stats and "corners" in stats[equipo]:
            corners_total += stats[equipo]["corners"]
            partidos += 1

    return corners_total / partidos if partidos > 0 else 5.0


def sincronizar_resultados(api_key: str, partidos_app: list) -> dict:
    """
    Compara los fixtures de la API con los partidos en nuestra app
    y devuelve un dict de actualizaciones: {(ea, eb): (goles_a, goles_b)}
    """
    fixtures_api = obtener_fixtures_mundial(api_key)
    actualizaciones = {}

    for f in fixtures_api:
        if f["estado"] != "FT":
            continue
        if f["goles_a"] is None or f["goles_b"] is None:
            continue

        ea, eb = f["equipo_a"], f["equipo_b"]
        actualizaciones[(ea, eb)] = (f["goles_a"], f["goles_b"])
        # También en orden inverso por si acaso
        actualizaciones[(eb, ea)] = (f["goles_b"], f["goles_a"])

    return actualizaciones


def mostrar_status_api(api_key: str):
    """Muestra un badge en la UI indicando si la API está conectada."""
    try:
        r = requests.get(
            f"{BASE_URL}/football-get-all-leagues",
            headers=_headers(api_key),
            timeout=5
        )
        if r.status_code == 200:
            st.success("🟢 API conectada — datos en tiempo real activos")
            return True
        elif r.status_code == 429:
            st.info("⚪ Límite mensual de API alcanzado — usando datos manuales (se renueva el 1ro del mes)")
            return False
        else:
            st.info(f"⚪ API en modo offline — usando datos manuales")
            return False
    except Exception:
        st.info("⚪ Modo offline — usando datos manuales")
        return False
