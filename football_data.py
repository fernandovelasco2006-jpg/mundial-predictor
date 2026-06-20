"""
Módulo de actualización automática — football-data.org
=======================================================
Actualiza resultados del Mundial 2026 automáticamente.
Plan gratuito: 10 requests/minuto, sin límite mensual.
Cache de 30 minutos para no gastar requests innecesariamente.
"""

import requests
import streamlit as st
from datetime import datetime, timezone, timedelta

BASE_URL = "https://api.football-data.org/v4"
WC_2026_ID = 2000  # ID del Mundial 2026 en football-data.org

# Mapa de nombres de equipos API → nombres en nuestra app
NOMBRES_API = {
    # Exactos de la API
    "Mexico":                    "Mexico",
    "South Africa":              "Sudafrica",
    "Korea Republic":            "Corea del Sur",
    "Czechia":                   "Chequia",
    "Czech Republic":            "Chequia",
    "Canada":                    "Canada",
    "Bosnia and Herzegovina":    "Bosnia y Herzegovina",
    "Qatar":                     "Catar",
    "Switzerland":               "Suiza",
    "Brazil":                    "Brasil",
    "Morocco":                   "Marruecos",
    "Haiti":                     "Haiti",
    "Scotland":                  "Escocia",
    "United States":             "Estados Unidos",
    "USA":                       "Estados Unidos",
    "Paraguay":                  "Paraguay",
    "Australia":                 "Australia",
    "Turkey":                    "Turquia",
    "Türkiye":                   "Turquia",
    "Germany":                   "Alemania",
    "Curaçao":                   "Curazao",
    "Curacao":                   "Curazao",
    "Côte d'Ivoire":             "Costa de Marfil",
    "Ivory Coast":               "Costa de Marfil",
    "Ecuador":                   "Ecuador",
    "Netherlands":               "Paises Bajos",
    "Japan":                     "Japon",
    "Sweden":                    "Suecia",
    "Tunisia":                   "Tunez",
    "Belgium":                   "Belgica",
    "Egypt":                     "Egipto",
    "Iran":                      "Iran",
    "New Zealand":               "Nueva Zelanda",
    "Spain":                     "Espana",
    "Cape Verde":                "Cabo Verde",
    "Saudi Arabia":              "Arabia Saudi",
    "Uruguay":                   "Uruguay",
    "France":                    "Francia",
    "Senegal":                   "Senegal",
    "Iraq":                      "Irak",
    "Norway":                    "Noruega",
    "Argentina":                 "Argentina",
    "Algeria":                   "Algeria",
    "Austria":                   "Austria",
    "Jordan":                    "Jordania",
    "Portugal":                  "Portugal",
    "DR Congo":                  "RD Congo",
    "Uzbekistan":                "Uzbekistan",
    "Colombia":                  "Colombia",
    "England":                   "Inglaterra",
    "Croatia":                   "Croacia",
    "Ghana":                     "Ghana",
    "Panama":                    "Panama",
}

def _normalizar(nombre: str) -> str:
    return NOMBRES_API.get(nombre, nombre)

def _headers(token: str) -> dict:
    return {"X-Auth-Token": token}


@st.cache_data(ttl=1800)  # cache 30 minutos
def obtener_resultados_mundial(token: str) -> list:
    """
    Obtiene todos los partidos terminados del Mundial 2026.
    Retorna lista de dicts: {ea, eb, goles_a, goles_b, amarillas, rojas}
    """
    try:
        # Primero buscar el ID correcto del Mundial 2026
        r = requests.get(
            f"{BASE_URL}/competitions",
            headers=_headers(token),
            timeout=10
        )
        if r.status_code == 200:
            for comp in r.json().get("competitions", []):
                if "world cup" in comp.get("name", "").lower() and "2026" in str(comp.get("currentSeason", {}).get("startDate", "")):
                    wc_id = comp["id"]
                    break
            else:
                wc_id = WC_2026_ID
        else:
            wc_id = WC_2026_ID

        # Obtener todos los partidos del torneo
        r2 = requests.get(
            f"{BASE_URL}/competitions/{wc_id}/matches",
            headers=_headers(token),
            timeout=10
        )

        if r2.status_code != 200:
            return []

        partidos = []
        for match in r2.json().get("matches", []):
            if match.get("status") != "FINISHED":
                continue

            home = _normalizar(match["homeTeam"]["name"])
            away = _normalizar(match["awayTeam"]["name"])
            score = match.get("score", {}).get("fullTime", {})
            goles_h = score.get("home")
            goles_a = score.get("away")

            if goles_h is None or goles_a is None:
                continue

            # Tarjetas si están disponibles
            amarillas = rojas = 0
            for booking in match.get("bookings", []):
                if booking.get("card") == "YELLOW":
                    amarillas += 1
                elif booking.get("card") in ("RED", "YELLOW_RED"):
                    rojas += 1

            partidos.append({
                "ea":        home,
                "eb":        away,
                "goles_a":   int(goles_h),
                "goles_b":   int(goles_a),
                "amarillas": amarillas,
                "rojas":     rojas,
                "match_id":  match.get("id"),
            })

        return partidos

    except Exception:
        return []


def actualizar_fixture_y_forma(token: str, PARTIDOS: list, FORMA_MUNDIAL: dict,
                                TARJETAS_MUNDIAL: dict) -> tuple:
    """
    Sincroniza resultados de la API con el fixture y las estadísticas.
    Retorna (PARTIDOS actualizado, FORMA_MUNDIAL actualizada, TARJETAS_MUNDIAL actualizada,
             n_actualizaciones)
    """
    resultados = obtener_resultados_mundial(token)
    if not resultados:
        return PARTIDOS, FORMA_MUNDIAL, TARJETAS_MUNDIAL, 0

    # Construir mapa de resultados para búsqueda rápida
    mapa = {}
    for r in resultados:
        mapa[(r["ea"], r["eb"])] = r
        mapa[(r["eb"], r["ea"])] = {
            **r,
            "ea": r["eb"], "eb": r["ea"],
            "goles_a": r["goles_b"], "goles_b": r["goles_a"]
        }

    # Actualizar fixture
    nuevos_partidos = []
    actualizaciones = 0
    for p in PARTIDOS:
        ea, eb = p[0], p[1]
        clave = (ea, eb)
        if clave in mapa and p[4] is None:
            res = mapa[clave]
            nuevo_p = (ea, eb, p[2], p[3], (res["goles_a"], res["goles_b"]), p[5])
            nuevos_partidos.append(nuevo_p)
            actualizaciones += 1
        else:
            nuevos_partidos.append(p)

    # Reconstruir FORMA_MUNDIAL desde resultados reales
    nueva_forma = dict(FORMA_MUNDIAL)
    gf_equipo = {}
    gc_equipo = {}
    pj_equipo = {}

    for r in resultados:
        for equipo, gf, gc in [(r["ea"], r["goles_a"], r["goles_b"]),
                                (r["eb"], r["goles_b"], r["goles_a"])]:
            gf_equipo[equipo] = gf_equipo.get(equipo, 0) + gf
            gc_equipo[equipo] = gc_equipo.get(equipo, 0) + gc
            pj_equipo[equipo] = pj_equipo.get(equipo, 0) + 1

    for equipo in pj_equipo:
        nueva_forma[equipo] = (gf_equipo[equipo], gc_equipo[equipo], pj_equipo[equipo])

    # Reconstruir TARJETAS_MUNDIAL desde resultados reales
    nuevas_tarjetas = dict(TARJETAS_MUNDIAL)
    am_equipo = {}
    ro_equipo = {}

    for r in resultados:
        # Las tarjetas del API son totales del partido, dividir aprox entre equipos
        am_total = r["amarillas"]
        ro_total = r["rojas"]
        for equipo in [r["ea"], r["eb"]]:
            am_equipo[equipo] = am_equipo.get(equipo, 0) + am_total / 2
            ro_equipo[equipo] = ro_equipo.get(equipo, 0) + ro_total / 2

    for equipo in pj_equipo:
        if equipo in am_equipo:
            nuevas_tarjetas[equipo] = (
                round(am_equipo[equipo]),
                round(ro_equipo.get(equipo, 0)),
                pj_equipo[equipo]
            )

    return nuevos_partidos, nueva_forma, nuevas_tarjetas, actualizaciones
