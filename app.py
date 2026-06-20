"""
Módulo de cuotas en tiempo real — OddsPapi (RapidAPI)
======================================================
Obtiene cuotas de Bet365, Pinnacle y otras casas para partidos del Mundial.
Calcula probabilidad implícita y compara con nuestro modelo para detectar valor (EV+).
"""

import requests
import streamlit as st

RAPIDAPI_HOST = "odds-api1.p.rapidapi.com"
BASE_URL = f"https://{RAPIDAPI_HOST}"

# Nombres de equipos en la API (inglés) → nombres en nuestra app
EQUIPOS_ODDS = {
    "mexico": "Mexico", "south africa": "Sudafrica",
    "south korea": "Corea del Sur", "korea republic": "Corea del Sur",
    "czech republic": "Chequia", "czechia": "Chequia",
    "canada": "Canada", "bosnia": "Bosnia y Herzegovina",
    "bosnia and herzegovina": "Bosnia y Herzegovina",
    "qatar": "Catar", "switzerland": "Suiza",
    "brazil": "Brasil", "morocco": "Marruecos",
    "haiti": "Haiti", "scotland": "Escocia",
    "usa": "Estados Unidos", "united states": "Estados Unidos",
    "paraguay": "Paraguay", "australia": "Australia",
    "turkey": "Turquia", "turkiye": "Turquia",
    "germany": "Alemania", "curacao": "Curazao",
    "ivory coast": "Costa de Marfil", "cote d'ivoire": "Costa de Marfil",
    "ecuador": "Ecuador", "netherlands": "Paises Bajos",
    "japan": "Japon", "sweden": "Suecia", "tunisia": "Tunez",
    "belgium": "Belgica", "egypt": "Egipto", "iran": "Iran",
    "new zealand": "Nueva Zelanda", "spain": "Espana",
    "cape verde": "Cabo Verde", "saudi arabia": "Arabia Saudi",
    "uruguay": "Uruguay", "france": "Francia", "senegal": "Senegal",
    "iraq": "Irak", "norway": "Noruega", "argentina": "Argentina",
    "algeria": "Algeria", "austria": "Austria", "jordan": "Jordania",
    "portugal": "Portugal", "dr congo": "RD Congo", "uzbekistan": "Uzbekistan",
    "colombia": "Colombia", "england": "Inglaterra", "croatia": "Croacia",
    "ghana": "Ghana", "panama": "Panama",
}

def _headers(api_key: str) -> dict:
    return {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }

def cuota_a_probabilidad(cuota: float) -> float:
    """Convierte cuota decimal a probabilidad implícita (sin margen)."""
    if cuota <= 0:
        return 0.0
    return round(1 / cuota * 100, 1)

def normalizar_probabilidades(p1: float, px: float, p2: float) -> tuple:
    """Elimina el margen de la casa para obtener probabilidades justas."""
    total = p1 + px + p2
    if total <= 0:
        return p1, px, p2
    factor = 100 / total
    return round(p1 * factor, 1), round(px * factor, 1), round(p2 * factor, 1)

@st.cache_data(ttl=300)  # cache 5 minutos
def obtener_odds_mundial(api_key: str) -> list:
    """
    Obtiene las cuotas de todos los partidos del Mundial 2026.
    Retorna lista de dicts con probabilidades implícitas por partido.
    """
    try:
        # Primero buscar el sport key para soccer/FIFA World Cup
        r_sports = requests.get(
            f"{BASE_URL}/sports",
            headers=_headers(api_key),
            timeout=10
        )
        if r_sports.status_code != 200:
            return []

        sports = r_sports.json()
        if isinstance(sports, dict):
            sports = sports.get("data", sports.get("sports", []))

        # Buscar FIFA World Cup
        wc_key = None
        for s in sports:
            nombre = str(s.get("title", "") + s.get("key", "") + s.get("group", "")).lower()
            if "world cup" in nombre or "fifa" in nombre:
                wc_key = s.get("key") or s.get("sport_key")
                break

        if not wc_key:
            # Intentar con soccer genérico
            for s in sports:
                if "soccer" in str(s).lower():
                    wc_key = s.get("key") or s.get("sport_key")
                    break

        if not wc_key:
            return []

        # Obtener odds del torneo
        r_odds = requests.get(
            f"{BASE_URL}/odds",
            headers=_headers(api_key),
            params={
                "sport": wc_key,
                "regions": "eu",           # Europa tiene Bet365 y Pinnacle
                "markets": "h2h",          # moneyline (1X2)
                "bookmakers": "bet365,pinnacle",
                "oddsFormat": "decimal",
            },
            timeout=10
        )

        if r_odds.status_code != 200:
            return []

        data = r_odds.json()
        eventos = data if isinstance(data, list) else data.get("data", [])

        partidos_odds = []
        for evento in eventos:
            home = str(evento.get("home_team", "")).lower()
            away = str(evento.get("away_team", "")).lower()

            home_app = EQUIPOS_ODDS.get(home, home.title())
            away_app = EQUIPOS_ODDS.get(away, away.title())

            # Extraer cuotas de Bet365 o Pinnacle
            bookmakers = evento.get("bookmakers", [])
            cuota_home = cuota_draw = cuota_away = None

            for bm in bookmakers:
                nombre_bm = bm.get("title", "").lower()
                if "bet365" in nombre_bm or "pinnacle" in nombre_bm:
                    for market in bm.get("markets", []):
                        if market.get("key") == "h2h":
                            outcomes = market.get("outcomes", [])
                            for o in outcomes:
                                nombre_o = o.get("name", "").lower()
                                price = o.get("price", 0)
                                if "draw" in nombre_o:
                                    cuota_draw = price
                                elif nombre_o in home.lower() or home[:4] in nombre_o:
                                    cuota_home = price
                                else:
                                    cuota_away = price
                    if cuota_home:
                        break

            if cuota_home and cuota_draw and cuota_away:
                p_home = cuota_a_probabilidad(cuota_home)
                p_draw = cuota_a_probabilidad(cuota_draw)
                p_away = cuota_a_probabilidad(cuota_away)
                p_home_n, p_draw_n, p_away_n = normalizar_probabilidades(
                    p_home, p_draw, p_away
                )
                margen = round(p_home + p_draw + p_away - 100, 1)

                partidos_odds.append({
                    "equipo_a": home_app,
                    "equipo_b": away_app,
                    "cuota_a": cuota_home,
                    "cuota_x": cuota_draw,
                    "cuota_b": cuota_away,
                    "prob_implicita_a": p_home_n,
                    "prob_implicita_x": p_draw_n,
                    "prob_implicita_b": p_away_n,
                    "margen_casa": margen,
                    "fuente": "Bet365/Pinnacle"
                })

        return partidos_odds

    except Exception:
        return []


def buscar_odds_partido(ea: str, eb: str, api_key: str) -> dict | None:
    """
    Busca las cuotas de un partido específico.
    Devuelve dict con probabilidades implícitas o None si no hay datos.
    """
    todos = obtener_odds_mundial(api_key)
    for p in todos:
        if (ea.lower() in p["equipo_a"].lower() or p["equipo_a"].lower() in ea.lower()) and \
           (eb.lower() in p["equipo_b"].lower() or p["equipo_b"].lower() in eb.lower()):
            return p
        if (eb.lower() in p["equipo_a"].lower() or p["equipo_a"].lower() in eb.lower()) and \
           (ea.lower() in p["equipo_b"].lower() or p["equipo_b"].lower() in ea.lower()):
            # Partido en orden inverso — invertir probabilidades
            return {
                **p,
                "equipo_a": ea,
                "equipo_b": eb,
                "cuota_a": p["cuota_b"],
                "cuota_b": p["cuota_a"],
                "prob_implicita_a": p["prob_implicita_b"],
                "prob_implicita_b": p["prob_implicita_a"],
            }
    return None


def calcular_valor(prob_modelo: float, prob_implicita: float) -> dict:
    """
    Calcula si hay valor en una apuesta (EV+).
    prob_modelo: probabilidad de nuestro modelo (0-100)
    prob_implicita: probabilidad implícita de la casa (0-100)
    """
    diferencia = prob_modelo - prob_implicita
    if diferencia >= 8:
        return {"emoji": "🟢", "texto": f"VALOR +{diferencia:.1f}%", "color": "#4ade80"}
    elif diferencia >= 3:
        return {"emoji": "🟡", "texto": f"Valor leve +{diferencia:.1f}%", "color": "#f0c040"}
    elif diferencia >= -3:
        return {"emoji": "⚪", "texto": "Sin valor claro", "color": "#6677aa"}
    else:
        return {"emoji": "🔴", "texto": f"Sin valor -{abs(diferencia):.1f}%", "color": "#ef4444"}


def mostrar_comparacion_odds(ea: str, eb: str, r: dict, api_key: str):
    """
    Muestra la comparación entre nuestro modelo y las cuotas de las casas.
    Se llama desde app.py después de la simulación.
    """
    if not api_key:
        return

    odds = buscar_odds_partido(ea, eb, api_key)
    if not odds:
        return  # sin datos, silencioso

    pa = r["prob_a"]
    pd_ = r["prob_emp"]
    pb = r["prob_b"]

    val_a = calcular_valor(pa, odds["prob_implicita_a"])
    val_x = calcular_valor(pd_, odds["prob_implicita_x"])
    val_b = calcular_valor(pb, odds["prob_implicita_b"])

    st.markdown(f"""
    <div style="background:#0d1827;border:1px solid #1e3a5f;border-radius:12px;
    padding:1rem 1.2rem;margin-top:1rem">
      <div style="font-size:0.6rem;color:#6677aa;letter-spacing:2px;
      text-transform:uppercase;margin-bottom:0.75rem">
      📊 Comparación con casas de apuestas ({odds['fuente']})
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;text-align:center">

        <div style="background:#111827;border-radius:8px;padding:0.6rem">
          <div style="font-size:0.7rem;color:#e8eaf0;font-weight:600">{ea[:10]}</div>
          <div style="font-size:1.1rem;color:#f0c040;font-family:Bebas Neue">{odds['cuota_a']:.2f}</div>
          <div style="font-size:0.65rem;color:#6677aa">Casa: {odds['prob_implicita_a']:.1f}%</div>
          <div style="font-size:0.65rem;color:#8899bb">Modelo: {pa:.1f}%</div>
          <div style="font-size:0.65rem;color:{val_a['color']};margin-top:0.2rem">
          {val_a['emoji']} {val_a['texto']}</div>
        </div>

        <div style="background:#111827;border-radius:8px;padding:0.6rem">
          <div style="font-size:0.7rem;color:#e8eaf0;font-weight:600">Empate</div>
          <div style="font-size:1.1rem;color:#f0c040;font-family:Bebas Neue">{odds['cuota_x']:.2f}</div>
          <div style="font-size:0.65rem;color:#6677aa">Casa: {odds['prob_implicita_x']:.1f}%</div>
          <div style="font-size:0.65rem;color:#8899bb">Modelo: {pd_:.1f}%</div>
          <div style="font-size:0.65rem;color:{val_x['color']};margin-top:0.2rem">
          {val_x['emoji']} {val_x['texto']}</div>
        </div>

        <div style="background:#111827;border-radius:8px;padding:0.6rem">
          <div style="font-size:0.7rem;color:#e8eaf0;font-weight:600">{eb[:10]}</div>
          <div style="font-size:1.1rem;color:#f0c040;font-family:Bebas Neue">{odds['cuota_b']:.2f}</div>
          <div style="font-size:0.65rem;color:#6677aa">Casa: {odds['prob_implicita_b']:.1f}%</div>
          <div style="font-size:0.65rem;color:#8899bb">Modelo: {pb:.1f}%</div>
          <div style="font-size:0.65rem;color:{val_b['color']};margin-top:0.2rem">
          {val_b['emoji']} {val_b['texto']}</div>
        </div>

      </div>
      <div style="font-size:0.6rem;color:#4a5568;margin-top:0.5rem;text-align:right">
      Margen de la casa: {odds['margen_casa']:.1f}% · 🟢 Valor si modelo > casa en 8%+
      </div>
    </div>
    """, unsafe_allow_html=True)
