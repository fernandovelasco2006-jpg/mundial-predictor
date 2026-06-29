import streamlit as st
import numpy as np
from collections import Counter
import os

# Módulo de cuotas en tiempo real
try:
    from odds_api import mostrar_comparacion_odds, buscar_odds_partido
    ODDS_DISPONIBLE = True
except ImportError:
    ODDS_DISPONIBLE = False

# Módulo de actualización automática de resultados
try:
    from football_data import actualizar_fixture_y_forma
    FD_DISPONIBLE = True
except ImportError:
    FD_DISPONIBLE = False

# Módulo de predicciones persistentes (Supabase)
try:
    from supabase_preds import (simular_y_guardar_dia, cargar_todas_predicciones,
                                 calcular_accuracy, guardar_apuestas_dia,
                                 actualizar_aciertos, cargar_historial_apuestas,
                                 calcular_stats_apuestas)
    SUPABASE_DISPONIBLE = True
except ImportError:
    SUPABASE_DISPONIBLE = False

# Módulo Dixon-Coles Bayesiano
try:
    from dixon_coles import calcular_lambdas_dc, simular_dc, disponible as dc_disponible
    DC_DISPONIBLE = dc_disponible()
except Exception:
    DC_DISPONIBLE = False

# Módulo ML — XGBoost (importación condicional)
try:
    from modelo_ml import calcular_lambdas_xgb, disponible as ml_disponible
    ML_DISPONIBLE = ml_disponible()
except Exception:
    ML_DISPONIBLE = False

# Módulo de API en tiempo real (importación condicional)
try:
    from api_football import (
        sincronizar_resultados, calcular_corners_promedio,
        mostrar_status_api, obtener_fixtures_mundial
    )
    API_DISPONIBLE = True
except ImportError:
    API_DISPONIBLE = False

# ── Cache del modelo Dixon-Coles — se entrena UNA SOLA VEZ por sesión ────────
@st.cache_resource(show_spinner=False)
def _cargar_modelo_dc():
    if not DC_DISPONIBLE:
        return None
    try:
        from dixon_coles import inicializar_modelo
        return inicializar_modelo()
    except Exception:
        return True

_modelo_dc_cargado = _cargar_modelo_dc()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mundial 2026 · Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }
.stApp { background: #0a0e1a; color: #e8eaf0; }
.block-container { padding: 2rem 2rem 4rem; max-width: 1100px; }

.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
    border: 1px solid #1e3a5f; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero::before {
    content: "🏆"; position: absolute; right: 2rem; top: 50%;
    transform: translateY(-50%); font-size: 5rem; opacity: 0.07;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 3rem;
    letter-spacing: 4px; color: #f0c040; margin: 0; line-height: 1;
}
.hero-sub {
    color: #8899bb; font-size: 0.8rem; margin-top: 0.4rem;
    letter-spacing: 1px; text-transform: uppercase;
}

.prob-bar { display:flex; height:12px; border-radius:6px; overflow:hidden; margin:0.75rem 0; }
.bar-a { background:#3b82f6; }
.bar-draw { background:#4b5563; }
.bar-b { background:#ef4444; }

.result-box {
    background: linear-gradient(135deg, #0d1b2a, #1a2744);
    border: 1px solid #2a4a7f; border-radius: 14px;
    padding: 1.5rem 1rem; text-align: center;
}
.result-box-draw { border-color: #374151; }
.result-box-b { border-color: #3b1f1f; }
.team-flag { font-size: 2rem; }
.team-name {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.3rem;
    letter-spacing: 2px; color: #e8eaf0; margin: 0.2rem 0;
}
.prob-pct { font-family: 'Bebas Neue', sans-serif; font-size: 3rem; line-height:1; color: #f0c040; }
.prob-pct-b { color: #f87171; }
.prob-pct-draw { color: #9ca3af; }
.prob-lbl { font-size: 0.6rem; color: #6677aa; letter-spacing: 2px; text-transform: uppercase; }
.goles-esp { font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem; color: #60a5fa; }

.real-result {
    background: #0d1f16; border: 1px solid #2d6b45;
    border-radius: 12px; padding: 1rem 1.5rem;
    text-align: center; margin-bottom: 1.5rem;
}
.real-score {
    font-family: 'Bebas Neue', sans-serif; font-size: 2.5rem;
    color: #4ade80; letter-spacing: 6px;
}

.score-top {
    display:inline-block; background:#1a1800; border:1px solid #f0c040;
    border-radius:8px; padding:0.3rem 0.9rem; margin:0.2rem;
    font-family:'Bebas Neue', sans-serif; font-size:1.2rem; color:#f0c040;
}
.score-badge {
    display:inline-block; background:#1e2d45; border:1px solid #2a4060;
    border-radius:8px; padding:0.3rem 0.9rem; margin:0.2rem;
    font-family:'Bebas Neue', sans-serif; font-size:1.2rem; color:#e8eaf0;
}

.tag {
    display:inline-block; border-radius:20px; padding:2px 10px;
    font-size:0.65rem; letter-spacing:1px; text-transform:uppercase;
    margin-right:0.4rem;
}
.tag-group   { background:#2a1a00; color:#f0c040; border:1px solid #5a3a00; }
.tag-played  { background:#1a3a2a; color:#4ade80; border:1px solid #2d6b45; }
.tag-pending { background:#1a2744; color:#60a5fa; border:1px solid #2a4a7f; }

.card-y { display:inline-block; width:11px; height:15px; background:#f0c040; border-radius:2px; margin-right:3px; vertical-align:middle; }
.card-r { display:inline-block; width:11px; height:15px; background:#ef4444; border-radius:2px; margin-right:3px; vertical-align:middle; }

.metric-box {
    background:#111827; border:1px solid #1e2d45;
    border-radius:10px; padding:0.9rem; text-align:center;
}
.metric-val { font-family:'Bebas Neue',sans-serif; font-size:2rem; color:#f0c040; line-height:1; }
.metric-lbl { font-size:0.6rem; color:#6677aa; letter-spacing:1px; text-transform:uppercase; margin-top:0.2rem; }

.model-note {
    background:#0d1620; border-left:3px solid #3b82f6;
    border-radius:0 8px 8px 0; padding:0.7rem 1rem;
    font-size:0.75rem; color:#8899bb; margin-top:1.2rem;
}

.stSelectbox > div > div {
    background:#111827 !important; border:1px solid #1e2d45 !important;
    color:#e8eaf0 !important; border-radius:8px !important;
}
.stButton > button {
    background: linear-gradient(135deg,#1a4a7f,#2a6abf) !important;
    font-size:1.1rem !important;
    color:white !important; border:none !important; border-radius:8px !important;
    font-family:'Bebas Neue', sans-serif !important; font-size:1rem !important;
    letter-spacing:2px !important; padding:0.6rem 2rem !important; width:100% !important;
}
.stButton > button:hover { background:linear-gradient(135deg,#2a5a8f,#3a7acf) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATOS: ELO, ALTITUDES, BANDERAS
# ─────────────────────────────────────────────────────────────────────────────
ELO = {
    "Argentina":             1877,
    "Espana":                1875,
    "Francia":               1871,
    "Inglaterra":            1828,
    "Portugal":              1768,
    "Brasil":                1746,
    "Marruecos":             1710,
    "Paises Bajos":          1728,
    "Belgica":               1720,
    "Alemania":              1750,
    "Croacia":               1715,
    "Colombia":              1698,
    "Mexico":                1640,
    "Senegal":               1684,
    "Uruguay":               1680,
    "Estados Unidos":        1650,
    "Japon":                 1660,
    "Suiza":                 1660,
    "Iran":                  1580,
    "Turquia":               1620,
    "Ecuador":               1610,
    "Austria":               1597,
    "Corea del Sur":         1630,
    "Australia":             1590,
    "Argelia":               1571,
    "Egipto":                1560,
    "Canada":                1670,
    "Noruega":               1557,
    "Costa de Marfil":       1570,
    "Panama":                1539,
    "Escocia":               1545,
    "Chequia":               1590,
    "Paraguay":              1520,
    "Suecia":                1600,
    "Tunez":                 1530,
    "RD Congo":              1474,
    "Ghana":          2.3,
    "Catar":                 1420,
    "Arabia Saudita":        1424,
    "Jordania":              1388,
    "Bosnia y Herzegovina":  1540,
    "Irak":                  1446,
    "Uzbekistan":            1459,
    "Cabo Verde":            1500,
    "Sudafrica":             1480,
    "Haiti":                 1380,
    "Nueva Zelanda":  3.8,
    "Curazao":               1320,
    # aliases
    "Algeria":               1571,
    "Arabia Saudi":          1424,
}

ALTITUD = {
    "Azteca":       2240,
    "Guadalajara":  1566,
    "Monterrey":     540,
    "Atlanta":       320,
    "Kansas City":   270,
    "Dallas":        180,
    "Los Angeles":    25,
    "Toronto":        76,
    "Boston":         65,
    "Philadelphia":   12,
    "Seattle":        10,
    "Houston":        14,
    "San Francisco":  11,
    "Miami":           3,
    "Nueva York":      2,
    "Vancouver":       2,
}

CORNERS_EQUIPO = {
    # Calibrado con datos reales Mundial 2026 (60% real + 40% histórico)
    # J2+J3 desglosados disponibles
    "Espana":          5.9,  "Alemania":       4.4,  "Brasil":         5.8,
    "Inglaterra":      7.5,  "Paises Bajos":   5.8,  "Marruecos":      7.2,
    "Japon":           3.8,  "Escocia":        5.6,  "Canada":         5.9,
    "Corea del Sur":   5.4,  "Ecuador":        5.6,  "Panama":         5.5,
    "Noruega":         4.8,  "Colombia":       5.5,  "Belgica":        5.8,
    "Bosnia y Herzegovina": 4.4, "Francia":    4.5,  "Chequia":        4.6,
    "Suecia":          5.8,  "Australia":      3.8,  "Senegal":        7.2,
    "Catar":           4.2,  "Cabo Verde":     3.8,  "Turquia":        3.0,
    "Algeria":         3.6,  "Sudafrica":      3.6,  "Paraguay":       3.6,
    "Nueva Zelanda":   3.6,  "Portugal":       3.9,  "Costa de Marfil":4.7,
    "Jordania":        3.8,  "Egipto":         5.2,  "Tunez":          3.6,
    "RD Congo":        3.3,  "Croacia":        2.7,  "Suiza":          3.0,
    "Uruguay":         5.5,  "Estados Unidos": 6.5,  "Mexico":         2.6,
    "Argentina":       2.5,  "Uzbekistan":     2.5,  "Irak":           2.6,
    "Ghana":           2.8,  "Iran":           2.3,  "Haiti":          1.8,
    "Arabia Saudita":  1.5,  "Arabia Saudi":   1.6,  "Curazao":        2.2,
    "Austria":         3.5,  "Argelia":        3.6,
}
CORNERS_DEFAULT = 4.0

HORARIOS_PARTIDO = {
    ('Turquia', 'Paraguay'): '2026-06-19 21:00',
    ('Paises Bajos', 'Suecia'): '2026-06-20 11:00',
    ('Alemania', 'Costa de Marfil'): '2026-06-20 14:00',
    ('Ecuador', 'Curazao'): '2026-06-20 18:00',
    ('Tunez', 'Japon'): '2026-06-20 22:00',
    ('Espana', 'Arabia Saudita'): '2026-06-21 10:00',
    ('Belgica', 'Iran'): '2026-06-21 13:00',
    ('Cabo Verde', 'Uruguay'): '2026-06-21 16:00',
    ('Nueva Zelanda', 'Egipto'): '2026-06-21 19:00',
    ('Argentina', 'Austria'): '2026-06-22 11:00',
    ('Francia', 'Irak'): '2026-06-22 15:00',
    ('Senegal', 'Noruega'): '2026-06-22 18:00',
    ('Algeria', 'Jordania'): '2026-06-22 21:00',
    ('Portugal', 'Uzbekistan'): '2026-06-23 11:00',
    ('Inglaterra', 'Ghana'): '2026-06-23 14:00',
    ('Croacia', 'Panama'): '2026-06-23 17:00',
    ('Colombia', 'RD Congo'): '2026-06-23 20:00',
    ('Suiza', 'Canada'): '2026-06-24 13:00',
    ('Bosnia y Herzegovina', 'Catar'): '2026-06-24 13:00',
    ('Escocia', 'Brasil'): '2026-06-24 16:00',
    ('Marruecos', 'Haiti'): '2026-06-24 16:00',
    ('Chequia', 'Mexico'): '2026-06-24 19:00',
    ('Sudafrica', 'Corea del Sur'): '2026-06-24 19:00',
    ('Paraguay', 'Australia'): '2026-06-25 13:00',
    ('Turquia', 'Estados Unidos'): '2026-06-25 13:00',
    ('Japon', 'Suecia'): '2026-06-25 16:00',
    ('Tunez', 'Paises Bajos'): '2026-06-25 16:00',
    ('Curazao', 'Costa de Marfil'): '2026-06-25 19:00',
    ('Ecuador', 'Alemania'): '2026-06-25 19:00',
    ('Noruega', 'Francia'): '2026-06-26 13:00',
    ('Senegal', 'Irak'): '2026-06-26 13:00',
    ('Nueva Zelanda', 'Belgica'): '2026-06-26 16:00',
    ('Egipto', 'Iran'): '2026-06-26 16:00',
    ('Uruguay', 'Espana'): '2026-06-26 19:00',
    ('Cabo Verde', 'Arabia Saudita'): '2026-06-26 19:00',
    ('Jordania', 'Argentina'): '2026-06-27 13:00',
    ('Algeria', 'Austria'): '2026-06-27 13:00',
    ('Panama', 'Inglaterra'): '2026-06-27 16:00',
    ('Croacia', 'Ghana'): '2026-06-27 16:00',
    ('Colombia', 'Portugal'): '2026-06-27 19:00',
    ('RD Congo', 'Uzbekistan'): '2026-06-27 19:00',
    ("Sudafrica",           "Canada"):            "2026-06-28 13:00",
    ("Alemania",            "Paraguay"):          "2026-06-29 14:30",
    ("Paises Bajos",        "Marruecos"):         "2026-06-29 19:00",
    ("Brasil",              "Japon"):             "2026-06-29 11:00",
    ("Costa de Marfil",     "Noruega"):           "2026-06-30 11:00",
    ("Francia",             "Suecia"):            "2026-06-30 15:00",
    ("Mexico",              "Ecuador"):           "2026-06-30 19:00",
    ("Inglaterra",          "RD Congo"):          "2026-07-01 10:00",
    ("Belgica",             "Senegal"):           "2026-07-01 14:00",
    ("Estados Unidos",      "Bosnia y Herzegovina"): "2026-07-01 18:00",
    ("Espana",              "Austria"):           "2026-07-02 13:00",
    ("Portugal",            "Croacia"):           "2026-07-02 17:00",
    ("Suiza",               "Algeria"):           "2026-07-02 21:00",
    ("Australia",           "Egipto"):            "2026-07-03 12:00",
    ("Argentina",           "Cabo Verde"):        "2026-07-03 16:00",
    ("Colombia",            "Ghana"):             "2026-07-03 19:30",
    ("TBD-R16-1A",  "TBD-R16-1B"):  "2026-07-04 16:00",
    ("TBD-R16-2A",  "TBD-R16-2B"):  "2026-07-04 12:00",
    ("TBD-R16-3A",  "TBD-R16-3B"):  "2026-07-05 15:00",
    ("TBD-R16-4A",  "TBD-R16-4B"):  "2026-07-05 19:00",
    ("TBD-R16-5A",  "TBD-R16-5B"):  "2026-07-06 14:00",
    ("TBD-R16-6A",  "TBD-R16-6B"):  "2026-07-06 19:00",
    ("TBD-R16-7A",  "TBD-R16-7B"):  "2026-07-06 11:00",
    ("TBD-R16-8A",  "TBD-R16-8B"):  "2026-07-07 15:00",
    ("TBD-QF-1A",   "TBD-QF-1B"):   "2026-07-09 15:00",
    ("TBD-QF-2A",   "TBD-QF-2B"):   "2026-07-10 14:00",
    ("TBD-QF-3A",   "TBD-QF-3B"):   "2026-07-11 16:00",
    ("TBD-QF-4A",   "TBD-QF-4B"):   "2026-07-11 20:00",
    ("TBD-SF-1A",   "TBD-SF-1B"):   "2026-07-14 14:00",
    ("TBD-SF-2A",   "TBD-SF-2B"):   "2026-07-15 14:00",
    ("TBD-3P-1A",   "TBD-3P-1B"):   "2026-07-18 16:00",
    ("TBD-F-1A",    "TBD-F-1B"):    "2026-07-19 14:00",
}

CLIMA = {
    "Azteca":       (24, 72),
    "Guadalajara":  (33, 55),
    "Monterrey":    (36, 60),
    "Miami":        (33, 84),
    "Houston":      (34, 75),
    "Dallas":       (36, 58),
    "Atlanta":      (32, 68),
    "Kansas City":  (31, 65),
    "Los Angeles":  (26, 70),
    "San Francisco":(20, 75),
    "Seattle":      (21, 65),
    "Boston":       (26, 67),
    "Nueva York":   (28, 65),
    "Philadelphia": (29, 67),
    "Toronto":      (25, 63),
    "Vancouver":    (20, 68),
}

EQUIPOS_CALOR = {
    "Brasil", "Senegal", "Costa de Marfil", "Ghana", "Camerun",
    "Nigeria", "RD Congo", "Marruecos", "Egipto", "Arabia Saudi",
    "Irak", "Iran", "Colombia", "Ecuador", "Panama", "Haiti",
    "Catar", "Uzbekistan", "Mexico", "Estados Unidos", "Canada",
}
EQUIPOS_FRIO = {
    "Noruega", "Suecia", "Dinamarca", "Finlandia", "Islandia",
    "Escocia", "Irlanda", "Belgica", "Paises Bajos", "Alemania",
    "Suiza", "Austria", "Chequia", "Polonia", "Croacia",
    "Bosnia y Herzegovina", "Eslovenia", "Serbia",
}

LOCAL_SEDES = {
    "Mexico":        ["Azteca", "Guadalajara", "Monterrey"],
    "Canada":        ["Toronto", "Vancouver"],
    "Estados Unidos": ["Los Angeles", "Dallas", "Nueva York", "Boston",
                       "Seattle", "Kansas City", "San Francisco", "Houston",
                       "Miami", "Atlanta", "Philadelphia"],
}

BANDERAS = {
    "Mexico": "🇲🇽", "Sudafrica": "🇿🇦", "Corea del Sur": "🇰🇷", "Chequia": "🇨🇿",
    "Canada": "🇨🇦", "Bosnia y Herzegovina": "🇧🇦", "Catar": "🇶🇦", "Suiza": "🇨🇭",
    "Brasil": "🇧🇷", "Marruecos": "🇲🇦", "Haiti": "🇭🇹", "Escocia": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Estados Unidos": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turquia": "🇹🇷",
    "Alemania": "🇩🇪", "Curazao": "🇨🇼", "Costa de Marfil": "🇨🇮", "Ecuador": "🇪🇨",
    "Paises Bajos": "🇳🇱", "Japon": "🇯🇵", "Suecia": "🇸🇪", "Tunez": "🇹🇳",
    "Belgica": "🇧🇪", "Egipto": "🇪🇬", "Iran": "🇮🇷", "Nueva Zelanda": "🇳🇿",
    "Espana": "🇪🇸", "Cabo Verde": "🇨🇻", "Arabia Saudita": "🇸🇦", "Uruguay": "🇺🇾",
    "Francia": "🇫🇷", "Senegal": "🇸🇳", "Irak": "🇮🇶", "Noruega": "🇳🇴",
    "Argentina": "🇦🇷", "Argelia": "🇩🇿", "Algeria": "🇩🇿",
    "Austria": "🇦🇹", "Jordania": "🇯🇴",
    "Portugal": "🇵🇹", "RD Congo": "🇨🇩", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    "Inglaterra": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croacia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
    "Arabia Saudi": "🇸🇦",
}

FLAG_ISO = {
    "Mexico":"mx","Sudafrica":"za","Corea del Sur":"kr","Chequia":"cz",
    "Canada":"ca","Bosnia y Herzegovina":"ba","Catar":"qa","Suiza":"ch",
    "Brasil":"br","Marruecos":"ma","Haiti":"ht","Escocia":"gb-sct",
    "Estados Unidos":"us","Paraguay":"py","Australia":"au","Turquia":"tr",
    "Alemania":"de","Curazao":"cw","Costa de Marfil":"ci","Ecuador":"ec",
    "Paises Bajos":"nl","Japon":"jp","Suecia":"se","Tunez":"tn",
    "Belgica":"be","Egipto":"eg","Iran":"ir","Nueva Zelanda":"nz",
    "Espana":"es","Cabo Verde":"cv","Arabia Saudi":"sa","Arabia Saudita":"sa",
    "Uruguay":"uy","Francia":"fr","Senegal":"sn","Irak":"iq","Noruega":"no",
    "Argentina":"ar","Algeria":"dz","Argelia":"dz","Austria":"at","Jordania":"jo",
    "Portugal":"pt","RD Congo":"cd","Uzbekistan":"uz","Colombia":"co",
    "Inglaterra":"gb-eng","Croacia":"hr","Ghana":"gh","Panama":"pa",
}

def flag_img(equipo: str, size: int = 48) -> str:
    iso = FLAG_ISO.get(equipo, "un")
    emoji = BANDERAS.get(equipo, "🏳️")
    border_r = "50%" if size >= 40 else "6px"
    shadow = "0 2px 8px rgba(0,0,0,0.5)" if size >= 40 else "0 1px 3px rgba(0,0,0,0.3)"
    border_css = "2px solid rgba(255,255,255,0.18)" if size >= 40 else "1px solid rgba(255,255,255,0.1)"
    src1 = f"https://flagcdn.com/w{size*2}/{iso}.png"
    src2 = f"https://flagicons.lipis.dev/flags/4x3/{iso}.svg"
    fallback_js = f"this.onerror=null;this.src='{src2}';this.onerror=function(){{this.style.display='none';this.parentNode.querySelector('.fe').style.display='block'}}"
    return (
        f'<span style="display:inline-block;width:{size}px;height:{size}px;'
        f'border-radius:{border_r};border:{border_css};box-shadow:{shadow};'
        f'overflow:hidden;vertical-align:middle;background:#1a2540;'
        f'position:relative;text-align:center;line-height:{size}px">'
        f'<img src="{src1}" width="{size}" height="{size}" '
        f'style="border-radius:{border_r};object-fit:cover;display:block" '
        f'onerror="{fallback_js}" alt="{equipo}">'
        f'<span class="fe" style="display:none;font-size:{int(size*0.75)}px;'
        f'position:absolute;top:0;left:0;width:100%;height:100%;'
        f'line-height:{size}px;text-align:center">{emoji}</span>'
        f'</span>'
    )

def flag(t): return BANDERAS.get(t, "🏳️")


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE — J1+J2 actualizados con resultados reales
# ─────────────────────────────────────────────────────────────────────────────
PARTIDOS = [
    # ── JORNADA 1 ─────────────────────────────────────────────────────────────
    ("Mexico",          "Sudafrica",              "A", "Azteca",        (2, 0),  "Wilton Sampaio"),
    ("Corea del Sur",   "Chequia",                "A", "Guadalajara",   (2, 1),  "Amin Mohamed Omar"),
    ("Canada",          "Bosnia y Herzegovina",   "B", "Toronto",       (1, 1),  None),
    ("Estados Unidos",  "Paraguay",               "D", "Los Angeles",   (4, 1),  "Felix Zwayer"),
    ("Catar",           "Suiza",                  "B", "Vancouver",     (1, 1),  None),
    ("Brasil",          "Marruecos",              "C", "Nueva York",    (1, 1),  "Mustapha Ghorbal"),
    ("Haiti",           "Escocia",                "C", "Boston",        (0, 1),  None),
    ("Australia",       "Turquia",                "D", "Kansas City",   (2, 0),  None),
    ("Alemania",        "Curazao",                "E", "Houston",       (7, 1),  None),
    ("Costa de Marfil", "Ecuador",                "E", "Philadelphia",  (1, 0),  None),
    ("Paises Bajos",    "Japon",                  "F", "Dallas",        (2, 2),  "Danny Makkelie"),
    ("Suecia",          "Tunez",                  "F", "Monterrey",     (5, 1),  None),
    ("Belgica",         "Egipto",                 "G", "Seattle",       (1, 1),  None),
    ("Iran",            "Nueva Zelanda",          "G", "Los Angeles",   (2, 2),  "Omar Al Ali"),
    ("Espana",          "Cabo Verde",             "H", "Atlanta",       (0, 0),  None),
    ("Arabia Saudi",    "Uruguay",                "H", "Miami",         (1, 1),  None),
    ("Francia",         "Senegal",                "I", "Nueva York",    (3, 1),  "Francois Letexier"),
    ("Irak",            "Noruega",                "I", "Boston",        (1, 4),  None),
    ("Argentina",       "Algeria",                "J", "Kansas City",   (3, 0),  "Szymon Marciniak"),
    ("Austria",         "Jordania",               "J", "San Francisco", (3, 1),  None),
    ("Portugal",        "RD Congo",               "K", "Houston",       (1, 1),  "Abdulrahman Al-Jassim"),
    ("Uzbekistan",      "Colombia",               "K", "Azteca",        (1, 3),  "Anthony Taylor"),
    ("Inglaterra",      "Croacia",                "L", "Dallas",        (4, 2),  "Clement Turpin"),
    ("Ghana",           "Panama",                 "L", "Toronto",       (1, 0),  "Glenn Nyberg"),

    # ── JORNADA 2 ─────────────────────────────────────────────────────────────
    ("Chequia",         "Sudafrica",              "A", "Atlanta",       (1, 1),  "Tori Penso"),
    ("Mexico",          "Corea del Sur",          "A", "Guadalajara",   (1, 0),  "Gustavo Tejera"),
    ("Suiza",           "Bosnia y Herzegovina",   "B", "Los Angeles",   (4, 1),  "Joao Pinheiro"),
    ("Canada",          "Catar",                  "B", "Vancouver",     (6, 0),  "Cristian Garay"),
    ("Escocia",         "Marruecos",              "C", "Boston",        (0, 1),  "Ilgiz Tantashev"),
    ("Brasil",          "Haiti",                  "C", "Philadelphia",  (3, 0),  "Raphael Claus"),
    ("Estados Unidos",  "Australia",              "D", "Seattle",       (2, 0),  "Felix Zwayer"),
    ("Turquia",         "Paraguay",               "D", "San Francisco", (0, 1),  "Szymon Marciniak"),
    ("Alemania",        "Costa de Marfil",        "E", "Toronto",       (2, 1),  "Dario Herrera"),
    ("Ecuador",         "Curazao",                "E", "Kansas City",   (0, 0),  "Raphael Claus"),
    ("Paises Bajos",    "Suecia",                 "F", "Houston",       (5, 1),  "Cesar Arturo Ramos"),
    ("Tunez",           "Japon",                  "F", "Monterrey",     (0, 4),  "Istvan Kovacs"),
    ("Belgica",         "Iran",                   "G", "Los Angeles",   (0, 0),  "Dario Herrera"),
    ("Nueva Zelanda",   "Egipto",                 "G", "Vancouver",     (1, 3),  "Omar Al Ali"),
    ("Espana",          "Arabia Saudita",         "H", "Atlanta",       (4, 0),  "Raphael Claus"),   # ← J2 actualizado
    ("Cabo Verde",      "Uruguay",                "H", "Miami",         (2, 2),  "Espen Eskas"),
    ("Francia",         "Irak",                   "I", "Philadelphia",  (3, 0),  "Drew Fischer"),    # ← J2 actualizado
    ("Senegal",         "Noruega",                "I", "Nueva York",    (2, 3),  "Wilton Sampaio"),  # ← J2 actualizado
    ("Argentina",       "Austria",                "J", "Dallas",        (2, 0),  "Amin Mohamed Omar"), # ← J2 actualizado
    ("Algeria",         "Jordania",               "J", "San Francisco", (2, 1),  "Slavko Vincic"),   # ← J2 actualizado
    ("Portugal",        "Uzbekistan",             "K", "Houston",       (5, 0),  "Jalal Jayed"),
    ("Colombia",        "RD Congo",               "K", "Guadalajara",   (1, 0),  "Maurizio Mariani"),
    ("Inglaterra",      "Ghana",                  "L", "Boston",        (0, 0),  "Said Martinez"),
    ("Croacia",         "Panama",                 "L", "Toronto",       (1, 0),  "Pierre Ghislain Atcho"),

    # ── JORNADA 3 ─────────────────────────────────────────────────────────────
    ("Mexico",               "Chequia",           "A", "Azteca",        (3, 0),  "Yael Falcon Perez"),
    ("Sudafrica",            "Corea del Sur",     "A", "Monterrey",     (1, 0),  "Facundo Tello"),
    ("Suiza",                "Canada",            "B", "Vancouver",     (2, 1),  "Ramon Abatti Abel"),
    ("Bosnia y Herzegovina", "Catar",             "B", "Seattle",       (3, 1),  "Jesus Valenzuela"),
    ("Escocia",              "Brasil",            "C", "Miami",         (0, 3),  "Cesar Ramos Palazuelos"),
    ("Marruecos",            "Haiti",             "C", "Philadelphia",  (4, 2),  "Danny Makkelie"),
    ("Turquia",              "Estados Unidos",    "D", "Seattle",       (3, 2),  "Mustapha Ghorbal"),
    ("Paraguay",             "Australia",         "D", "San Francisco", (0, 0),  "Clement Turpin"),
    ("Ecuador",              "Alemania",          "E", "Nueva York",    (2, 1),  "Tori Penso"),
    ("Curazao",              "Costa de Marfil",   "E", "Kansas City",   (0, 2),  "Glenn Nyberg"),
    ("Tunez",                "Paises Bajos",      "F", "Houston",       (1, 3),  "Katia Garcia"),
    ("Japon",                "Suecia",            "F", "Dallas",        (1, 1),  "Ivan Barton"),
    ("Nueva Zelanda",        "Belgica",           "G", "Vancouver",     (1, 5),  "Adham Mohammad"),
    ("Egipto",               "Iran",              "G", "Boston",        (1, 1),  "Szymon Marciniak"),
    ("Cabo Verde",           "Arabia Saudita",    "H", "Atlanta",       (0, 0),  "Francois Letexier"),
    ("Uruguay",              "Espana",            "H", "Guadalajara",   (0, 1),  "Ismail Elfath"),
    ("Senegal",              "Irak",              "I", "Kansas City",   (5, 0),  "Anthony Taylor"),
    ("Noruega",              "Francia",           "I", "Nueva York",    (1, 4),  "Michael Oliver"),
    ("Algeria",              "Austria",           "J", "Dallas",        (3, 3),  "Ilgiz Tantashev"),
    ("Jordania",             "Argentina",         "J", "Dallas",        (1, 3),  "Istvan Kovacs"),
    ("RD Congo",             "Uzbekistan",        "K", "Azteca",        (3, 1),  "Felix Zwayer"),
    ("Colombia",             "Portugal",          "K", "Miami",         (0, 0),  "Alireza Faghani"),
    ("Croacia",              "Ghana",             "L", "Toronto",       (2, 1),  "Drew Fischer"),
    ("Panama",               "Inglaterra",        "L", "Guadalajara",   (0, 2),  "Abdulrahman Al-Jassim"),

    # ── ELIMINATORIAS ─────────────────────────────────────────────────────────
    ("Sudafrica",           "Canada",             "R32", "Los Angeles",   (0, 1),  "Joao Pedro Pinheiro"),
    ("Alemania",            "Paraguay",           "R32", "Boston",        None,    "Jalal Jayed"),
    ("Paises Bajos",        "Marruecos",          "R32", "Monterrey",     None,    "Wilton Pereira Sampaio"),
    ("Brasil",              "Japon",              "R32", "Houston",       None,    "Maurizio Mariani"),
    ("Costa de Marfil",     "Noruega",            "R32", "Nueva York",     None, None),
    ("Francia",             "Suecia",             "R32", "Dallas",         None, None),
    ("Mexico",              "Ecuador",            "R32", "Azteca",        None, None),
    ("Inglaterra",          "RD Congo",           "R32", "Atlanta",       None, None),
    ("Belgica",             "Senegal",            "R32", "San Francisco",  None, None),
    ("Estados Unidos",      "Bosnia y Herzegovina","R32","Seattle",        None, None),
    ("Espana",              "Austria",            "R32", "Toronto",        None, None),
    ("Portugal",            "Croacia",            "R32", "Los Angeles",    None, None),
    ("Suiza",               "Algeria",            "R32", "Vancouver",     None, None),
    ("Australia",           "Egipto",             "R32", "Miami",          None, None),
    ("Argentina",           "Cabo Verde",         "R32", "Kansas City",    None, None),
    ("Colombia",            "Ghana",              "R32", "Dallas",         None, None),
    ("TBD-R16-1A",  "TBD-R16-1B",  "R16", "Philadelphia",   None, None),
    ("TBD-R16-2A",  "TBD-R16-2B",  "R16", "Houston",        None, None),
    ("TBD-R16-3A",  "TBD-R16-3B",  "R16", "Nueva York",     None, None),
    ("TBD-R16-4A",  "TBD-R16-4B",  "R16", "Azteca",         None, None),
    ("TBD-R16-5A",  "TBD-R16-5B",  "R16", "Dallas",         None, None),
    ("TBD-R16-6A",  "TBD-R16-6B",  "R16", "Seattle",        None, None),
    ("TBD-R16-7A",  "TBD-R16-7B",  "R16", "Atlanta",        None, None),
    ("TBD-R16-8A",  "TBD-R16-8B",  "R16", "Vancouver",      None, None),
    ("TBD-QF-1A",   "TBD-QF-1B",   "QF",  "Boston",         None, None),
    ("TBD-QF-2A",   "TBD-QF-2B",   "QF",  "Los Angeles",    None, None),
    ("TBD-QF-3A",   "TBD-QF-3B",   "QF",  "Miami",          None, None),
    ("TBD-QF-4A",   "TBD-QF-4B",   "QF",  "Kansas City",    None, None),
    ("TBD-SF-1A",   "TBD-SF-1B",   "SF",  "Dallas",         None, None),
    ("TBD-SF-2A",   "TBD-SF-2B",   "SF",  "Atlanta",        None, None),
    ("TBD-3P-1A",   "TBD-3P-1B",   "3P",  "Miami",          None, None),
    ("TBD-F-1A",    "TBD-F-1B",    "F",   "Nueva York",     None, None),
]


# ─────────────────────────────────────────────────────────────────────────────
# H2H — incluye resultados J1+J2 del Mundial 2026
# ─────────────────────────────────────────────────────────────────────────────
H2H = {
    ("Corea del Sur", "Mexico"):     [(2026, 0, 1, 2, 0), (2022, 2, 3, 6, 1), (2018, 1, 2, 4, 0)],
    ("Brasil", "Haiti"):             [(2016, 7, 1, 0, 0)],
    ("Brasil", "Marruecos"):         [(2023, 2, 1, 3, 0)],
    ("Escocia", "Marruecos"):        [(2023, 1, 2, 4, 0)],
    ("Australia", "Estados Unidos"): [(2023, 0, 2, 3, 0), (2025, 1, 2, 3, 0)],
    ("Ecuador", "Alemania"):         [(2022, 0, 2, 4, 0)],
    ("Japon", "Suecia"):             [(2023, 2, 1, 4, 0)],
    ("Paises Bajos", "Tunez"):       [(2022, 3, 1, 3, 0)],
    ("Belgica", "Iran"):             [(2022, 2, 0, 4, 1)],
    ("Arabia Saudi", "Espana"):      [(2022, 2, 1, 6, 1)],
    ("Arabia Saudita", "Espana"):    [(2022, 2, 1, 6, 1), (2026, 0, 4, 2, 0)],
    ("Argentina", "Austria"):        [(2024, 0, 1, 3, 0), (2026, 2, 0, 2, 0)],
    ("Ghana", "Panama"):             [(2022, 3, 2, 7, 1)],
    ("Inglaterra", "Ghana"):         [(2023, 3, 1, 3, 0)],
    # J1 2026
    ("Suecia", "Tunez"):             [(2026, 5, 1, 1, 0)],
    ("Argentina", "Algeria"):        [(2026, 3, 0, 0, 0)],
    # J2 2026 nuevos
    ("Francia", "Irak"):             [(2026, 3, 0, 0, 1)],
    ("Senegal", "Noruega"):          [(2026, 2, 3, 0, 0)],
    ("Algeria", "Jordania"):         [(2026, 2, 1, 1, 0)],
    # J2 2026 hoy
    ("Portugal", "Uzbekistan"):      [(2026, 5, 0, 0, 0)],
    ("Inglaterra", "Ghana"):         [(2023, 3, 1, 3, 0), (2026, 0, 0, 1, 0)],
    ("Croacia", "Panama"):           [(2026, 1, 0, 1, 0)],
    ("Colombia", "RD Congo"):        [(2026, 1, 0, 2, 0)],
    # J3 2026
    ("Bosnia y Herzegovina", "Catar"): [(2026, 3, 1, 1, 0)],
    ("Suiza", "Canada"):               [(2026, 2, 1, 1, 0)],
    # J3 grupo A y C
    ("Mexico", "Chequia"):            [(2026, 3, 0, 1, 0)],
    ("Sudafrica", "Corea del Sur"):   [(2026, 1, 0, 2, 0)],
    ("Marruecos", "Haiti"):           [(2026, 4, 2, 3, 0)],
    ("Escocia", "Brasil"):            [(2026, 0, 3, 3, 0)],
    # J3 grupos E y F
    ("Tunez", "Paises Bajos"):        [(2026, 1, 3, 0, 0)],
    ("Curazao", "Costa de Marfil"):   [(2026, 0, 2, 2, 0)],
    ("Ecuador", "Alemania"):          [(2026, 2, 1, 4, 0)],
    ("Japon", "Suecia"):              [(2026, 1, 1, 3, 0)],
    ("Paraguay", "Australia"):        [(2026, 0, 0, 2, 0)],
    ("Turquia", "Estados Unidos"):    [(2026, 3, 2, 1, 0)],
    ("Noruega", "Francia"):           [(2026, 1, 4, 2, 0)],
    ("Senegal", "Irak"):              [(2026, 5, 0, 4, 1)],
    ("Cabo Verde", "Arabia Saudita"): [(2026, 0, 0, 4, 0)],
    ("Uruguay", "Espana"):            [(2026, 0, 1, 4, 1)],
    ("Croacia", "Ghana"):             [(2026, 2, 1, 2, 0)],
    ("Nueva Zelanda", "Belgica"):     [(2026, 1, 5, 2, 0)],
    ("Egipto", "Iran"):               [(2026, 1, 1, 7, 0)],
    ("Panama", "Inglaterra"):         [(2026, 0, 2, 3, 0)],
    ("Algeria", "Austria"):           [(2026, 3, 3, 1, 0)],
    ("Jordania", "Argentina"):        [(2026, 1, 3, 2, 0)],
    ("Colombia", "Portugal"):         [(2026, 0, 0, 1, 0)],
    ("RD Congo", "Uzbekistan"):       [(2026, 3, 1, 5, 0)],
}

def calcular_factor_h2h(ea: str, eb: str) -> tuple:
    datos = H2H.get((ea, eb)) or H2H.get((eb, ea))
    if not datos:
        return 1.0, 1.0, None
    invertir = H2H.get((ea, eb)) is None
    año_actual = 2026
    peso_total = goles_a_pond = goles_b_pond = tarj_am_pond = tarj_ro_pond = 0
    for entrada in datos:
        año, ga, gb = entrada[0], entrada[1], entrada[2]
        am = entrada[3] if len(entrada) > 3 else None
        ro = entrada[4] if len(entrada) > 4 else None
        if invertir:
            ga, gb = gb, ga
        anos_atras = año_actual - año
        peso = 0.5 ** (anos_atras / 4)
        peso_total    += peso
        goles_a_pond  += ga * peso
        goles_b_pond  += gb * peso
        if am is not None:
            tarj_am_pond += am * peso
            tarj_ro_pond += ro * peso
    if peso_total == 0:
        return 1.0, 1.0, None
    avg_a = goles_a_pond / peso_total
    avg_b = goles_b_pond / peso_total
    diff  = avg_a - avg_b
    ajuste = min(abs(diff) * 0.04, 0.10)
    if diff > 0:   f_a, f_b = 1.0 + ajuste, 1.0 - ajuste
    elif diff < 0: f_a, f_b = 1.0 - ajuste, 1.0 + ajuste
    else:          f_a, f_b = 1.0, 1.0
    tarj_hist = None
    if tarj_am_pond > 0:
        tarj_hist = (round(tarj_am_pond / peso_total, 2), round(tarj_ro_pond / peso_total, 3))
    return f_a, f_b, tarj_hist


# ─────────────────────────────────────────────────────────────────────────────
# ÁRBITROS
# ─────────────────────────────────────────────────────────────────────────────
ARBITROS = {
    "Fernando Rapallini":         (4.80, 0.25),
    "Pierre-Ghislain Atcho":      (3.67, 0.17),
    "Juan Gabriel Benitez":       (4.64, 0.31),
    "Szymon Marciniak":           (4.10, 0.18),
    "Alejandro Hernandez":        (5.20, 0.30),
    "Istvan Kovacs":              (4.90, 0.28),
    "Joao Pinheiro":              (4.70, 0.23),
    "Maurizio Mariani":           (4.65, 0.25),
    "Felix Zwayer":               (4.40, 0.16),
    "Sandro Scharer":             (4.35, 0.21),
    "Slavko Vincic":              (4.20, 0.17),
    "Anthony Taylor":             (3.95, 0.14),
    "Espen Eskas":                (3.90, 0.13),
    "Francois Letexier":          (3.85, 0.19),
    "Glenn Nyberg":               (3.80, 0.11),
    "Michael Oliver":             (3.70, 0.12),
    "Clement Turpin":             (3.60, 0.22),
    "Danny Makkelie":             (3.45, 0.15),
    "Dario Herrera":              (5.40, 0.35),
    "Kevin Ortega":               (5.05, 0.33),
    "Wilton Sampaio":             (5.08, 0.28),
    "Andres Matonte":             (5.10, 0.31),
    "Gustavo Tejera":             (5.15, 0.29),
    "Facundo Tello":              (5.02, 0.32),
    "Piero Maza":                 (4.95, 0.27),
    "Cristian Garay":             (4.90, 0.25),
    "Raphael Claus":              (4.80, 0.29),
    "Andres Rojas":               (4.80, 0.28),
    "Yael Falcon Perez":          (4.75, 0.24),
    "Juan Benitez":               (4.70, 0.26),
    "Jesus Valenzuela":           (4.60, 0.22),
    "Ramon Abatti Abel":          (4.55, 0.20),
    "Cesar Ramos Palazuelos":     (4.50, 0.22),
    "Ivan Barton":                (4.70, 0.25),
    "Said Martinez":              (4.60, 0.22),
    "Ismail Elfath":              (4.45, 0.20),
    "Juan Calderon":              (4.40, 0.19),
    "Cesar Arturo Ramos":         (4.30, 0.22),
    "Oshane Nation":              (4.25, 0.21),
    "Katia Itzel Garcia":         (4.15, 0.15),
    "Drew Fischer":               (3.90, 0.14),
    "Tori Penso":                 (3.65, 0.12),
    "Pierre Atcho":               (4.30, 0.21),
    "Abongile Tom":               (4.20, 0.18),
    "Dahane Beida":               (4.15, 0.17),
    "Amin Mohamed Omar":          (4.10, 0.16),
    "Mustapha Ghorbal":           (4.05, 0.14),
    "Jalal Jayed":          (3.75, 0.17),
    "Ma Ning":                    (4.95, 0.29),
    "Adham Makhadmeh":            (4.50, 0.22),
    "Alireza Faghani":            (4.40, 0.20),
    "Omar Al Ali":                (4.20, 0.16),
    "Khalid Al-Turais":           (4.10, 0.14),
    "Ilgiz Tantashev":            (4.05, 0.15),
    "Abdulrahman Al-Jassim":      (3.80, 0.13),
    "Yusuke Araki":               (3.65, 0.11),
    "Campbell-Kirk Kawana-Waugh": (3.85, 0.14),
    "Katia Garcia":              (3.90, 0.10),  # CONCACAF
    "Adham Mohammad":         (3.64, 0.13),  # AFC
    "Wilton Pereira Sampaio": (5.09, 0.24),  # CONMEBOL
    "Joao Pedro Pinheiro":      (4.72, 0.20),  # UEFA
}
ARBITRO_DEFAULT = (3.80, 0.12)


# ─────────────────────────────────────────────────────────────────────────────
# TARJETAS_MUNDIAL — J1+J2 completa
# ─────────────────────────────────────────────────────────────────────────────
TARJETAS_MUNDIAL = {
    # Grupo A — J1+J2+J3 completa
    "Mexico":               (3, 0, 3),   # 2 J1+J2 + 1 J3
    "Sudafrica":            (3, 0, 3),   # 2 J1+J2 + 1 J3
    "Corea del Sur":        (5, 0, 3),   # 4 J1+J2 + 1 J3
    "Chequia":              (2, 0, 3),   # 2 J1+J2 + 0 J3
    # Grupo B — J1+J2+J3 completa
    "Canada":               (3, 0, 3),   # 1 J1 + 0 J2 + 2 J3
    "Bosnia y Herzegovina": (4, 1, 3),   # 3 J1+J2 + 1 J3
    "Catar":                (3, 2, 3),   # 2 J1+J2 + 1 J3
    "Suiza":                (3, 0, 3),   # 2 J1+J2 + 1 J3
    # Grupo C — J1+J2+J3 completa
    "Brasil":               (3, 0, 3),   # 1 J1+J2 + 2 J3
    "Marruecos":            (2, 0, 3),   # 2 J1+J2 + 0 J3
    "Haiti":                (4, 0, 3),   # 1 J1+J2 + 3 J3
    "Escocia":              (2, 0, 3),   # 1 J1+J2 + 1 J3
    # Grupo D — J1+J2+J3 completa
    "Estados Unidos":       (2, 0, 3),   # 1+0+1
    "Paraguay":             (3, 1, 3),   # 2+1 J3: 1am
    "Australia":            (2, 0, 3),   # 1+1 J3: 1am
    "Turquia":              (1, 0, 3),   # 0+0 J3: 0am (Ghorbal permisivo)
    # Grupo E — J1+J2+J3 completa
    "Alemania":             (2, 0, 3),   # 1+0 J3: 1am
    "Curazao":              (3, 0, 3),   # 2+1 J3: 2am
    "Costa de Marfil":      (4, 0, 3),   # 3+1 J3: 1am
    "Ecuador":              (4, 0, 3),   # 1+3 J3: 3am
    # Grupo F — J1+J2+J3 completa
    "Paises Bajos":         (4, 0, 3),   # 3+0 J3: 0am
    "Japon":                (4, 0, 3),   # 2+1 J3: 1am
    "Suecia":               (5, 0, 3),   # 2+2 J3: 2am
    "Tunez":                (3, 1, 3),   # 3+0 J3: 0am
    # Grupo G — J1+J2+J3 completa
    "Belgica":              (4, 0, 3),   # 2+2 J3: 2am NZ
    "Egipto":               (6, 0, 3),   # 2+3 J3: 3am
    "Iran":                 (8, 0, 3),   # 4+4 J3: 4am
    "Nueva Zelanda":        (4, 0, 3),   # 2+2 J3: 2am
    # Grupo H — J1+J2+J3 completa
    "Espana":               (2, 0, 3),   # 1+1 J3: 1am
    "Cabo Verde":           (5, 0, 3),   # 4+1 J3: 1am
    "Arabia Saudi":         (1, 0, 2),
    "Arabia Saudita":       (7, 0, 3),   # 4+3 J3: 3am
    "Uruguay":              (6, 1, 3),   # 2+3+roja J3
    # Grupo I — J1+J2+J3 completa
    "Francia":              (3, 0, 3),   # 2+1 J3: 1am
    "Senegal":              (6, 0, 3),   # 4+2 J3: 2am
    "Irak":                 (5, 1, 3),   # 3+2 J3: 2am (+ roja J3)
    "Noruega":              (5, 0, 3),   # 4+1 J3: 1am
    # Grupo J — J1+J2 completa (J3 pendiente 27 jun)
    "Argentina":            (2, 0, 2),
    "Algeria":              (3, 0, 2),
    "Argelia":              (3, 0, 2),
    "Austria":              (3, 0, 2),
    "Jordania":             (4, 0, 2),
    # Grupo K — J1+J2+J3 completa
    "Portugal":             (3, 0, 3),   # 1+0+2 J3
    "RD Congo":             (5, 0, 3),   # 3+2 J3
    "Uzbekistan":           (3, 1, 3),   # sin amarillas J3
    "Colombia":             (6, 0, 3),   # 1+2+3 J3
    # Grupo L — J1+J2 completa
    # Inglaterra 1 am J2, Ghana 1 am J2 (Said Martinez 4.2 prom)
    # Croacia 1 am J2 (Sučić), Panamá 1 am J2
    "Inglaterra":           (3, 0, 3),   # 1 J1 + 1 J2
    "Croacia":              (4, 0, 3),   # 2 J1 + 1 J2
    "Ghana":                (4, 0, 3),   # 2 J1 + 1 J2
    "Panama":               (4, 0, 3),   # 1 J1 + 1 J2
}


# ─────────────────────────────────────────────────────────────────────────────
# FORMA_MUNDIAL — J1+J2 completa para todos los grupos
# ─────────────────────────────────────────────────────────────────────────────
FORMA_MUNDIAL = {
    # Grupo A — J1+J2+J3 completa
    "Mexico":               (6, 0, 3),   # 2-0 + 1-0 + 3-0
    "Sudafrica":            (2, 3, 3),   # 0-2 + 1-1 + 1-0
    "Corea del Sur":        (2, 3, 3),   # 2-1 + 0-1 + 0-1
    "Chequia":              (2, 6, 3),   # 1-2 + 1-1 + 0-3
    # Grupo B
    "Canada":               (7, 1, 2),
    "Bosnia y Herzegovina": (2, 5, 2),
    "Catar":                (1, 7, 2),
    "Suiza":                (5, 2, 2),
    # Grupo C — J1+J2+J3 completa
    "Brasil":               (7, 1, 3),   # 1-1 + 3-0 + 3-0
    "Marruecos":            (6, 3, 3),   # 1-1 + 1-0 + 4-2
    "Haiti":                (2, 8, 3),   # 0-1 + 0-3 + 2-4
    "Escocia":              (1, 4, 3),   # 1-0 + 0-1 + 0-3
    # Grupo D
    "Estados Unidos":       (8, 4, 3),
    "Paraguay":             (2, 4, 3),
    "Australia":            (2, 4, 2),
    "Turquia":              (2, 3, 2),
    # Grupo E — J1+J2+J3 completa
    "Alemania":             (10, 4, 3),   # 4-0 CUR + 2-1 CIV + 1-2 ECU
    "Ecuador":              (2, 2, 3),    # 0-0 + 2-1 ALE (actualizar con J1 real)
    "Costa de Marfil":      (4, 2, 3),   # 1-2 + 2-1 ALE + 2-0 CUR
    "Curazao":              (1, 9, 3),   # 0-4 + 1-2 + 0-2
    # Grupo F
    "Paises Bajos":         (7, 3, 2),
    "Japon":                (6, 2, 2),
    "Suecia":               (6, 3, 2),
    "Tunez":                (1, 9, 2),
    # Grupo G
    "Belgica":              (6, 2, 3),
    "Egipto":               (5, 3, 3),
    "Iran":                 (3, 3, 3),
    "Nueva Zelanda":        (3, 5, 2),
    # Grupo H — J1+J2 completa
    "Espana":               (5, 0, 3),   # 0-0 Cabo Verde + 4-0 Arabia Saudita
    "Cabo Verde":           (2, 2, 3),
    "Arabia Saudi":         (1, 5, 2),   # 1-1 Uruguay + 0-4 España
    "Arabia Saudita":       (1, 5, 2),
    "Uruguay":              (3, 3, 2),
    # Grupo I — J1+J2 completa
    "Francia":             (10, 2, 3),   # 3-1 Senegal + 3-0 Irak
    "Senegal":              (3, 4, 2),   # 1-3 Francia + 2-3 Noruega
    "Irak":                 (1, 7, 2),   # 1-4 Noruega + 0-3 Francia
    "Noruega":              (7, 3, 2),   # 4-1 Irak + 3-2 Senegal
    # Grupo J — J1+J2 completa
    "Argentina":            (8, 1, 3),   # 3-0 Argelia + 2-0 Austria
    "Algeria":              (5, 7, 3),   # 0-3 Argentina + 2-1 Jordania
    "Argelia":              (2, 4, 2),
    "Austria":              (6, 6, 3),   # 3-1 Jordania + 0-2 Argentina
    "Jordania":             (3, 8, 3),   # 1-3 Austria + 1-2 Argelia
    # Grupo K — J1+J2 completa
    "Portugal":             (6, 1, 3),   # 1-1 RD Congo + 5-0 Uzbekistán
    "RD Congo":             (4, 3, 3),   # 1-1 Portugal + 0-1 Colombia
    "Uzbekistan":           (2, 11, 3),   # 1-3 Colombia + 0-5 Portugal
    "Colombia":             (4, 1, 3),   # 3-1 Uzbekistán + 1-0 RD Congo
    # Grupo L — J1+J2 completa
    "Inglaterra":           (6, 2, 3),   # 4-2 Croacia + 0-0 Ghana
    "Croacia":              (5, 5, 3),   # 2-4 Inglaterra + 1-0 Panamá
    "Ghana":                (2, 4, 3),   # 1-0 Panamá + 0-0 Inglaterra
    "Panama":               (0, 4, 3),   # 0-1 Ghana + 0-1 Croacia
}


# ─────────────────────────────────────────────────────────────────────────────
# BAJAS — actualizadas post J2
# ─────────────────────────────────────────────────────────────────────────────
BAJAS = {
    'Brasil':        0.90,  # Neymar Jr — baja J2+J3
    'Uruguay':       0.93,  # De Arrascaeta + Araujo
    'Austria':       0.92,  # Alaba (salió lesionado J2) + Posch
    'Japon':         0.92,  # Kubo + Ueda
    'Marruecos':     0.94,  # Ez Abde + Aguerd
    'Portugal':      0.95,  # Ruben Dias
    'Chequia':       0.96,  # Kuchta
    'Ghana':         0.95,  # Lawrence Ati Zigi (portero)
    'Espana':        0.96,  # Mikel Merino
    # Mexico: suspensión J2 ya cumplida — sin penalización J3
    'Paraguay':      0.95,  # Sosa + Caballero
    'Panama':        0.96,  # Carrasquilla
    'Suiza':         0.97,  # Muheim
    'Canada':        0.97,  # Jones + Flores
    'Argelia':       0.96,  # Burgess + Toure
    'Noruega':       0.97,  # Østigård dudoso J3
}

def h2h_mundial_2026(ea: str, eb: str) -> tuple:
    for partido in PARTIDOS:
        pa, pb = partido[0], partido[1]
        res = partido[4]
        if res is None:
            continue
        if (pa == ea and pb == eb):
            return (res[0], res[1], None, None)
        if (pa == eb and pb == ea):
            return (res[1], res[0], None, None)
    return None


def calcular_factor_h2h_completo(ea: str, eb: str) -> tuple:
    res_2026 = h2h_mundial_2026(ea, eb)
    datos_hist = H2H.get((ea, eb)) or H2H.get((eb, ea))
    invertir_hist = datos_hist is not None and H2H.get((ea, eb)) is None
    año_actual = 2026
    peso_total = goles_a_pond = goles_b_pond = tarj_am_pond = tarj_ro_pond = 0
    fuentes = []
    if datos_hist:
        for entrada in datos_hist:
            año, ga, gb = entrada[0], entrada[1], entrada[2]
            am = entrada[3] if len(entrada) > 3 else None
            ro = entrada[4] if len(entrada) > 4 else None
            if invertir_hist:
                ga, gb = gb, ga
            anos_atras = año_actual - año
            peso = 0.5 ** (anos_atras / 4)
            peso_total   += peso
            goles_a_pond += ga * peso
            goles_b_pond += gb * peso
            if am is not None:
                tarj_am_pond += am * peso
                tarj_ro_pond += ro * peso
        fuentes.append("historial")
    if res_2026:
        ga_26, gb_26, am_26, ro_26 = res_2026
        peso_26 = 4.0
        peso_total   += peso_26
        goles_a_pond += ga_26 * peso_26
        goles_b_pond += gb_26 * peso_26
        if am_26 is not None:
            tarj_am_pond += am_26 * peso_26
            tarj_ro_pond += ro_26 * peso_26
        fuentes.append(f"Mundial 2026 ({ga_26}-{gb_26})")
    if peso_total == 0:
        return 1.0, 1.0, None, "sin datos H2H"
    avg_a = goles_a_pond / peso_total
    avg_b = goles_b_pond / peso_total
    diff  = avg_a - avg_b
    ajuste = min(abs(diff) * 0.04, 0.12)
    if diff > 0:   f_a, f_b = 1.0 + ajuste, 1.0 - ajuste
    elif diff < 0: f_a, f_b = 1.0 - ajuste, 1.0 + ajuste
    else:          f_a, f_b = 1.0, 1.0
    tarj_hist = None
    if tarj_am_pond > 0:
        tarj_hist = (round(tarj_am_pond / peso_total, 2), round(tarj_ro_pond / peso_total, 3))
    desc = " + ".join(fuentes) if fuentes else "sin datos H2H"
    return f_a, f_b, tarj_hist, desc


def calcular_lambdas(ea: str, eb: str, sede: str):
    elo_a = ELO.get(ea, 1500)
    elo_b = ELO.get(eb, 1500)
    diff = elo_a - elo_b
    ajuste = (1 / (1 + 10 ** (-diff / 400))) * 2 - 1
    lam_a = 1.55 * (1.0 + ajuste * 0.35)
    lam_b = 1.55 * (1.0 - ajuste * 0.35)
    for equipo in [ea, eb]:
        forma = FORMA_MUNDIAL.get(equipo)
        if forma and forma[2] > 0:
            gf, gc, pj = forma
            avg_gf = gf / pj
            avg_gc = gc / pj
            f_of = max(1.0 + min((avg_gf - 1.3) / 1.3, 0.08), 0.92)
            f_def = max(1.0 + min((avg_gc - 1.3) / 1.3, 0.08), 0.92)
            if equipo == ea:
                lam_a *= f_of
                lam_b *= f_def
            else:
                lam_b *= f_of
                lam_a *= f_def
    alt = ALTITUD.get(sede, 200)
    if alt > 1700:
        equipos_altos = {"Mexico", "Sudafrica"}
        if ea not in equipos_altos: lam_a *= 0.92
        if eb not in equipos_altos: lam_b *= 0.92
    for equipo, sedes in LOCAL_SEDES.items():
        if ea == equipo and sede in sedes: lam_a *= 1.10
        if eb == equipo and sede in sedes: lam_b *= 1.10
    if ea in BAJAS: lam_a *= BAJAS[ea]
    if eb in BAJAS: lam_b *= BAJAS[eb]
    clima = CLIMA.get(sede, (25, 65))
    temp, humedad = clima
    indice_calor = (temp - 20) / 10 + (humedad - 60) / 40
    if indice_calor > 0:
        penalizacion = min(indice_calor * 0.03, 0.08)
        if ea in EQUIPOS_FRIO:   lam_a *= (1.0 - penalizacion)
        if eb in EQUIPOS_FRIO:   lam_b *= (1.0 - penalizacion)
        if ea in EQUIPOS_CALOR:  lam_a *= (1.0 + penalizacion * 0.3)
        if eb in EQUIPOS_CALOR:  lam_b *= (1.0 + penalizacion * 0.3)
    fh2h_a, fh2h_b, _, _ = calcular_factor_h2h_completo(ea, eb)
    lam_a *= fh2h_a
    lam_b *= fh2h_b
    return max(lam_a, 0.15), max(lam_b, 0.15)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULACIÓN — optimizada para web: int32 + cache inteligente + liberar RAM
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def simular(ea: str, eb: str, sede: str, arbitro: str = None, n: int = 10_000) -> dict:
    rng = np.random.default_rng()
    modelo_usado = "Manual"
    baja_a = BAJAS.get(ea, 1.0)
    baja_b = BAJAS.get(eb, 1.0)
    forma_a = FORMA_MUNDIAL.get(ea)
    forma_b = FORMA_MUNDIAL.get(eb)
    forma_dc_a = {'gf': forma_a[0]/forma_a[2], 'gc': forma_a[1]/forma_a[2], 'pj': forma_a[2]} if forma_a and forma_a[2]>0 else None
    forma_dc_b = {'gf': forma_b[0]/forma_b[2], 'gc': forma_b[1]/forma_b[2], 'pj': forma_b[2]} if forma_b and forma_b[2]>0 else None

    lam_a, lam_b = calcular_lambdas(ea, eb, sede)

    if DC_DISPONIBLE:
        try:
            _sedes_mexico = {"Azteca", "Akron", "Guadalajara", "Monterrey"}
            _sedes_canada = {"Toronto", "Vancouver", "BC Place", "BMO Field"}
            _sedes_usa    = {"Los Angeles", "New York", "Dallas", "Seattle",
                             "Houston", "Boston", "Philadelphia", "Kansas City",
                             "San Francisco", "Atlanta", "Miami", "SoFi"}
            _es_local_a = (
                (ea == "Mexico" and sede in _sedes_mexico) or
                (ea == "Canada" and sede in _sedes_canada) or
                (ea == "Estados Unidos" and sede in _sedes_usa)
            )
            _es_local_b = (
                (eb == "Mexico" and sede in _sedes_mexico) or
                (eb == "Canada" and sede in _sedes_canada) or
                (eb == "Estados Unidos" and sede in _sedes_usa)
            )
            lam_a_dc, lam_b_dc, _ = calcular_lambdas_dc(
                ea, eb, es_neutral=not _es_local_a and not _es_local_b,
                forma_mundial_a=forma_dc_a, forma_mundial_b=forma_dc_b,
                bajas_a=baja_a, bajas_b=baja_b
            )
            if lam_a_dc and lam_b_dc:
                lam_a, lam_b = lam_a_dc, lam_b_dc
                modelo_usado = "Dixon-Coles 🎯"
        except Exception:
            pass

    if modelo_usado == "Manual" and ML_DISPONIBLE:
        try:
            lam_a_xgb, lam_b_xgb, _ = calcular_lambdas_xgb(
                ea, eb, sede, es_neutral=True, torneo='FIFA World Cup', mes=6,
                bajas_a=baja_a, bajas_b=baja_b
            )
            if lam_a_xgb and lam_b_xgb:
                lam_a, lam_b = lam_a_xgb, lam_b_xgb
                modelo_usado = "XGBoost 🤖"
        except Exception:
            pass

    # Simulación con int32 — mitad de RAM vs int64
    if DC_DISPONIBLE and modelo_usado == "Dixon-Coles 🎯":
        ga, gb = simular_dc(lam_a, lam_b, n)
        ga = ga.astype(np.int32)
        gb = gb.astype(np.int32)
    elif lam_a >= lam_b:
        ga = rng.poisson(lam_a, n).astype(np.int32)
        gb = rng.negative_binomial(4, float(np.float32(4 / (4 + lam_b))), n).astype(np.int32)
    else:
        ga = rng.negative_binomial(4, float(np.float32(4 / (4 + lam_a))), n).astype(np.int32)
        gb = rng.poisson(lam_b, n).astype(np.int32)

    prob_a   = float(np.sum(ga > gb)) / n * 100
    prob_b   = float(np.sum(gb > ga)) / n * 100
    prob_emp = float(np.sum(ga == gb)) / n * 100

    goles_tot = ga + gb
    prob_over05 = float(np.mean(goles_tot > 0) * 100)
    prob_over15 = float(np.mean(goles_tot > 1) * 100)
    prob_over25 = float(np.mean(goles_tot > 2) * 100)
    prob_over35 = float(np.mean(goles_tot > 3) * 100)
    prob_btts   = float(np.mean((ga > 0) & (gb > 0)) * 100)
    top5 = Counter(zip(ga.tolist(), gb.tolist())).most_common(5)

    corners_a = CORNERS_EQUIPO.get(ea, CORNERS_DEFAULT)
    corners_b = CORNERS_EQUIPO.get(eb, CORNERS_DEFAULT)
    corners_total_esp = corners_a + corners_b
    corners_sim = rng.poisson(corners_total_esp, n).astype(np.int32)
    prob_corners_over65  = float(np.mean(corners_sim > 6) * 100)
    prob_corners_over75  = float(np.mean(corners_sim > 7) * 100)
    prob_corners_over85  = float(np.mean(corners_sim > 8) * 100)
    prob_corners_over95  = float(np.mean(corners_sim > 9) * 100)
    prob_corners_under85 = 100 - prob_corners_over85
    prob_corners_under75 = float(np.mean(corners_sim <= 7) * 100)
    prob_corners_under65 = float(np.mean(corners_sim <= 6) * 100)

    lam_am_arb_raw, lam_ro_arb_raw = ARBITROS.get(arbitro, ARBITRO_DEFAULT) if arbitro else ARBITRO_DEFAULT
    # Promedios REALES de cada árbitro en este Mundial 2026
    # Formato: {nombre: (am_promedio_real, partidos_pitados)}
    # Si el árbitro ya pitó en el torneo, usamos su promedio real
    # Si no, usamos histórico * 0.65 (factor de ajuste del torneo)
    ARBITROS_MUNDIAL_2026 = {
        "Abdulrahman Al-Jassim":   (3.0,  2),  # Brasil-Marruecos + Panamá-Inglaterra
        "Adham Mohammad":          (2.0,  1),  # NZ-Bélgica
        "Alireza Faghani":         (1.0,  1),  # Colombia-Portugal
        "Anthony Taylor":          (4.0,  1),  # Senegal-Irak
        "Campbell-Kirk Kawana-Waugh": (3.0, 1), # Ghana-Panamá
        "Cesar Ramos Palazuelos":  (3.0,  1),  # Escocia-Brasil
        "Clement Turpin":          (2.0,  1),  # Paraguay-Australia
        "Danny Makkelie":          (3.0,  1),  # Marruecos-Haití
        "Drew Fischer":            (2.0,  1),  # Croacia-Ghana
        "Facundo Tello":           (2.0,  1),  # Sudáfrica-Corea
        "Felix Zwayer":            (5.0,  1),  # RD Congo-Uzbekistan
        "Fernando Rapallini":      (3.0,  1),  # Inglaterra-Croacia
        "Francois Letexier":       (4.0,  1),  # Cabo Verde-Arabia Saudita
        "Glenn Nyberg":            (2.5,  2),  # Haití-Escocia + Curazao-CIV
        "Ilgiz Tantashev":         (1.0,  1),  # Algeria-Austria
        "Ismail Elfath":           (4.0,  1),  # Uruguay-España
        "Istvan Kovacs":           (3.0,  2),  # Argentina-Algeria + Jordania-Argentina
        "Ivan Barton":             (2.0,  2),  # Francia-Irak + Japón-Suecia
        "Jalal Jayed":             (0.0,  1),  # Portugal-Uzbekistan
        "Jesus Valenzuela":        (2.0,  1),  # Bosnia-Catar
        "Katia Garcia":            (0.0,  1),  # Túnez-PB
        "Maurizio Mariani":        (3.0,  1),  # Colombia-RD Congo
        "Michael Oliver":          (2.0,  1),  # Noruega-Francia
        "Mustapha Ghorbal":        (1.0,  1),  # Turquía-EEUU
        "Pierre Ghislain Atcho":   (2.0,  1),  # Croacia-Panamá
        "Ramon Abatti Abel":       (3.0,  1),  # Suiza-Canadá
        "Raphael Claus":           (3.0,  1),  # México-Sudáfrica
        "Said Martinez":           (2.0,  2),  # España-KSA + Inglaterra-Ghana
        "Slavko Vincic":           (2.0,  1),  # Algeria-Jordania
        "Szymon Marciniak":        (7.0,  1),  # Egipto-Irán (outlier)
        "Tori Penso":              (4.0,  1),  # Ecuador-Alemania
        "Wilton Pereira Sampaio":  (0.0,  1),  # Senegal-Noruega
        "Yael Falcon Perez":       (1.0,  1),  # México-Chequia
        "Yusuke Araki":            (4.0,  1),  # Corea-Chequia
        "Joao Pedro Pinheiro":     (2.0,  1),  # Sudáfrica-Canadá (0 SUF + 2 CAN)
    }
    # Factor ajuste para árbitros sin datos del torneo
    FACTOR_ARB_MUNDIAL = 0.65

    if arbitro and arbitro in ARBITROS_MUNDIAL_2026:
        am_real, n_partidos = ARBITROS_MUNDIAL_2026[arbitro]
        # Con más partidos, más peso al dato real; con 1 partido, 60% real + 40% histórico
        peso_real = min(0.9, 0.6 + n_partidos * 0.15)
        lam_am_arb = am_real * peso_real + lam_am_arb_raw * FACTOR_ARB_MUNDIAL * (1 - peso_real)
        lam_ro_arb = lam_ro_arb_raw * FACTOR_ARB_MUNDIAL  # rojas: pocos datos, usar histórico ajustado
    else:
        lam_am_arb = lam_am_arb_raw * FACTOR_ARB_MUNDIAL
        lam_ro_arb = lam_ro_arb_raw * FACTOR_ARB_MUNDIAL

    def factor_tarjetas_equipo(equipo):
        datos = TARJETAS_MUNDIAL.get(equipo)
        if not datos or datos[2] == 0:
            return 1.0
        am, ro, pj = datos
        return max(0.7, min(1.4, (am / pj) / 1.9))

    factor_equipos = (factor_tarjetas_equipo(ea) + factor_tarjetas_equipo(eb)) / 2
    _, _, tarj_h2h, desc_h2h = calcular_factor_h2h_completo(ea, eb)

    if tarj_h2h:
        lam_am = lam_am_arb * 0.50 + tarj_h2h[0] * 0.25 + (lam_am_arb * factor_equipos) * 0.25
        lam_ro = lam_ro_arb * 0.50 + tarj_h2h[1] * 0.25 + (lam_ro_arb * factor_equipos) * 0.25
        fuente_tarj = f"Árbitro 50% + H2H 25% + Mundial 25% ({desc_h2h})"
    else:
        lam_am = lam_am_arb * 0.70 + (lam_am_arb * factor_equipos) * 0.30
        lam_ro = lam_ro_arb * 0.70 + (lam_ro_arb * factor_equipos) * 0.30
        fuente_tarj = f"Árbitro 70% + Equipos Mundial 30% ({desc_h2h})"

    tarjetas_am_sim = rng.poisson(lam_am, n).astype(np.int32)
    tarjetas_ro_sim = rng.poisson(max(lam_ro, 0.01), n).astype(np.int32)
    amarillas = float(np.mean(tarjetas_am_sim))
    rojas     = float(np.mean(tarjetas_ro_sim))
    prob_am_over15 = float(np.mean(tarjetas_am_sim > 1) * 100)
    prob_am_over25 = float(np.mean(tarjetas_am_sim > 2) * 100)
    prob_am_over35 = float(np.mean(tarjetas_am_sim > 3) * 100)
    prob_am_over45 = float(np.mean(tarjetas_am_sim > 4) * 100)
    prob_am_under35 = 100 - prob_am_over35
    prob_am_under25 = 100 - prob_am_over25

    # Guardar medias antes de liberar arrays grandes
    _ga_mean = float(np.mean(ga))
    _gb_mean = float(np.mean(gb))
    del ga, gb, goles_tot, corners_sim, tarjetas_am_sim, tarjetas_ro_sim

    return {
        "prob_a": prob_a, "prob_b": prob_b, "prob_emp": prob_emp,
        "prob_over05": prob_over05, "prob_over15": prob_over15,
        "prob_over25": prob_over25, "prob_over35_goles": prob_over35,
        "prob_btts": prob_btts,
        "goles_a": _ga_mean, "goles_b": _gb_mean,
        "top5": top5,
        "amarillas": round(amarillas, 1),
        "rojas": round(rojas, 2),
        "lam_a": round(lam_a, 3), "lam_b": round(lam_b, 3),
        "elo_a": ELO.get(ea, 1500), "elo_b": ELO.get(eb, 1500),
        "alt": ALTITUD.get(sede, 200),
        "arbitro": arbitro or "Sin asignar",
        "arbitro_am": round(lam_am_arb, 2), "arbitro_ro": round(lam_ro_arb, 3),
        "tarj_h2h": tarj_h2h,
        "fuente_tarj": fuente_tarj,
        "h2h_desc": desc_h2h,
        "modelo": modelo_usado,
        "corners_esp": corners_total_esp,
        "prob_corners_over65":  prob_corners_over65,
        "prob_corners_over75":  prob_corners_over75,
        "prob_corners_over85":  prob_corners_over85,
        "prob_corners_over95":  prob_corners_over95,
        "prob_corners_under85": prob_corners_under85,
        "prob_corners_under75": prob_corners_under75,
        "prob_corners_under65": prob_corners_under65,
        "prob_am_over15": prob_am_over15,
        "prob_am_over25": prob_am_over25,
        "prob_am_over35": prob_am_over35,
        "prob_am_over45": prob_am_over45,
        "prob_am_under35": prob_am_under35,
        "prob_am_under25": prob_am_under25,
    }


def analizar_apuestas(ea: str, eb: str, r: dict) -> list:
    apuestas = []
    UMBRAL_RESULTADO = 75.0
    UMBRAL_MERCADOS  = 80.0
    lam_a_r = r.get("lam_a", 1.5)
    lam_b_r = r.get("lam_b", 1.0)
    lam_total = lam_a_r + lam_b_r
    import math as _math
    prob_00 = _math.exp(-lam_a_r) * _math.exp(-lam_b_r) * 100
    ES_PARTIDO_DEFENSIVO = prob_00 > 8 or lam_total < 2.2
    if prob_00 > 8:
        UMBRAL_OVER05 = min(92.0, 80.0 + prob_00 * 0.5)
    elif prob_00 > 5:
        UMBRAL_OVER05 = 85.0
    else:
        UMBRAL_OVER05 = 80.0
    UMBRAL_OVER15 = 82.0 if lam_total < 2.5 else 80.0
    UMBRAL_TARJ = 75.0 if ES_PARTIDO_DEFENSIVO else 80.0
    UMBRAL_CORN = 75.0 if ES_PARTIDO_DEFENSIVO else 80.0

    pa  = r["prob_a"]
    pd_ = r["prob_emp"]
    pb  = r["prob_b"]
    amarillas = r["amarillas"]
    p_over05   = r.get("prob_over05",   95.0)
    p_over15   = r.get("prob_over15",   70.0)
    p_over25   = r.get("prob_over25",   45.0)
    p_over35_g = r.get("prob_over35_goles", 25.0)
    p_under25  = 100 - p_over25
    p_under15  = 100 - p_over15
    p_btts    = r.get("prob_btts",    40.0)
    p_no_btts = 100 - p_btts
    p_am_over15  = r.get("prob_am_over15",  70.0)
    p_am_over25  = r.get("prob_am_over25",  60.0)
    p_am_over35  = r.get("prob_am_over35",  40.0)
    p_am_over45  = r.get("prob_am_over45",  20.0)
    p_am_under35 = r.get("prob_am_under35", 60.0)
    p_am_under25 = r.get("prob_am_under25", 40.0)
    p_c_over65  = r.get("prob_corners_over65",  60.0)
    p_c_over75  = r.get("prob_corners_over75",  50.0)
    p_c_over85  = r.get("prob_corners_over85",  50.0)
    p_c_over95  = r.get("prob_corners_over95",  35.0)
    p_c_under85 = r.get("prob_corners_under85", 50.0)
    p_c_under75 = r.get("prob_corners_under75", 30.0)
    corners_esp = r.get("corners_esp", 8.0)

    def ap(mercado, seleccion, confianza, nota, donde):
        apuestas.append({
            "mercado": mercado, "seleccion": seleccion, "confianza": confianza,
            "nivel": "ALTA" if confianza >= 82 else "MEDIA",
            "nota": nota, "donde": donde
        })

    if pa >= UMBRAL_RESULTADO:
        ap("Resultado (1X2)", f"✅ Gana {ea}", pa, f"{pa:.1f}% de {r.get('n_sims','10M')} simulaciones", "Playdoit / Draftea → 1X2 → '1'")
    if pb >= UMBRAL_RESULTADO:
        ap("Resultado (1X2)", f"✅ Gana {eb}", pb, f"{pb:.1f}% de {r.get('n_sims','10M')} simulaciones", "Playdoit / Draftea → 1X2 → '2'")
    conf_1x = min(pa + pd_, 99)
    conf_x2 = min(pb + pd_, 99)
    if conf_1x >= UMBRAL_RESULTADO and pa < UMBRAL_RESULTADO:
        ap("Doble Oportunidad", f"✅ {ea} o Empate (1X)", conf_1x, f"{ea} {pa:.1f}% + Empate {pd_:.1f}% = {conf_1x:.1f}%", "Playdoit / Draftea → Doble Oportunidad → '1X'")
    if conf_x2 >= UMBRAL_RESULTADO and pb < UMBRAL_RESULTADO:
        ap("Doble Oportunidad", f"✅ {eb} o Empate (X2)", conf_x2, f"{eb} {pb:.1f}% + Empate {pd_:.1f}% = {conf_x2:.1f}%", "Playdoit / Draftea → Doble Oportunidad → 'X2'")
    if p_over05 >= UMBRAL_OVER05:
        ap("Total Goles", "✅ Over 0.5 (al menos 1 gol)", p_over05, f"{p_over05:.1f}% de simulaciones", "Playdoit / Draftea → Totales → 'Más/Menos 0.5' → Over")
    if p_over15 >= UMBRAL_OVER15:
        ap("Total Goles", "✅ Over 1.5 (2+ goles)", p_over15, f"{p_over15:.1f}% de simulaciones", "Playdoit / Draftea → Totales → 'Más/Menos 1.5' → Over")
    if p_over25 >= UMBRAL_MERCADOS:
        ap("Total Goles", "✅ Over 2.5 (3+ goles)", p_over25, f"{p_over25:.1f}% de simulaciones", "Playdoit / Draftea → Totales → 'Más/Menos 2.5' → Over")
    if p_over35_g >= UMBRAL_MERCADOS:
        ap("Total Goles", "✅ Over 3.5 (4+ goles)", p_over35_g, f"{p_over35_g:.1f}% de simulaciones", "Playdoit / Draftea → Totales → 'Más/Menos 3.5' → Over")
    if p_under15 >= UMBRAL_MERCADOS:
        ap("Total Goles", "✅ Under 1.5 (0 o 1 gol)", p_under15, f"{p_under15:.1f}% de simulaciones", "Playdoit / Draftea → Totales → 'Más/Menos 1.5' → Under")
    if p_under25 >= UMBRAL_MERCADOS:
        ap("Total Goles", "✅ Under 2.5 (0, 1 o 2 goles)", p_under25, f"{p_under25:.1f}% de simulaciones", "Playdoit / Draftea → Totales → 'Más/Menos 2.5' → Under")
    if p_btts >= UMBRAL_MERCADOS:
        ap("Ambos Marcan", "✅ Sí — ambos anotan", p_btts, f"{p_btts:.1f}% de simulaciones", "Playdoit / Draftea → Ambos Marcan → 'Sí'")
    if p_no_btts >= UMBRAL_MERCADOS:
        ap("Ambos Marcan", "✅ No — al menos uno no anota", p_no_btts, f"{p_no_btts:.1f}% de simulaciones", "Playdoit / Draftea → Ambos Marcan → 'No'")
    if p_am_over15 >= UMBRAL_TARJ and p_am_over15 < 98:
        ap("Tarjetas Amarillas", "✅ Over 1.5 amarillas (2+)", p_am_over15, f"{p_am_over15:.1f}% · {amarillas:.1f} esp.", "Playdoit / Draftea → Tarjetas → 'Más/Menos 1.5' → Over")
    if p_am_over25 >= UMBRAL_TARJ:
        ap("Tarjetas Amarillas", "✅ Over 2.5 amarillas (3+)", p_am_over25, f"{p_am_over25:.1f}% · {amarillas:.1f} esp.", "Playdoit / Draftea → Tarjetas → 'Más/Menos 2.5' → Over")
    if p_am_over35 >= UMBRAL_TARJ:
        ap("Tarjetas Amarillas", "✅ Over 3.5 amarillas (4+)", p_am_over35, f"{p_am_over35:.1f}% · {amarillas:.1f} esp.", "Playdoit / Draftea → Tarjetas → 'Más/Menos 3.5' → Over")
    if p_am_over45 >= UMBRAL_TARJ:
        ap("Tarjetas Amarillas", "✅ Over 4.5 amarillas (5+)", p_am_over45, f"{p_am_over45:.1f}% · {amarillas:.1f} esp.", "Playdoit / Draftea → Tarjetas → 'Más/Menos 4.5' → Over")
    if p_am_under25 >= UMBRAL_TARJ:
        ap("Tarjetas Amarillas", "✅ Under 2.5 amarillas (máx 2)", p_am_under25, f"{p_am_under25:.1f}% de simulaciones", "Playdoit / Draftea → Tarjetas → 'Más/Menos 2.5' → Under")
    if p_am_under35 >= UMBRAL_TARJ:
        ap("Tarjetas Amarillas", "✅ Under 3.5 amarillas (máx 3)", p_am_under35, f"{p_am_under35:.1f}% de simulaciones", "Playdoit / Draftea → Tarjetas → 'Más/Menos 3.5' → Under")
    if p_c_over65 >= UMBRAL_CORN and corners_esp >= 7:
        ap("Córners", "✅ Over 6.5 córners (7+)", p_c_over65, f"{p_c_over65:.1f}% · {corners_esp:.1f} esp.", "Playdoit / Draftea → Esquinas → 'Más/Menos 6.5' → Over")
    if p_c_over75 >= UMBRAL_CORN and corners_esp >= 8:
        ap("Córners", "✅ Over 7.5 córners (8+)", p_c_over75, f"{p_c_over75:.1f}% · {corners_esp:.1f} esp.", "Playdoit / Draftea → Esquinas → 'Más/Menos 7.5' → Over")
    if p_c_over85 >= UMBRAL_CORN:
        ap("Córners", "✅ Over 8.5 córners (9+)", p_c_over85, f"{p_c_over85:.1f}% · {corners_esp:.1f} esp.", "Playdoit / Draftea → Esquinas → 'Más/Menos 8.5' → Over")
    if p_c_over95 >= UMBRAL_CORN:
        ap("Córners", "✅ Over 9.5 córners (10+)", p_c_over95, f"{p_c_over95:.1f}% · {corners_esp:.1f} esp.", "Playdoit / Draftea → Esquinas → 'Más/Menos 9.5' → Over")
    if p_c_under85 >= UMBRAL_CORN:
        ap("Córners", "✅ Under 8.5 córners (máx 8)", p_c_under85, f"{p_c_under85:.1f}% · {corners_esp:.1f} esp.", "Playdoit / Draftea → Esquinas → 'Más/Menos 8.5' → Under")
    if p_c_under75 >= UMBRAL_CORN:
        ap("Córners", "✅ Under 7.5 córners (máx 7)", p_c_under75, f"{p_c_under75:.1f}% · {corners_esp:.1f} esp.", "Playdoit / Draftea → Esquinas → 'Más/Menos 7.5' → Under")

    apuestas.sort(key=lambda x: x["confianza"], reverse=True)

    # ── Filtro: solo la mejor apuesta por categoría ───────────────────────
    # Para Over tarjetas: solo la línea más alta que supere el umbral
    # Para Over córners: solo la línea más alta que supere el umbral
    # Para Under tarjetas/córners: solo la línea más baja (más fácil de cumplir)
    # Para Total Goles Over: solo la línea más alta
    filtradas = []
    categorias_vistas = set()

    for a in apuestas:
        merc = a["mercado"]
        sel  = a["seleccion"].lower()

        # Definir clave de categoría
        if merc == "Tarjetas Amarillas":
            if "over" in sel:
                cat = "am_over"   # solo la más alta (ya viene ordenada por confianza → línea más alta primero si empatan)
            elif "under" in sel:
                cat = "am_under"
            else:
                cat = merc
        elif merc == "Córners":
            if "over" in sel:
                cat = "co_over"
            elif "under" in sel:
                cat = "co_under"
            else:
                cat = merc
        elif merc == "Total Goles":
            if "over" in sel:
                cat = "goles_over"
            elif "under" in sel:
                cat = "goles_under"
            else:
                cat = merc
        else:
            cat = merc  # Resultado, Doble Oportunidad, Ambos Marcan — se muestran todos

        if cat not in categorias_vistas:
            categorias_vistas.add(cat)
            filtradas.append(a)

    return filtradas


def tag(cls, txt):
    return f'<span class="tag {cls}">{txt}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Mundial 2026 · Predictor</div>
  <div class="hero-sub">Monte Carlo · 10,000,000 simulaciones · ELO + H2H + Clima + Altitud + Árbitro</div>
</div>""", unsafe_allow_html=True)

API_KEY = None
ODDS_KEY = None
try:
    API_KEY  = st.secrets["RAPIDAPI_KEY"]
    ODDS_KEY = API_KEY
    FD_TOKEN    = st.secrets.get("FD_TOKEN", None)
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", None)
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", None)
except Exception:
    API_KEY  = os.environ.get("RAPIDAPI_KEY", None)
    ODDS_KEY = API_KEY
    FD_TOKEN    = os.environ.get("FD_TOKEN", None)
    SUPABASE_URL = os.environ.get("SUPABASE_URL", None)
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", None)

if API_KEY and API_DISPONIBLE:
    try:
        with st.spinner("🔄 Sincronizando resultados en tiempo real..."):
            actualizaciones = sincronizar_resultados(API_KEY, PARTIDOS)
            if actualizaciones:
                PARTIDOS_ACTUALIZADOS = []
                for p in PARTIDOS:
                    ea, eb = p[0], p[1]
                    if (ea, eb) in actualizaciones and p[4] is None:
                        PARTIDOS_ACTUALIZADOS.append((ea, eb, p[2], p[3], actualizaciones[(ea, eb)], p[5]))
                    else:
                        PARTIDOS_ACTUALIZADOS.append(p)
                PARTIDOS = PARTIDOS_ACTUALIZADOS
    except Exception:
        pass

from datetime import date as _date, datetime as _dt_hoy, timezone, timedelta as _td
_tz_mx = timezone(_td(hours=-6))
_ahora_mx = _dt_hoy.now(_tz_mx)
_hoy_fecha = _ahora_mx.strftime("%Y-%m-%d")

partidos_hoy = []
for _p in PARTIDOS:
    if _p[4] is not None or str(_p[0]).startswith('TBD'):
        continue
    _horario = HORARIOS_PARTIDO.get((_p[0], _p[1])) or HORARIOS_PARTIDO.get((_p[1], _p[0]), "")
    if not _horario or _horario[:10] != _hoy_fecha:
        continue
    try:
        _inicio = _dt_hoy.strptime(_horario, "%Y-%m-%d %H:%M").replace(tzinfo=_tz_mx)
        if _ahora_mx > _inicio + _td(hours=2, minutes=15):
            continue
    except Exception:
        pass
    partidos_hoy.append(_p)

if partidos_hoy:
    with st.expander("🎰 APUESTAS MÁS FUERTES DE HOY — Click para ver", expanded=False):
        st.markdown('<div style="font-size:0.7rem;color:#f0c040;letter-spacing:1px;margin-bottom:1rem">Simulación automática · Solo señales con confianza ALTA</div>', unsafe_allow_html=True)
        partidos_con_apuestas = []
        total_apuestas = 0
        for p in partidos_hoy:
            ea_d, eb_d, gr_d, sede_d, _, arb_d = p
            horario_p = HORARIOS_PARTIDO.get((ea_d, eb_d)) or HORARIOS_PARTIDO.get((eb_d, ea_d), "")
            hora_str = horario_p[11:] if horario_p else ""
            try:
                r_d = simular(ea_d, eb_d, sede_d, arbitro=arb_d, n=500_000)
                r_d["goles_totales_esperados"] = r_d["goles_a"] + r_d["goles_b"]
                sugs_d = [s for s in analizar_apuestas(ea_d, eb_d, r_d) if s["nivel"] == "ALTA"]
                if sugs_d:
                    partidos_con_apuestas.append({"ea": ea_d, "eb": eb_d, "grupo": gr_d, "hora": hora_str, "apuestas": sugs_d})
                    total_apuestas += len(sugs_d)
                    # ── Guardar en Supabase ────────────────────────────────
                    _guardar_apuestas_supabase(ea_d, eb_d, gr_d, sugs_d, p[4])
            except Exception:
                continue
        if not partidos_con_apuestas:
            st.info("Hoy no hay señales de confianza ALTA. El modelo es conservador.")
        else:
            st.markdown(f'<div style="font-size:0.75rem;color:#4ade80;margin-bottom:1rem">✓ {total_apuestas} apuesta(s) en {len(partidos_con_apuestas)} partido(s) de hoy</div>', unsafe_allow_html=True)
            for pd_item in partidos_con_apuestas:
                ea_d, eb_d = pd_item["ea"], pd_item["eb"]
                _hora_html = f'<span style="font-size:0.7rem;color:#6677aa">⏰ {pd_item["hora"]}h</span>' if pd_item["hora"] else ""
                st.markdown(f'<div style="display:flex;align-items:center;gap:0.5rem;margin:1rem 0 0.5rem;padding-bottom:0.4rem;border-bottom:1px solid #1e2d45"><span style="font-size:0.6rem;background:#2a1a00;color:#f0c040;border:1px solid #5a3a00;border-radius:20px;padding:2px 8px">Grupo {pd_item["grupo"]}</span><span style="font-size:0.85rem;color:#e8eaf0;font-weight:600">{flag_img(ea_d,20)} {ea_d} vs {flag_img(eb_d,20)} {eb_d}</span>{_hora_html}</div>', unsafe_allow_html=True)
                cols_ap = st.columns(min(len(pd_item["apuestas"]), 3))
                for i_ap, ap_d in enumerate(pd_item["apuestas"]):
                    with cols_ap[i_ap % 3]:
                        conf_d = min(ap_d["confianza"], 99)
                        st.markdown(f'<div style="background:#0d2818;border:1px solid #2d6b45;border-radius:10px;padding:0.75rem;margin-bottom:0.5rem"><div style="font-size:0.55rem;color:#6677aa;letter-spacing:2px;text-transform:uppercase">{ap_d["mercado"]}</div><div style="font-size:0.9rem;color:#e8eaf0;margin:0.2rem 0;font-weight:600">{ap_d["seleccion"]}</div><div style="background:#1e2d45;border-radius:3px;height:4px;margin:0.3rem 0"><div style="width:{conf_d:.0f}%;height:4px;border-radius:3px;background:linear-gradient(90deg,#3b82f6,#4ade80)"></div></div><div style="font-size:0.65rem;color:#4ade80;font-weight:600">{conf_d:.0f}% confianza</div><div style="font-size:0.58rem;color:#4a5568;margin-top:0.2rem">📱 {ap_d["donde"]}</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.65rem;color:#4a5568;padding-top:0.5rem;border-top:1px solid #1e2d45">⚠️ Solo informativo · Apuesta responsablemente</div>', unsafe_allow_html=True)

import json as _json_auto
from datetime import datetime as _dt_auto, timezone as _tz_auto, timedelta as _td_auto
_tz_mx_auto = _tz_auto(_td_auto(hours=-6))
_hoy_auto = _dt_auto.now(_tz_mx_auto).strftime("%Y-%m-%d")
_archivo_preds = "predicciones_mundial.json"
try:
    with open(_archivo_preds, "r", encoding="utf-8") as _f:
        _todas_preds = _json_auto.load(_f)
except Exception:
    _todas_preds = {}
if "predicciones_guardadas" not in st.session_state:
    st.session_state["predicciones_guardadas"] = dict(_todas_preds)
_clave_dia = f"sims_del_dia_{_hoy_auto}"
_ya_simule_hoy = st.session_state.get(_clave_dia, False) or _clave_dia in _todas_preds
_partidos_hoy_auto = []
for _p in PARTIDOS:
    if _p[4] is not None or str(_p[0]).startswith('TBD'):
        continue
    _hor = HORARIOS_PARTIDO.get((_p[0], _p[1])) or HORARIOS_PARTIDO.get((_p[1], _p[0]), "")
    if not _hor or _hor[:10] != _hoy_auto:
        continue
    try:
        _ini = _dt_auto.strptime(_hor, "%Y-%m-%d %H:%M").replace(tzinfo=_tz_mx_auto)
        if _dt_auto.now(_tz_mx_auto) > _ini + _td_auto(hours=2, minutes=15):
            continue
    except Exception:
        pass
    _partidos_hoy_auto.append(_p)

if _partidos_hoy_auto and not _ya_simule_hoy:
    _placeholder = st.empty()
    _placeholder.info(f"⚡ Calculando predicciones del día ({len(_partidos_hoy_auto)} partidos)...")
    _nuevas_sb = 0
    if SUPABASE_DISPONIBLE and SUPABASE_URL and SUPABASE_KEY:
        try:
            _nuevas_sb = simular_y_guardar_dia(SUPABASE_URL, SUPABASE_KEY, _partidos_hoy_auto, simular, HORARIOS_PARTIDO)
            _preds_sb = cargar_todas_predicciones(SUPABASE_URL, SUPABASE_KEY)
            st.session_state["predicciones_guardadas"].update(_preds_sb)
            _todas_preds.update(_preds_sb)
        except Exception:
            _nuevas_sb = 0
    if _nuevas_sb == 0:
        for _p in _partidos_hoy_auto:
            _ea, _eb, _gr, _sede, _, _arb = _p
            _pred_key = f"pred_{_ea}_{_eb}".replace(" ", "_")
            if _pred_key not in _todas_preds:
                try:
                    _r = simular(_ea, _eb, _sede, arbitro=_arb, n=500_000)
                    _pred_nueva = {"ea": _ea, "eb": _eb, "grupo": _gr, "guardada_en": _dt_auto.now(_tz_mx_auto).strftime("%Y-%m-%d %H:%M"), "tipo": "automatica", "prob_a": round(_r["prob_a"], 1), "prob_emp": round(_r["prob_emp"], 1), "prob_b": round(_r["prob_b"], 1), "goles_a_esp": round(_r["goles_a"], 2), "goles_b_esp": round(_r["goles_b"], 2), "favorito": _ea if _r["prob_a"] > _r["prob_b"] else _eb, "prob_fav": round(max(_r["prob_a"], _r["prob_b"]), 1), "modelo": _r.get("modelo", "Manual"), "arbitro": _arb or "desconocido", "lam_a": round(_r.get("lam_a", 0), 3), "lam_b": round(_r.get("lam_b", 0), 3), "fecha_partido": _hoy_auto}
                    _todas_preds[_pred_key] = _pred_nueva
                    st.session_state["predicciones_guardadas"][_pred_key] = _pred_nueva
                    _nuevas_sb += 1
                except Exception:
                    pass
        try:
            with open(_archivo_preds, "w", encoding="utf-8") as _f:
                _json_auto.dump(_todas_preds, _f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    _todas_preds[_clave_dia] = {"fecha": _hoy_auto, "partidos": len(_partidos_hoy_auto)}
    st.session_state[_clave_dia] = True
    if SUPABASE_DISPONIBLE and SUPABASE_URL and SUPABASE_KEY:
        try:
            guardar_apuestas_dia(SUPABASE_URL, SUPABASE_KEY, _partidos_hoy_auto, simular, analizar_apuestas, HORARIOS_PARTIDO)
            _pts_term = [p for p in PARTIDOS if p[4] is not None]
            if _pts_term:
                actualizar_aciertos(SUPABASE_URL, SUPABASE_KEY, _pts_term)
        except Exception:
            pass
    _placeholder.empty()
    if _nuevas_sb > 0:
        st.toast(f"📊 {_nuevas_sb} predicciones guardadas", icon="✅")


# ══════════════════════════════════════════════════════════════════════════════
# DATOS REALES POR PARTIDO — tarjetas y córners reales de Sofascore
# Ale agrega esto cada noche con los datos del día.
# Formato: { "ea_eb": {"am": int, "co": int} }
# El sistema usa estos datos para evaluar apuestas de tarjetas y córners.
# ══════════════════════════════════════════════════════════════════════════════
DATOS_REALES = {
    # Jornada 1
    "Mexico_Sudafrica":              {"am": 3,  "co": 8},
    "Corea del Sur_Chequia":         {"am": 4,  "co": 7},
    "Estados Unidos_Paraguay":       {"am": 5,  "co": 9},
    "Brasil_Marruecos":              {"am": 3,  "co": 7},
    "Haiti_Escocia":                 {"am": 2,  "co": 5},
    "Australia_Turquia":             {"am": 3,  "co": 8},
    "Alemania_Curazao":              {"am": 2,  "co": 6},
    "Costa de Marfil_Ecuador":       {"am": 3,  "co": 7},
    "Paises Bajos_Japon":            {"am": 4,  "co": 9},
    "Suecia_Tunez":                  {"am": 1,  "co": 7},
    "Belgica_Egipto":                {"am": 3,  "co": 8},
    "Iran_Nueva Zelanda":            {"am": 4,  "co": 7},
    "Espana_Cabo Verde":             {"am": 2,  "co": 8},
    "Arabia Saudi_Uruguay":          {"am": 3,  "co": 7},
    "Francia_Senegal":               {"am": 3,  "co": 6},
    "Irak_Noruega":                  {"am": 2,  "co": 6},
    "Argentina_Algeria":             {"am": 0,  "co": 5},
    "Austria_Jordania":              {"am": 4,  "co": 6},
    "Portugal_RD Congo":             {"am": 3,  "co": 8},
    "Uzbekistan_Colombia":           {"am": 4,  "co": 7},
    "Inglaterra_Croacia":            {"am": 3,  "co": 9},
    "Ghana_Panama":                  {"am": 3,  "co": 7},
    # Jornada 2
    "Chequia_Sudafrica":             {"am": 2,  "co": 7},
    "Mexico_Corea del Sur":          {"am": 2,  "co": 8},
    "Suiza_Bosnia y Herzegovina":    {"am": 3,  "co": 8},
    "Canada_Catar":                  {"am": 3,  "co": 7},
    "Escocia_Marruecos":             {"am": 2,  "co": 6},
    "Brasil_Haiti":                  {"am": 1,  "co": 8},
    "Estados Unidos_Australia":      {"am": 1,  "co": 7},
    "Turquia_Paraguay":              {"am": 4,  "co": 8},
    "Alemania_Costa de Marfil":      {"am": 3,  "co": 8},
    "Ecuador_Curazao":               {"am": 2,  "co": 6},
    "Paises Bajos_Suecia":           {"am": 3,  "co": 9},
    "Tunez_Japon":                   {"am": 3,  "co": 7},
    "Belgica_Iran":                  {"am": 2,  "co": 7},
    "Nueva Zelanda_Egipto":          {"am": 3,  "co": 6},
    "Espana_Arabia Saudita":         {"am": 2,  "co": 7},   # ESP 0 + KSA 2
    "Cabo Verde_Uruguay":            {"am": 4,  "co": 8},
    "Francia_Irak":                  {"am": 1,  "co": 6},   # FRA 0 + IRQ 1
    "Senegal_Noruega":               {"am": 0,  "co": 9},   # sin tarjetas reportadas
    "Argentina_Austria":             {"am": 4,  "co": 4},   # ARG 2 + AUT 2
    "Algeria_Jordania":              {"am": 2,  "co": 11},  # ALG 1 + JOR 1, 10 corners JOR
    "Portugal_Uzbekistan":           {"am": 0,  "co": 7},   # Jalal Jayed permisivo
    "Inglaterra_Ghana":              {"am": 2,  "co": 11},  # 9 ENG + 2 GHA
    "Croacia_Panama":                {"am": 2,  "co": 9},   # 7 PAN + 2 CRO
    "Colombia_RD Congo":             {"am": 3,  "co": 9},   # COL 2 + RDC 1
    # Jornada 3
    "Bosnia y Herzegovina_Catar":    {"am": 2,  "co": 10},  # BIH 1 + QAT 1
    "Suiza_Canada":                  {"am": 3,  "co": 9},   # SUI 1 + CAN 2
    # Jornada 3 — 24 junio tarde/noche
    "Marruecos_Haiti":               {"am": 3,  "co": 10},  # MAR 0 + HAI 3 | 9 MAR + 1 HAI
    "Escocia_Brasil":                {"am": 3,  "co": 13},  # ESC 1 + BRA 2 | 7 ESC + 6 BRA
    "Mexico_Chequia":                {"am": 1,  "co": 6},   # MEX 1 + CHE 0 | 1 MEX + 5 CHE
    "Sudafrica_Corea del Sur":       {"am": 2,  "co": 10},  # SUF 1 + COR 1 | 4 SUF + 6 COR
    # Jornada 3 — 25 junio
    "Curazao_Costa de Marfil":       {"am": 3,  "co": 10},  # CUR 2 + CIV 1 | 4 CUR + 6 CIV
    "Ecuador_Alemania":              {"am": 4,  "co": 5},   # ECU 3 + ALE 1 | 3 ECU + 2 ALE
    "Tunez_Paises Bajos":            {"am": 0,  "co": 10},  # TUN 0 + PB 0  | 4 TUN + 6 PB
    "Japon_Suecia":                  {"am": 3,  "co": 10},  # JAP 1 + SUE 2 | 2 JAP + 8 SUE
    # J3 noche 25/26 junio
    "Paraguay_Australia":            {"am": 2,  "co": 4},   # 1 PAR + 3 AUS
    "Turquia_Estados Unidos":        {"am": 1,  "co": 11},  # 2 TUR + 9 EEUU
    "Noruega_Francia":               {"am": 2,  "co": 9},   # 4 NOR + 5 FRA
    "Senegal_Irak":                  {"am": 4,  "co": 15},  # 12 SEN + 3 IRQ
    "Cabo Verde_Arabia Saudita":     {"am": 4,  "co": 6},   # 4 CPV + 2 KSA
    "Uruguay_Espana":                {"am": 4,  "co": 7},   # 1 URU + 6 ESP
    # J3 grupos G y L — 26/27 junio
    "Croacia_Ghana":                 {"am": 2,  "co": 5},   # 3 CRO + 2 GHA
    "Nueva Zelanda_Belgica":         {"am": 2,  "co": 13},  # 5 NZ + 8 BEL
    "Egipto_Iran":                   {"am": 7,  "co": 10},  # 8 EGY + 2 IRN
    "Panama_Inglaterra":             {"am": 3,  "co": 10},  # 3 PAN + 7 ENG
    # J3 grupos J y K — 27 junio
    "Algeria_Austria":               {"am": 1,  "co": 3},   # 0 ALG + 1 AUT | 0 ALG + 3 AUT
    "Jordania_Argentina":            {"am": 2,  "co": 5},   # estimado
    "Colombia_Portugal":             {"am": 1,  "co": 7},   # 1 COL + 0 POR | 5 COL + 2 POR
    "RD Congo_Uzbekistan":           {"am": 5,  "co": 6},   # 3 RDC + 2 UZB | 2 RDC + 4 UZB
    # Dieciseisavos de final
    "Sudafrica_Canada":              {"am": 2,  "co": 5},   # 0 SUF + 2 CAN | 1 SUF + 4 CAN
}


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-ACTUALIZACIÓN DE ACIERTOS — corre al inicio de cada sesión
# Evalúa automáticamente todas las apuestas de partidos ya jugados
# usando DATOS_REALES para tarjetas y córners.
# ══════════════════════════════════════════════════════════════════════════════
def _auto_actualizar_aciertos():
    """
    Corre al inicio de cada sesión.
    Re-evalúa todas las apuestas de partidos con resultado.
    Usa DATOS_REALES para tarjetas y córners.
    """
    if not SUPABASE_DISPONIBLE or not SUPABASE_URL or not SUPABASE_KEY:
        return

    partidos_jugados = [p for p in PARTIDOS
                        if p[4] is not None and not str(p[0]).startswith('TBD')]
    if not partidos_jugados:
        return

    # Guard: resetea cuando cambia nº partidos jugados O cuando se agregan datos reales nuevos
    _n_jugados = len(partidos_jugados)
    _n_datos = len(DATOS_REALES)  # cambia cada vez que agrego datos de Sofascore
    _clave = f"aciertos_{_hoy_auto}_n{_n_jugados}_d{_n_datos}"
    if st.session_state.get(_clave):
        return

    # Marcar ANTES de procesar para evitar loops con st.rerun()
    st.session_state[_clave] = True

    try:
        import requests as _req
        import urllib.parse as _up

        def _hdrs(k):
            return {"apikey": k, "Authorization": f"Bearer {k}",
                    "Content-Type": "application/json", "Prefer": "return=minimal"}

        actualizadas = 0

        # Obtener TODAS las apuestas de una vez (más eficiente que por partido)
        try:
            r_all = _req.get(
                f"{SUPABASE_URL}/rest/v1/apuestas_historial",
                headers={**_hdrs(SUPABASE_KEY), "Prefer": ""},
                params={"select": "*", "limit": 1000},
                timeout=15
            )
            todas_apuestas = r_all.json() if r_all.status_code == 200 else []
        except Exception:
            todas_apuestas = []

        if not todas_apuestas:
            return

        # Crear mapa de resultados para lookup rápido
        mapa_resultados = {}
        for p in partidos_jugados:
            ea, eb = p[0], p[1]
            mapa_resultados[(ea, eb)] = p[4]

        for ap in todas_apuestas:
            ap_ea = ap.get("ea", "")
            ap_eb = ap.get("eb", "")

            # Saltar TBDs
            if str(ap_ea).startswith("TBD") or str(ap_eb).startswith("TBD"):
                continue

            # Buscar resultado del partido
            resultado = mapa_resultados.get((ap_ea, ap_eb))
            if resultado is None:
                continue  # partido sin resultado todavía

            ga, gb = resultado
            datos = DATOS_REALES.get(f"{ap_ea}_{ap_eb}", {})
            am_reales = datos.get("am")
            co_reales = datos.get("co")

            from supabase_preds import _evaluar_acierto
            nuevo_acierto = _evaluar_acierto(
                ap, ga, gb,
                am_reales=am_reales, co_reales=co_reales
            )

            if nuevo_acierto is None:
                continue  # sin datos suficientes

            acierto_actual = ap.get("acierto")
            if acierto_actual is None or acierto_actual != nuevo_acierto:
                try:
                    _req.patch(
                        f"{SUPABASE_URL}/rest/v1/apuestas_historial",
                        headers={**_hdrs(SUPABASE_KEY), "Prefer": ""},
                        params={"id": f"eq.{ap['id']}"},
                        json={
                            "acierto":          nuevo_acierto,
                            "goles_a":          ga,
                            "goles_b":          gb,
                            "resultado_real":   f"{ga}-{gb}",
                            "amarillas_reales": am_reales,
                            "corners_reales":   co_reales,
                        },
                        timeout=8
                    )
                    actualizadas += 1
                except Exception:
                    continue

        if actualizadas > 0:
            # Limpiar cache de Streamlit para que tab_hist_ap cargue datos frescos
            st.cache_data.clear()
            st.rerun()

    except Exception:
        pass


# Ejecutar silenciosamente al cargar
_auto_actualizar_aciertos()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — Guardar apuestas en Supabase (reutilizable desde cualquier tab)
# ══════════════════════════════════════════════════════════════════════════════
def _guardar_apuestas_supabase(ea, eb, grupo, sugerencias, res=None):
    """
    Guarda todas las apuestas de nivel ALTA en Supabase.
    Si el partido ya terminó (res != None), evalúa el acierto inmediatamente.
    Si ya existe la apuesta y no tiene resultado, actualiza confianza/selección.
    """
    if not (SUPABASE_DISPONIBLE and SUPABASE_URL and SUPABASE_KEY):
        return 0

    import requests as _rq
    from datetime import datetime as _dt, timezone as _tz2, timedelta as _td
    _ahora = _dt.now(_tz2(_td(hours=-6)))
    _hor = HORARIOS_PARTIDO.get((ea, eb)) or HORARIOS_PARTIDO.get((eb, ea), "")
    _fecha = _hor[:10] if _hor else _ahora.strftime("%Y-%m-%d")
    _hdrs = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
             "Content-Type": "application/json", "Prefer": "return=minimal"}
    _ga, _gb = (res[0], res[1]) if res is not None else (None, None)
    guardadas = 0

    for _i, _s in enumerate(sugerencias):
        if _s["nivel"] != "ALTA":
            continue
        _apid = f"ap_{ea}_{eb}_{_i}_{_fecha}".replace(" ", "_")
        _acierto = None
        if _ga is not None:
            from supabase_preds import _evaluar_acierto
            _dd = DATOS_REALES.get(f"{ea}_{eb}", {})
            _acierto = _evaluar_acierto(
                {"seleccion": _s["seleccion"].replace("✅ ", ""),
                 "mercado": _s["mercado"], "ea": ea, "eb": eb},
                _ga, _gb,
                am_reales=_dd.get("am"), co_reales=_dd.get("co")
            )
        _ap = {
            "id": _apid, "ea": ea, "eb": eb, "grupo": grupo,
            "fecha_partido": _fecha,
            "guardada_en": _ahora.strftime("%Y-%m-%d %H:%M"),
            "mercado": _s["mercado"],
            "seleccion": _s["seleccion"].replace("✅ ", ""),
            "confianza": round(_s["confianza"], 1),
            "donde": _s.get("donde", "Playdoit / Draftea"),
            "resultado_real": f"{_ga}-{_gb}" if _ga is not None else None,
            "goles_a": _ga, "goles_b": _gb, "acierto": _acierto,
        }
        try:
            _chk = _rq.get(
                f"{SUPABASE_URL}/rest/v1/apuestas_historial",
                headers={**_hdrs, "Prefer": ""},
                params={"id": f"eq.{_apid}", "select": "id,acierto"},
                timeout=5
            )
            if _chk.status_code == 200 and not _chk.json():
                # No existe — insertar
                _rq.post(f"{SUPABASE_URL}/rest/v1/apuestas_historial",
                         headers=_hdrs, json=_ap, timeout=5)
                guardadas += 1
            elif _chk.status_code == 200 and _chk.json():
                # Ya existe — actualizar si no tiene resultado aún
                if _chk.json()[0].get("acierto") is None:
                    _rq.patch(
                        f"{SUPABASE_URL}/rest/v1/apuestas_historial",
                        headers={**_hdrs, "Prefer": ""},
                        params={"id": f"eq.{_apid}"},
                        json={"confianza": _ap["confianza"],
                              "seleccion": _ap["seleccion"],
                              "mercado":   _ap["mercado"],
                              "resultado_real": _ap["resultado_real"],
                              "goles_a": _ga, "goles_b": _gb,
                              "acierto": _acierto},
                        timeout=5
                    )
        except Exception:
            pass

    return guardadas

tab_pred, tab_res, tab_apuestas, tab_hist, tab_hist_ap, tab_info = st.tabs(["🎯 Predictor", "📊 Resultados reales", "🎰 Apuestas", "📈 Historial", "🎲 Apuestas Hist.", "⚙️ Modelo"])

# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    col_izq, col_der = st.columns([1, 2.5], gap="large")
    with col_izq:
        st.markdown("#### Elige el partido")
        partidos_pendientes = [p for p in PARTIDOS if p[4] is None and not str(p[0]).startswith('TBD')]
        if partidos_pendientes:
            if st.button("📅 Partidos de hoy", use_container_width=True):
                st.session_state["filtro_hoy"] = True
            elif "filtro_hoy" not in st.session_state:
                st.session_state["filtro_hoy"] = False
        else:
            st.session_state["filtro_hoy"] = False
        grupos = sorted(set(p[2] for p in PARTIDOS))
        grupo_sel = st.selectbox("Grupo", ["Todos"] + [f"Grupo {g}" for g in grupos])
        estado_sel = st.radio("Mostrar", ["Todos", "Por jugarse", "Ya jugados"], horizontal=True, label_visibility="collapsed")
        filtrados = PARTIDOS
        if st.session_state.get("filtro_hoy", False):
            filtrados = partidos_pendientes
        else:
            if grupo_sel != "Todos":
                letra = grupo_sel.replace("Grupo ", "")
                filtrados = [p for p in filtrados if p[2] == letra]
            if estado_sel == "Por jugarse":
                filtrados = [p for p in filtrados if p[4] is None]
            elif estado_sel == "Ya jugados":
                filtrados = [p for p in filtrados if p[4] is not None]
        if not filtrados:
            st.info("Sin partidos en este filtro.")
            idx_sel = None
        else:
            opciones = {}
            for p in filtrados:
                lbl = f"{flag(p[0])} {p[0]}  vs  {flag(p[1])} {p[1]}  [G{p[2]}]"
                opciones[lbl] = PARTIDOS.index(p)
            lbl_sel = st.selectbox("Partido", list(opciones.keys()), label_visibility="collapsed")
            idx_sel = opciones[lbl_sel]
        st.markdown("---")
        n_sims = 10_000_000
        st.markdown('<div style="font-size:0.65rem;color:#6677aa;letter-spacing:1px;margin-bottom:0.5rem">⚡ 10,000,000 simulaciones automáticas</div>', unsafe_allow_html=True)
        btn = st.button("⚽ Simular partido")

    with col_der:
        if idx_sel is None:
            st.markdown('<div style="text-align:center;padding:4rem;color:#4a5568"><div style="font-size:3rem">⚽</div><div style="margin-top:0.5rem">Selecciona un partido para comenzar</div></div>', unsafe_allow_html=True)
        else:
            ea, eb, grupo, sede, resultado_real, arbitro = PARTIDOS[idx_sel]
            alt = ALTITUD.get(sede, 0)
            estado_tag = tag("tag-played", "✓ Jugado") if resultado_real else tag("tag-pending", "⏳ Por jugarse")
            temp_c, hum = CLIMA.get(sede, (25, 65))
            arb_txt = arbitro if arbitro else "Por confirmar"
            st.markdown(f'{tag("tag-group", f"Grupo {grupo}")} {estado_tag}<div style="font-size:0.75rem;color:#6677aa;margin:0.5rem 0 1rem">📍 {sede} &nbsp;·&nbsp; ⛰️ {alt:,} m &nbsp;·&nbsp; 🌡️ {temp_c}°C &nbsp;·&nbsp; 💧 {hum}% &nbsp;·&nbsp; 🧑‍⚖️ {arb_txt}</div>', unsafe_allow_html=True)

            if resultado_real:
                ga_r, gb_r = resultado_real
                ganador_txt = (f"🏆 Ganó {ea}" if ga_r > gb_r else f"🏆 Ganó {eb}" if gb_r > ga_r else "🤝 Empate")
                st.markdown(f'<div class="real-result"><div style="font-size:0.6rem;color:#4ade80;letter-spacing:2px;margin-bottom:0.3rem">RESULTADO REAL</div><div style="display:flex;align-items:center;justify-content:center;gap:2rem"><div style="text-align:right">{flag_img(ea, 48)}<div style="font-size:0.75rem;color:#aabbcc;margin-top:0.2rem">{ea}</div></div><div class="real-score">{ga_r} – {gb_r}</div><div style="text-align:left">{flag_img(eb, 48)}<div style="font-size:0.75rem;color:#aabbcc;margin-top:0.2rem">{eb}</div></div></div><div style="font-size:0.75rem;color:#4ade80;margin-top:0.4rem">{ganador_txt}</div></div>', unsafe_allow_html=True)

            if btn or resultado_real:
                with st.spinner(f"Simulando {n_sims:,} partidos..."):
                    r = simular(ea, eb, sede, arbitro=arbitro, n=n_sims)

                if resultado_real is None:
                    try:
                        import json as _json
                        from datetime import datetime as _dtnow, timezone as _tzj, timedelta as _tdj
                        _pred_key = f"pred_{ea}_{eb}".replace(" ", "_")
                        try:
                            with open(_archivo_preds, "r", encoding="utf-8") as _f:
                                _todas_preds_local = _json.load(_f)
                        except Exception:
                            _todas_preds_local = {}
                        if _pred_key not in _todas_preds_local:
                            _nueva_pred = {"ea": ea, "eb": eb, "grupo": grupo, "guardada_en": _dtnow.now(_tzj(_tdj(hours=-6))).strftime("%Y-%m-%d %H:%M"), "prob_a": round(r["prob_a"], 1), "prob_emp": round(r["prob_emp"], 1), "prob_b": round(r["prob_b"], 1), "goles_a_esp": round(r["goles_a"], 2), "goles_b_esp": round(r["goles_b"], 2), "favorito": ea if r["prob_a"] > r["prob_b"] else eb, "prob_fav": round(max(r["prob_a"], r["prob_b"]), 1), "modelo": r.get("modelo", "Manual"), "arbitro": arbitro or "desconocido", "lam_a": round(r.get("lam_a", 0), 3), "lam_b": round(r.get("lam_b", 0), 3)}
                            _todas_preds_local[_pred_key] = _nueva_pred
                            with open(_archivo_preds, "w", encoding="utf-8") as _f:
                                _json.dump(_todas_preds_local, _f, ensure_ascii=False, indent=2)
                            if "predicciones_guardadas" not in st.session_state:
                                st.session_state["predicciones_guardadas"] = {}
                            st.session_state["predicciones_guardadas"][_pred_key] = _nueva_pred
                            st.toast(f"📊 Predicción guardada: {ea} {r['prob_a']:.0f}%", icon="✅")
                    except Exception:
                        pass

                pa, pd_, pb = r["prob_a"], r["prob_emp"], r["prob_b"]
                st.markdown(f'<div style="font-size:0.6rem;color:#6677aa;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.2rem">Probabilidades — {n_sims:,} simulaciones</div><div class="prob-bar"><div class="bar-a" style="width:{pa:.1f}%"></div><div class="bar-draw" style="width:{pd_:.1f}%"></div><div class="bar-b" style="width:{pb:.1f}%"></div></div>', unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="result-box">{flag_img(ea, 64)}<div class="team-name">{ea}</div><div class="prob-pct">{pa:.1f}%</div><div class="prob-lbl">victoria</div><div class="goles-esp">{r["goles_a"]:.2f}</div><div class="prob-lbl">goles esp.</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="result-box result-box-draw"><div style="font-size:3.5rem;line-height:1.1">🤝</div><div class="team-name" style="color:#9ca3af">Empate</div><div class="prob-pct prob-pct-draw">{pd_:.1f}%</div><div class="prob-lbl">probabilidad</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="result-box result-box-b">{flag_img(eb, 64)}<div class="team-name">{eb}</div><div class="prob-pct prob-pct-b">{pb:.1f}%</div><div class="prob-lbl">victoria</div><div class="goles-esp">{r["goles_b"]:.2f}</div><div class="prob-lbl">goles esp.</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                cm1, cm2 = st.columns(2)
                with cm1:
                    st.markdown('<div style="font-size:0.6rem;color:#6677aa;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem">Top 5 marcadores</div>', unsafe_allow_html=True)
                    badges = ""
                    for i, (marcador, cnt) in enumerate(r["top5"]):
                        pct = cnt / n_sims * 100
                        bg = "#1a1800" if i == 0 else "#1e2d45"
                        border = "#f0c040" if i == 0 else "#2a4060"
                        color = "#f0c040" if i == 0 else "#e8eaf0"
                        badges += f'<div style="display:inline-block;background:{bg};border:1px solid {border};border-radius:8px;padding:0.3rem 0.6rem;margin:0.15rem;text-align:center"><div style="font-family:Bebas Neue,sans-serif;font-size:1.1rem;color:{color}">{marcador[0]}–{marcador[1]}</div><div style="font-size:0.6rem;color:#6677aa">{pct:.1f}%</div></div>'
                    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:0.1rem">{badges}</div>', unsafe_allow_html=True)
                with cm2:
                    st.markdown('<div style="font-size:0.6rem;color:#6677aa;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem">Tarjetas esperadas</div>', unsafe_allow_html=True)
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        st.markdown(f'<div class="metric-box"><div><span class="card-y"></span></div><div class="metric-val">{r["amarillas"]}</div><div class="metric-lbl">Amarillas</div></div>', unsafe_allow_html=True)
                    with tc2:
                        st.markdown(f'<div class="metric-box"><div><span class="card-r"></span></div><div class="metric-val">{r["rojas"]}</div><div class="metric-lbl">Rojas</div></div>', unsafe_allow_html=True)

                if ODDS_DISPONIBLE and ODDS_KEY:
                    mostrar_comparacion_odds(ea, eb, r, ODDS_KEY)

                # ── Clasificación (solo en R32 y fases eliminatorias) ─────
                _pd_item_sel = next((p for p in PARTIDOS
                    if (p[0]==ea and p[1]==eb) or (p[0]==eb and p[1]==ea)), None)
                _grupo_sel = _pd_item_sel[2] if _pd_item_sel else "?"
                if _grupo_sel in ("R32", "QF", "SF", "F", "3rd"):
                    # Calcular probabilidad de clasificación con extra time + penales
                    # En eliminación directa: gana → clasifica
                    # Empate → extra time (~50/50 con ligera ventaja al que iba ganando)
                    # En penales estadísticamente: 50% cada equipo pero con factor forma
                    _prob_pen_a = 50.0  # base penales
                    _prob_pen_b = 50.0
                    # Ajuste por ELO si hay diferencia grande
                    _elo_a = ELO.get(ea, 1500)
                    _elo_b = ELO.get(eb, 1500)
                    _diff_elo = _elo_a - _elo_b
                    _prob_pen_a = max(35, min(65, 50 + _diff_elo * 0.008))
                    _prob_pen_b = 100 - _prob_pen_a

                    # P(clasifica A) = P(gana 90min) + P(empate) * [P(gana ET≈55%) + P(empate ET≈45%) * P(pen_a)]
                    _p_clasifica_a = pa + pd_ * (0.55 * _prob_pen_a/100 + 0.45 * _prob_pen_a/100)
                    _p_clasifica_b = pb + pd_ * (0.55 * _prob_pen_b/100 + 0.45 * _prob_pen_b/100)

                    # Normalizar
                    _total = _p_clasifica_a + _p_clasifica_b
                    _p_clasifica_a = _p_clasifica_a / _total * 100
                    _p_clasifica_b = _p_clasifica_b / _total * 100

                    st.markdown("<div style='margin:1.2rem 0 0.4rem'>", unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,#0d1b2a,#1a2d45);border:1px solid #2a4a6b;'
                        f'border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.8rem">'
                        f'<div style="font-size:0.65rem;color:#4a90d9;letter-spacing:2px;text-transform:uppercase;'
                        f'margin-bottom:0.8rem">🏆 Probabilidad de clasificación — incluye prórroga y penales</div>'
                        f'<div style="display:flex;align-items:center;gap:1rem">'
                        f'<div style="text-align:center;flex:1">'
                        f'{flag_img(ea,40)}'
                        f'<div style="color:#e8eaf0;font-weight:700;font-size:0.9rem;margin-top:0.3rem">{ea}</div>'
                        f'<div style="font-size:2rem;font-weight:900;color:{"#4ade80" if _p_clasifica_a > _p_clasifica_b else "#94a3b8"};'
                        f'font-family:Bebas Neue,sans-serif">{_p_clasifica_a:.0f}%</div>'
                        f'</div>'
                        f'<div style="text-align:center;color:#4a5568;font-size:0.8rem">'
                        f'<div style="font-size:1.2rem">⚔️</div>'
                        f'<div style="font-size:0.65rem;margin-top:0.2rem">partido<br>único</div>'
                        f'</div>'
                        f'<div style="text-align:center;flex:1">'
                        f'{flag_img(eb,40)}'
                        f'<div style="color:#e8eaf0;font-weight:700;font-size:0.9rem;margin-top:0.3rem">{eb}</div>'
                        f'<div style="font-size:2rem;font-weight:900;color:{"#4ade80" if _p_clasifica_b > _p_clasifica_a else "#94a3b8"};'
                        f'font-family:Bebas Neue,sans-serif">{_p_clasifica_b:.0f}%</div>'
                        f'</div>'
                        f'</div>'
                        f'<div style="margin-top:0.6rem;background:#0a1628;border-radius:6px;height:8px;overflow:hidden">'
                        f'<div style="background:linear-gradient(90deg,#4ade80,#22d3ee);height:100%;width:{_p_clasifica_a:.1f}%"></div>'
                        f'</div>'
                        f'<div style="font-size:0.6rem;color:#4a5568;margin-top:0.3rem;text-align:center">'
                        f'90min + prórroga + penales · ELO {ea}: {_elo_a} vs {eb}: {_elo_b}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                bajas_activas = [e for e in [ea, eb] if e in BAJAS]
                _bajas_html = (" · ⚠️ Bajas: " + ", ".join(bajas_activas)) if bajas_activas else ""
                st.markdown(f'<div class="model-note">{r["modelo"]} · ELO: {r["elo_a"]} ({ea}) vs {r["elo_b"]} ({eb}) · λ_a={r["lam_a"]} · λ_b={r["lam_b"]} · Altitud: {r["alt"]:,} m · Árbitro: {r["arbitro"]} ({r["arbitro_am"]} T.A. / {r["arbitro_ro"]} T.R.) · Tarjetas: {r["fuente_tarj"]} · H2H: {r["h2h_desc"]}{_bajas_html}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                r["goles_totales_esperados"] = r["goles_a"] + r["goles_b"]
                sugs = analizar_apuestas(ea, eb, r)
                if sugs:
                    st.markdown('<div style="font-family:\'Bebas Neue\',sans-serif;font-size:1.3rem;letter-spacing:2px;color:#f0c040;margin-bottom:0.75rem">🎰 APUESTAS SUGERIDAS — PLAYDOIT / DRAFTEA</div>', unsafe_allow_html=True)
                    cols_ap = st.columns(min(len(sugs), 3))
                    for i_ap, ap in enumerate(sugs[:3]):
                        with cols_ap[i_ap]:
                            color_bg = "#0d2818" if ap["nivel"] == "ALTA" else "#0d1827"
                            color_br = "#2d6b45" if ap["nivel"] == "ALTA" else "#1e3a5f"
                            conf = min(ap["confianza"], 99)
                            st.markdown(f'<div style="background:{color_bg};border:1px solid {color_br};border-radius:10px;padding:0.9rem;height:100%"><div style="font-size:0.6rem;color:#6677aa;letter-spacing:2px;text-transform:uppercase">{ap["mercado"]}</div><div style="font-size:0.95rem;color:#e8eaf0;margin:0.3rem 0;font-weight:600">{ap["seleccion"]}</div><div style="background:#1e2d45;border-radius:3px;height:5px;margin:0.3rem 0"><div style="width:{conf:.0f}%;height:5px;border-radius:3px;background:linear-gradient(90deg,#3b82f6,#4ade80)"></div></div><div style="font-size:0.65rem;color:#4ade80">{conf:.0f}% confianza</div><div style="font-size:0.6rem;color:#4a5568;margin-top:0.3rem">{ap["nota"]}</div></div>', unsafe_allow_html=True)

                    def _filtrar_parlay(apuestas):
                        seleccionadas = []
                        mercados_usados = set()
                        tiene_resultado = False
                        tiene_doble_op  = False
                        gol_mercados_over = []
                        gol_mercados_under = []
                        for ap in sorted(apuestas, key=lambda x: x["confianza"], reverse=True):
                            sel, merc = ap["seleccion"], ap["mercado"]
                            if merc == "Total Goles":
                                if "Over" in sel: gol_mercados_over.append(ap)
                                elif "Under" in sel: gol_mercados_under.append(ap)
                                continue
                            if merc == "Resultado (1X2)":
                                if not tiene_resultado and not tiene_doble_op:
                                    seleccionadas.append(ap); tiene_resultado = True
                                continue
                            if merc == "Doble Oportunidad":
                                if not tiene_resultado and not tiene_doble_op:
                                    seleccionadas.append(ap); tiene_doble_op = True
                                continue
                            if merc not in mercados_usados:
                                seleccionadas.append(ap); mercados_usados.add(merc)
                        if gol_mercados_over:
                            def _nivel_over(a):
                                for n in ["3.5","2.5","1.5","0.5"]:
                                    if n in a["seleccion"]: return float(n)
                                return 0
                            seleccionadas.append(max(gol_mercados_over, key=_nivel_over))
                        elif gol_mercados_under:
                            seleccionadas.append(gol_mercados_under[0])
                        return [a for a in seleccionadas if a["nivel"] == "ALTA"]

                    altas_i = _filtrar_parlay(sugs)
                    if len(altas_i) >= 2:
                        prob_p = 1.0
                        for a in altas_i: prob_p *= a["confianza"] / 100
                        sels = " + ".join([a["seleccion"].replace("✅ ","") for a in altas_i])
                        st.markdown(f'<div style="background:linear-gradient(135deg,#1a1500,#2a2000);border:1px solid #f0c040;border-radius:10px;padding:0.8rem 1rem;margin-top:0.75rem"><span style="font-size:0.6rem;color:#f0c040;letter-spacing:2px">💛 PARLAY SUGERIDO</span><div style="font-size:0.85rem;color:#f0c040;margin:0.2rem 0">{sels}</div><div style="font-size:0.65rem;color:#8899bb">Prob. combinada: <b style="color:#f0c040">{prob_p*100:.1f}%</b></div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background:#111827;border:1px solid #1e2d45;border-radius:8px;padding:0.6rem 1rem;font-size:0.8rem;color:#6677aa;margin-top:0.5rem">🎰 Sin señales claras de apuesta para este partido.</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.65rem;color:#4a5568;margin-top:0.4rem">⚠️ Solo informativo. Apuesta responsablemente.</div>', unsafe_allow_html=True)

            elif not resultado_real:
                st.markdown('<div style="text-align:center;padding:3rem;color:#4a5568"><div style="font-size:3rem">⚽</div><div style="margin-top:0.5rem;font-size:0.9rem">Presiona "Simular partido" para ver la predicción</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab_res:
    st.markdown("#### Resultados registrados")
    st.caption("Partidos ya disputados del Mundial 2026.")
    jugados = [p for p in PARTIDOS if p[4] is not None]
    if not jugados:
        st.info("Aún no hay resultados registrados.")
    else:
        for g in sorted(set(p[2] for p in jugados)):
            st.markdown(f"**Grupo {g}**")
            for ea, eb, grupo, sede, res, *_ in jugados:
                if grupo != g: continue
                ga, gb = res
                color = "#0d1f16" if ga != gb else "#0d1827"
                ganador_lbl = (f"→ Ganó **{ea}**" if ga > gb else f"→ Ganó **{eb}**" if gb > ga else "→ **Empate**")
                st.markdown(f'<div style="background:{color};border-radius:8px;padding:0.5rem 1rem;margin-bottom:0.35rem;font-size:0.88rem">{flag_img(ea,24)} {ea} <b style="font-size:1.1rem;color:#4ade80;margin:0 0.4rem">{ga}–{gb}</b>{eb} {flag(eb)}<span style="color:#6677aa;font-size:0.72rem;margin-left:0.8rem">📍 {sede} · {ganador_lbl}</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab_apuestas:
    st.markdown("#### 🎰 Análisis de apuestas — Playdoit & Draftea")
    st.markdown('<div style="background:#1a1500;border:1px solid #5a3a00;border-radius:10px;padding:0.8rem 1rem;margin-bottom:1.5rem;font-size:0.8rem;color:#f0c040">⚠️ <b>Solo informativo.</b> Apuesta solo lo que puedas permitirte perder.</div>', unsafe_allow_html=True)
    if idx_sel is None or not (btn or resultado_real):
        st.markdown('<div style="text-align:center;padding:3rem;color:#4a5568"><div style="font-size:2.5rem">🎰</div><div style="margin-top:0.5rem;font-size:0.9rem">Ve al tab Predictor y simula un partido primero</div></div>', unsafe_allow_html=True)
    else:
        ea2, eb2, grupo2, sede2, res2, arb2 = PARTIDOS[idx_sel]
        r2 = simular(ea2, eb2, sede2, arbitro=arb2, n=500_000)
        r2["goles_totales_esperados"] = r2["goles_a"] + r2["goles_b"]
        sugerencias = analizar_apuestas(ea2, eb2, r2)

        # ── Guardar apuestas en Supabase para historial ──────────────────────
        _guardar_apuestas_supabase(ea2, eb2, grupo2, sugerencias, res2)

        if not sugerencias:
            st.info("El modelo no encontró señales suficientemente claras. Partido demasiado equilibrado.")
        else:
            st.markdown(f"##### {flag(ea2)} {ea2} vs {flag(eb2)} {eb2} — {len(sugerencias)} apuesta(s)")
            for ap in sugerencias:
                conf_pct = min(ap["confianza"], 99)
                nivel_color = "#0d2818" if ap["nivel"] == "ALTA" else "#0d1827"
                nivel_borde = "#2d6b45" if ap["nivel"] == "ALTA" else "#1e3a5f"
                st.markdown(f'<div style="background:{nivel_color};border:1px solid {nivel_borde};border-radius:10px;padding:0.9rem;margin-bottom:0.75rem"><div style="display:flex;justify-content:space-between;margin-bottom:0.3rem"><span style="font-size:0.6rem;color:#6677aa;letter-spacing:2px;text-transform:uppercase">{ap["mercado"]}</span><span style="font-size:0.6rem;color:#4ade80;font-weight:600">{ap["nivel"]}</span></div><div style="font-size:0.95rem;color:#e8eaf0;margin:0.2rem 0;font-weight:600">{ap["seleccion"]}</div><div style="background:#1e2d45;border-radius:3px;height:5px;margin:0.3rem 0"><div style="width:{conf_pct:.0f}%;height:5px;border-radius:3px;background:linear-gradient(90deg,#3b82f6,#4ade80)"></div></div><div style="font-size:0.65rem;color:#4ade80">{conf_pct:.0f}% confianza</div><div style="font-size:0.6rem;color:#6677aa;margin-top:0.2rem">{ap["nota"]}</div><div style="font-size:0.6rem;color:#4a5568;margin-top:0.3rem">📱 {ap["donde"]}</div></div>', unsafe_allow_html=True)
            altas = [a for a in sugerencias if a["nivel"] == "ALTA"]
            if len(altas) >= 2:
                prob_parlay = 1.0
                for a in altas: prob_parlay *= a["confianza"] / 100
                selecciones = " + ".join([a["seleccion"].replace("✅ ", "") for a in altas])
                st.markdown(f'<div style="background:linear-gradient(135deg,#1a1500,#2a2000);border:1px solid #f0c040;border-radius:10px;padding:0.8rem 1rem;margin-top:0.5rem"><div style="font-size:0.65rem;color:#f0c040;letter-spacing:2px;margin-bottom:0.3rem">💛 PARLAY SUGERIDO</div><div style="font-family:\'Bebas Neue\',sans-serif;font-size:1.2rem;color:#f0c040;margin-bottom:0.4rem">{selecciones}</div><div style="font-size:0.75rem;color:#8899bb">Prob. combinada: <b style="color:#f0c040">{prob_parlay*100:.1f}%</b></div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown("#### 📈 Historial de predicciones vs resultados reales")
    partidos_jugados = [p for p in PARTIDOS if p[4] is not None and not str(p[0]).startswith('TBD')]
    total = len(partidos_jugados)
    if total == 0:
        st.info("Aún no hay partidos terminados para calcular accuracy.")
    else:
        aciertos_ganador = 0
        aciertos_over25  = 0
        total_calculados = 0
        historial_rows   = []
        for _p in partidos_jugados:
            _ea_h, _eb_h, _, _sede_h, _res_real, _arb_h = _p[0], _p[1], _p[2], _p[3], _p[4], _p[5]
            try:
                _r_h = simular(_ea_h, _eb_h, _sede_h, arbitro=_arb_h, n=50_000)
                _favorito = _ea_h if _r_h["prob_a"] > _r_h["prob_b"] else _eb_h
                _prob_fav = max(_r_h["prob_a"], _r_h["prob_b"])
                _ga_r, _gb_r = _res_real
                _ganador_real = _ea_h if _ga_r > _gb_r else (_eb_h if _gb_r > _ga_r else "Empate")
                _modelo_correcto = _favorito == _ganador_real
                if _modelo_correcto: aciertos_ganador += 1
                if ((_ga_r + _gb_r) > 2) == (_r_h.get("prob_over25", 50) > 50): aciertos_over25 += 1
                total_calculados += 1
                historial_rows.append({"partido": f"{flag_img(_ea_h,16)} {_ea_h} vs {flag_img(_eb_h,16)} {_eb_h}", "resultado": f"{_ga_r}-{_gb_r}", "favorito_modelo": _favorito, "prob": f"{_prob_fav:.1f}%", "correcto": _modelo_correcto, "ganador_real": _ganador_real})
            except Exception:
                continue
        if total_calculados > 0:
            acc_ganador = aciertos_ganador / total_calculados * 100
            acc_over25  = aciertos_over25  / total_calculados * 100
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                color = "#4ade80" if acc_ganador >= 55 else "#f0c040" if acc_ganador >= 45 else "#ef4444"
                st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color}">{acc_ganador:.1f}%</div><div class="metric-lbl">Accuracy ganador</div><div style="font-size:0.6rem;color:#6677aa;margin-top:0.2rem">{aciertos_ganador}/{total_calculados} partidos</div></div>', unsafe_allow_html=True)
            with col_m2:
                color2 = "#4ade80" if acc_over25 >= 60 else "#f0c040"
                st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color2}">{acc_over25:.1f}%</div><div class="metric-lbl">Accuracy Over/Under 2.5</div><div style="font-size:0.6rem;color:#6677aa;margin-top:0.2rem">{aciertos_over25}/{total_calculados} partidos</div></div>', unsafe_allow_html=True)
            with col_m3:
                st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#60a5fa">{total_calculados}</div><div class="metric-lbl">Partidos analizados</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            for row in historial_rows:
                color_row = "#0d2818" if row["correcto"] else "#1a0d0d"
                borde = "#2d6b45" if row["correcto"] else "#6b2d2d"
                icono = "✅" if row["correcto"] else "❌"
                st.markdown(f'<div style="background:{color_row};border:1px solid {borde};border-radius:8px;padding:0.5rem 0.9rem;margin-bottom:0.3rem;display:flex;align-items:center;gap:0.5rem;font-size:0.82rem"><span>{icono}</span><span style="color:#e8eaf0;flex:2">{row["partido"]}</span><span style="color:#4ade80;font-family:\'Bebas Neue\',sans-serif;font-size:1rem;flex:0.5;text-align:center">{row["resultado"]}</span><span style="color:#6677aa;flex:1.5;text-align:center">Modelo: <b style="color:#f0c040">{row["favorito_modelo"]}</b> ({row["prob"]})</span><span style="color:#8899bb;flex:1;text-align:right">Real: {row["ganador_real"]}</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="model-note">📊 Un buen modelo de fútbol tiene ~55-65% de accuracy. El valor está en identificar apuestas con EV+ positivo.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab_hist_ap:
    st.markdown("#### 🎰 Historial de apuestas sugeridas")
    _apuestas_hist = []
    if SUPABASE_DISPONIBLE and SUPABASE_URL and SUPABASE_KEY:
        try:
            _pts = [p for p in PARTIDOS if p[4] is not None]
            if _pts: actualizar_aciertos(SUPABASE_URL, SUPABASE_KEY, _pts)
            _apuestas_hist = [a for a in cargar_historial_apuestas(SUPABASE_URL, SUPABASE_KEY) if not str(a.get("ea","")).startswith("TBD")]
        except Exception:
            pass
    if not _apuestas_hist:
        st.info("⏳ Aún no hay apuestas registradas. Se guardarán automáticamente al abrir la app el día de los partidos.")
    else:
        _stats = calcular_stats_apuestas(_apuestas_hist)
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        color_ap = "#4ade80" if _stats["accuracy"] >= 60 else "#f0c040" if _stats["accuracy"] >= 45 else "#ef4444"
        with col_a1:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color_ap}">{_stats["accuracy"]:.1f}%</div><div class="metric-lbl">Accuracy apuestas</div><div style="font-size:0.6rem;color:#6677aa">{_stats["aciertos"]}/{_stats["total_evaluadas"]}</div></div>', unsafe_allow_html=True)
        with col_a2:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#4ade80">{_stats["aciertos"]}</div><div class="metric-lbl">✅ Aciertos</div></div>', unsafe_allow_html=True)
        with col_a3:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#ef4444">{_stats["fallos"]}</div><div class="metric-lbl">❌ Fallos</div></div>', unsafe_allow_html=True)
        with col_a4:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#f0c040">{_stats["total_pendientes"]}</div><div class="metric-lbl">⏳ Pendientes</div></div>', unsafe_allow_html=True)
        if _stats.get("evaluadas"):
            st.markdown("##### ✅ Apuestas evaluadas")
            for _ap in _stats["evaluadas"]:
                _color = "#0d2818" if _ap["acierto"] else "#1a0d0d"
                _borde = "#2d6b45" if _ap["acierto"] else "#6b2d2d"
                _icono = "✅" if _ap["acierto"] else "❌"
                _res = _ap.get("resultado_real", "?")
                _fecha = _ap.get("fecha_partido", "?")
                _merc = _ap.get("mercado", "")
                _sel = _ap.get("seleccion", "")
                _conf = _ap.get("confianza", 0)
                _donde = _ap.get("donde", "")
                _guard = _ap.get("guardada_en", "?")
                st.markdown(
                    f'<div style="background:{_color};border:1px solid {_borde};border-radius:8px;padding:0.5rem 0.9rem;margin-bottom:0.3rem">' +
                    f'<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap">' +
                    f'<span>{_icono}</span>' +
                    f'<span style="color:#e8eaf0;font-size:0.8rem;font-weight:600">{flag_img(_ap["ea"],16)} {_ap["ea"]} vs {flag_img(_ap["eb"],16)} {_ap["eb"]}</span>' +
                    f'<span style="color:#4ade80;font-family:Bebas Neue,sans-serif">{_res}</span>' +
                    f'<span style="color:#6677aa;font-size:0.7rem;margin-left:auto">{_fecha}</span>' +
                    f'</div>' +
                    f'<div style="display:flex;gap:1rem;margin-top:0.3rem;flex-wrap:wrap">' +
                    f'<span style="font-size:0.75rem;color:#f0c040">📋 {_merc}</span>' +
                    f'<span style="font-size:0.75rem;color:#e8eaf0">→ {_sel}</span>' +
                    f'<span style="font-size:0.7rem;color:#4ade80">{_conf:.0f}% confianza</span>' +
                    f'</div>' +
                    f'<div style="font-size:0.6rem;color:#4a5568;margin-top:0.2rem">📱 {_donde} · Guardada: {_guard}</div>' +
                    '</div>',
                    unsafe_allow_html=True
                )

        if _stats.get("pendientes"):
            st.markdown("---")
            st.markdown(f"##### ⏳ Apuestas pendientes ({len(_stats['pendientes'])})")
            for _ap in _stats["pendientes"]:
                _fecha = _ap.get("fecha_partido", "?")
                _merc = _ap.get("mercado", "")
                _sel = _ap.get("seleccion", "")
                _conf = _ap.get("confianza", 0)
                st.markdown(
                    f'<div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.5rem 0.9rem;margin-bottom:0.3rem">' +
                    f'<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap">' +
                    f'<span style="color:#e8eaf0;font-size:0.8rem;font-weight:600">{flag_img(_ap["ea"],16)} {_ap["ea"]} vs {flag_img(_ap["eb"],16)} {_ap["eb"]}</span>' +
                    f'<span style="color:#6677aa;font-size:0.7rem;margin-left:auto">{_fecha}</span>' +
                    f'</div>' +
                    f'<div style="display:flex;gap:1rem;margin-top:0.3rem;flex-wrap:wrap">' +
                    f'<span style="font-size:0.75rem;color:#f0c040">📋 {_merc}</span>' +
                    f'<span style="font-size:0.75rem;color:#e8eaf0">→ {_sel}</span>' +
                    f'<span style="font-size:0.7rem;color:#4ade80">{_conf:.0f}% confianza</span>' +
                    f'</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown(
            '<div class="model-note">🎯 Solo apuestas de confianza ALTA (≥75-82%). ' +
            'Se guardan automáticamente antes de cada partido.<br>⚠️ Solo informativo — apuesta responsablemente.</div>',
            unsafe_allow_html=True
        )
# ══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.markdown("#### ¿Cómo funciona el modelo?")
    st.markdown("""
El predictor usa **simulación Monte Carlo con distribución de Poisson** — el mismo enfoque de casas de apuestas y modelos académicos serios.

**En cada simulación el modelo combina:**
- **ELO Rating** unificado (escala 1200-1900) — diferencia de 400 puntos ≈ 90% de victorias
- **Forma en el Mundial** — goles reales de J1+J2 ajustan ligeramente las lambdas
- **Altitud de la sede** — penaliza hasta 8% en sedes >1,700m (Azteca: 2,240m)
- **Ventaja local** — +10% para México, Canadá y EEUU en sus estadios
- **Bajas confirmadas** — lesiones y suspensiones reducen el lambda ofensivo
- **Clima** — calor + humedad penaliza a equipos de climas fríos
- **H2H** — historial con peso exponencial + resultado de este Mundial (4x)
- **Árbitro** — promedio histórico de tarjetas por partido
- **Córners** — calibrados con datos reales del torneo

**Optimizaciones versión web (Streamlit Cloud):**
- Arrays `int32` en vez de `int64` → mitad de RAM
- `@st.cache_resource` para Dixon-Coles → modelo entrena una sola vez
- `@st.cache_data(ttl=3600)` → resultados cacheados 1 hora
- Liberación explícita de arrays antes del cache

**Rendimiento:** 10M simulaciones en ~3-5 segundos en Streamlit Cloud.
""")
    st.markdown("---")
    st.markdown("#### ELO Ratings — 48 selecciones")
    sorted_elo = sorted([(k, v) for k, v in ELO.items() if k not in ("Algeria", "Arabia Saudi")], key=lambda x: x[1], reverse=True)
    cols = st.columns(3)
    for i, (equipo, elo) in enumerate(sorted_elo):
        with cols[i % 3]:
            st.markdown(f"{flag(equipo)} **{equipo}** — `{elo}`")
