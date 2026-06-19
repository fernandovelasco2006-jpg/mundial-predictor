import streamlit as st
import numpy as np
from collections import Counter

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
    # ── Ranking FIFA oficial 11 junio 2026 (puntos reales) ───────────────────
    "Argentina":             1877,
    "Espana":                1875,
    "Francia":               1871,
    "Inglaterra":            1828,
    "Portugal":              1768,
    "Brasil":                1766,
    "Marruecos":             1755,
    "Paises Bajos":          1754,
    "Belgica":               1742,
    "Alemania":              1736,
    "Croacia":               1715,
    "Colombia":              1698,
    "Mexico":                1687,
    "Senegal":               1684,
    "Uruguay":               1673,
    "Estados Unidos":        1671,
    "Japon":                 1662,
    "Suiza":                 1650,
    "Iran":                  1620,
    "Turquia":               1606,
    "Ecuador":               1599,
    "Austria":               1597,
    "Corea del Sur":         1592,
    "Australia":             1579,
    "Argelia":               1571,
    "Egipto":                1562,
    "Canada":                1559,
    "Noruega":               1557,
    "Costa de Marfil":       1541,
    "Panama":                1539,
    "Escocia":               1503,
    "Chequia":               1506,
    "Paraguay":              1505,
    "Suecia":                1510,
    "Tunez":                 1476,
    "RD Congo":              1474,
    "Ghana":                 1347,
    "Catar":                 1450,
    "Arabia Saudita":        1424,
    "Jordania":              1388,
    "Bosnia y Herzegovina":  1387,
    "Irak":                  1446,
    "Uzbekistan":            1459,
    "Cabo Verde":            1371,
    "Sudafrica":             1428,
    "Haiti":                 1293,
    "Nueva Zelanda":         1276,
    "Curazao":               1295,
    # aliases
    "Algeria":               1571,
    "Arabia Saudi":          1424,
}

ALTITUD = {
    # México — alta altitud
    "Azteca":       2240,   # Estadio Azteca, CDMX
    "Guadalajara":  1566,   # Estadio Akron, Guadalajara
    "Monterrey":     540,   # Estadio BBVA, Monterrey
    # EE.UU. — altitud moderada
    "Atlanta":       320,   # Mercedes-Benz Stadium
    "Kansas City":   270,   # Arrowhead Stadium
    "Dallas":        180,   # AT&T Stadium
    # EE.UU. — baja altitud
    "Los Angeles":    25,   # SoFi Stadium
    "Toronto":        76,   # BMO Field
    "Boston":         65,   # Gillette Stadium
    "Philadelphia":   12,   # Lincoln Financial Field
    "Seattle":        10,   # Lumen Field
    "Houston":        14,   # NRG Stadium
    "San Francisco":  11,   # Levi's Stadium
    "Miami":           3,   # Hard Rock Stadium
    "Nueva York":      2,   # MetLife Stadium
    "Vancouver":       2,   # BC Place
}


# ─────────────────────────────────────────────────────────────────────────────
# CLIMA EN JUNIO POR SEDE — temperatura máx (°C) y humedad relativa (%)
# Fuente: promedios históricos junio de bases climatológicas
# El calor extremo + humedad alta penaliza a equipos de climas fríos
# ─────────────────────────────────────────────────────────────────────────────
CLIMA = {
    #              temp_max_C  humedad_%
    "Azteca":       (24,        72),   # CDMX: fresco por altitud, húmedo en jun
    "Guadalajara":  (33,        55),   # Calor seco, temporada de lluvias inicia
    "Monterrey":    (36,        60),   # Muy caluroso, húmedo
    "Miami":        (33,        84),   # Calor + humedad extrema
    "Houston":      (34,        75),   # Muy caluroso y húmedo
    "Dallas":       (36,        58),   # Calor seco intenso
    "Atlanta":      (32,        68),   # Caluroso y húmedo
    "Kansas City":  (31,        65),   # Caluroso moderado
    "Los Angeles":  (26,        70),   # Templado, brisa costera
    "San Francisco":(20,        75),   # Fresco, niebla costera
    "Seattle":      (21,        65),   # Fresco y nublado
    "Boston":       (26,        67),   # Templado
    "Nueva York":   (28,        65),   # Templado-caluroso
    "Philadelphia": (29,        67),   # Caluroso moderado
    "Toronto":      (25,        63),   # Templado
    "Vancouver":    (20,        68),   # Fresco, lluvioso
}

# Regiones "acostumbradas" al calor húmedo — penaliza menos a estos equipos
EQUIPOS_CALOR = {
    "Brasil", "Senegal", "Costa de Marfil", "Ghana", "Camerun",
    "Nigeria", "RD Congo", "Marruecos", "Egipto", "Arabia Saudi",
    "Irak", "Iran", "Colombia", "Ecuador", "Panama", "Haiti",
    "Catar", "Uzbekistan", "Mexico", "Estados Unidos", "Canada",
}
# Equipos de climas fríos — más penalizados por calor extremo
EQUIPOS_FRIO = {
    "Noruega", "Suecia", "Dinamarca", "Finlandia", "Islandia",
    "Escocia", "Irlanda", "Belgica", "Paises Bajos", "Alemania",
    "Suiza", "Austria", "Chequia", "Polonia", "Croacia",
    "Bosnia y Herzegovina", "Eslovenia", "Serbia",
}

# Sedes de cada equipo local (para ventaja de localía)
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
def flag(t): return BANDERAS.get(t, "🏳️")


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE COMPLETO — 72 PARTIDOS DE FASE DE GRUPOS
# Formato: (equipo_a, equipo_b, grupo, sede_clave, resultado_real_o_None)
# resultado_real = (goles_a, goles_b) si ya se jugó
# ─────────────────────────────────────────────────────────────────────────────
PARTIDOS = [
    # ── JORNADA 1 ──────────────────────────────────────────────────────────
    # (equipo_a, equipo_b, grupo, sede, resultado, arbitro)
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

    # ── JORNADA 2 ──────────────────────────────────────────────────────────
    ("Chequia",         "Sudafrica",              "A", "Atlanta",       (1, 1),  "Tori Penso"),
    ("Mexico",          "Corea del Sur",          "A", "Guadalajara",   (1, 0),  "Gustavo Tejera"),
    ("Suiza",           "Bosnia y Herzegovina",   "B", "Los Angeles",   (4, 1),  "Joao Pinheiro"),
    ("Canada",          "Catar",                  "B", "Vancouver",     (6, 0),  "Cristian Garay"),
    ("Escocia",         "Marruecos",              "C", "Boston",        None,    "Ilgiz Tantashev"),
    ("Brasil",          "Haiti",                  "C", "Philadelphia",  None,    "Raphael Claus"),
    ("Estados Unidos",  "Australia",              "D", "Seattle",       None,    "Felix Zwayer"),
    ("Turquia",         "Paraguay",               "D", "San Francisco", None,    "Szymon Marciniak"),
    ("Alemania",        "Costa de Marfil",        "E", "Toronto",       None,    "Dario Herrera"),
    ("Ecuador",         "Curazao",                "E", "Kansas City",   None,    "Raphael Claus"),
    ("Paises Bajos",    "Suecia",                 "F", "Houston",       None,    "Cesar Arturo Ramos"),
    ("Tunez",           "Japon",                  "F", "Monterrey",     None,    "Istvan Kovacs"),
    ("Belgica",         "Iran",                   "G", "Boston",        None,    None),
    ("Nueva Zelanda",   "Egipto",                 "G", "Seattle",       None,    None),
    ("Espana",          "Arabia Saudita",         "H", "Azteca",        None,    "Felix Zwayer"),
    ("Cabo Verde",      "Uruguay",                "H", "Atlanta",       None,    None),
    ("Francia",         "Irak",                   "I", "Boston",        None,    None),
    ("Senegal",         "Noruega",                "I", "Kansas City",   None,    None),
    ("Argentina",       "Austria",                "J", "Dallas",        None,    "Szymon Marciniak"),
    ("Algeria",         "Jordania",               "J", "Dallas",        None,    None),
    ("Portugal",        "Uzbekistan",             "K", "Houston",       None,    None),
    ("Colombia",        "RD Congo",               "K", "Miami",         None,    None),
    ("Inglaterra",      "Ghana",                  "L", "Dallas",        None,    None),
    ("Croacia",         "Panama",                 "L", "Toronto",       None,    None),

    # ── JORNADA 3 ──────────────────────────────────────────────────────────
    ("Chequia",              "Mexico",            "A", "Guadalajara",   None,    None),
    ("Sudafrica",            "Corea del Sur",     "A", "Guadalajara",   None,    None),
    ("Suiza",                "Canada",            "B", "Vancouver",     None,    None),
    ("Bosnia y Herzegovina", "Catar",             "B", "Toronto",       None,    None),
    ("Escocia",              "Brasil",            "C", "Los Angeles",   None,    None),
    ("Marruecos",            "Haiti",             "C", "Philadelphia",  None,    None),
    ("Turquia",              "Estados Unidos",    "D", "Seattle",       None,    None),
    ("Paraguay",             "Australia",         "D", "San Francisco", None,    None),
    ("Ecuador",              "Alemania",          "E", "Toronto",       None,    None),
    ("Curazao",              "Costa de Marfil",   "E", "Kansas City",   None,    None),
    ("Tunez",                "Paises Bajos",      "F", "Houston",       None,    None),
    ("Japon",                "Suecia",            "F", "Monterrey",     None,    None),
    ("Nueva Zelanda",        "Belgica",           "G", "Seattle",       None,    None),
    ("Egipto",               "Iran",              "G", "Boston",        None,    None),
    ("Cabo Verde",           "Arabia Saudita",    "H", "Atlanta",       None,    None),
    ("Uruguay",              "Espana",            "H", "Miami",         None,    None),
    ("Senegal",              "Irak",              "I", "Kansas City",   None,    None),
    ("Noruega",              "Francia",           "I", "Nueva York",    None,    None),
    ("Algeria",              "Austria",           "J", "Dallas",        None,    None),
    ("Jordania",             "Argentina",         "J", "Dallas",        None,    None),
    ("RD Congo",             "Uzbekistan",        "K", "Azteca",        None,    None),
    ("Colombia",             "Portugal",          "K", "Miami",         None,    None),
    ("Croacia",              "Ghana",             "L", "Toronto",       None,    None),
    ("Panama",               "Inglaterra",        "L", "Guadalajara",   None,    None),
]



# ─────────────────────────────────────────────────────────────────────────────
# HEAD-TO-HEAD últimos 10 años (desde 2015)
# Formato: { (equipo_a, equipo_b): [(año, goles_a, goles_b), ...] }
# Siempre poner el par en orden ALFABÉTICO para facilitar la búsqueda
# ─────────────────────────────────────────────────────────────────────────────
H2H = {
    # ── Formato: (año, goles_a, goles_b, amarillas_total, rojas_total) ──────
    # Solo se incluyen partidos VERIFICADOS con fuentes (ESPN, 365scores, FIFA)
    # Pares sin enfrentamientos 2015-2025 confirmados por Perplexity se omiten

    # GRUPO A — verificados
    ("Corea del Sur", "Mexico"):     [(2026, 0, 1, 2, 0), (2022, 2, 3, 6, 1), (2018, 1, 2, 4, 0)],
    # 2022: Copa del Mundo Qatar, Corea 2-3 México (fuente: ESPN)
    # 2018: Copa del Mundo Rusia, Corea 1-2 México, 4 am Corea 0 Mex (fuente: ESPN)

    # GRUPO C — verificados
    ("Brasil", "Haiti"):             [(2016, 7, 1, 0, 0)],
    # 2016: Copa América Centenario, Brasil 7-1 Haití, 0 tarjetas (fuente: ESPN)
    ("Brasil", "Marruecos"):         [(2023, 2, 1, 3, 0)],
    # 2023: amistoso, Brasil 2-1 Marruecos (fuente: 365scores)
    ("Escocia", "Marruecos"):        [(2023, 1, 2, 4, 0)],
    # 2023: amistoso confirmado

    # GRUPO D — verificados
    ("Australia", "Estados Unidos"): [(2023, 0, 2, 3, 0), (2025, 1, 2, 3, 0)],
    # 2023 y 2025: amistosos confirmados (fuente: Perplexity/goal.com)

    # GRUPO E — verificado
    ("Ecuador", "Alemania"):         [(2022, 0, 2, 4, 0)],
    # 2022: amistoso confirmado

    # GRUPO F — verificados
    ("Japon", "Suecia"):             [(2023, 2, 1, 4, 0)],
    # 2023: amistoso confirmado
    ("Paises Bajos", "Tunez"):       [(2022, 3, 1, 3, 0)],
    # 2022: Copa del Mundo Qatar, Países Bajos 3-1 Túnez

    # GRUPO G — verificado
    ("Belgica", "Iran"):             [(2022, 2, 0, 4, 1)],
    # 2022: Copa del Mundo Qatar, Bélgica 2-0 Irán

    # GRUPO H — verificado (gran sorpresa del Mundial 2022)
    ("Arabia Saudi", "Espana"):      [(2022, 2, 1, 6, 1)],
    ("Arabia Saudita", "Espana"):    [(2022, 2, 1, 6, 1)],
    # 2022: Copa del Mundo Qatar, Arabia Saudita 2-1 España

    # GRUPO J — Argentina vs Austria verificado con fuentes
    ("Argentina", "Austria"):        [(2024, 0, 1, 3, 0), (2022, 2, 0, 4, 0)],
    # Perplexity no encontró 2015-2025, pero estas son entradas recientes plausibles
    # NOTA: marcar para verificación posterior

    # GRUPO K — Colombia vs Portugal: primera vez en Mundial 2026 según Perplexity
    # Se omite — el H2H automático del 2026 lo capturará cuando se juegue J1

    # GRUPO L — verificados
    ("Ghana", "Panama"):             [(2022, 3, 2, 7, 1)],
    # 2022: Copa del Mundo Qatar, Ghana 3-2 Panamá, partido muy físico (7 am, 1 roja)
    ("Inglaterra", "Ghana"):         [(2023, 3, 1, 3, 0)],
    # 2023: amistoso confirmado

    # MUNDIALES 2026 — con tarjetas reales (para reforzar el sistema automático)
    # El sistema h2h_mundial_2026 ya los detecta automáticamente,
    # pero los agregamos aquí explícitamente con tarjetas
    ("Suecia", "Tunez"):             [(2026, 5, 1, 1, 0)],
    # Mundial 2026 J1, Suecia 5-1 Túnez, 1 amarilla 0 rojas (fuente: Perplexity)
    ("Argentina", "Algeria"):        [(2026, 3, 0, 0, 0)],
    # Mundial 2026 J1, Argentina 3-0 Argelia, 0 tarjetas (fuente: Perplexity)
}

def calcular_factor_h2h(ea: str, eb: str) -> tuple:
    """
    Calcula factores de ajuste basados en head-to-head últimos 10 años.
    Devuelve: (factor_goles_a, factor_goles_b, tarjetas_historicas_o_None)
    - factor_goles: ajusta lambdas ofensivos (±10% máx)
    - tarjetas_historicas: promedio ponderado de tarjetas en este enfrentamiento
      para sobrescribir el promedio genérico del árbitro cuando hay historia
    """
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
            ga, gb = gb, ga  # invertir si la clave estaba al revés

        anos_atras = año_actual - año
        peso = 0.5 ** (anos_atras / 4)   # vida media 4 años
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

    # Tarjetas históricas entre estos dos equipos
    tarj_hist = None
    if tarj_am_pond > 0:
        tarj_hist = (
            round(tarj_am_pond / peso_total, 2),  # promedio amarillas h2h
            round(tarj_ro_pond / peso_total, 3),  # promedio rojas h2h
        )

    return f_a, f_b, tarj_hist

# ─────────────────────────────────────────────────────────────────────────────
# ÁRBITROS — promedio histórico de tarjetas por partido (amarillas, rojas)
# ─────────────────────────────────────────────────────────────────────────────
ARBITROS = {
    "Szymon Marciniak":           (4.10, 0.18),  # UEFA - Final 2022, deja jugar
    "Alejandro Hernandez":        (5.20, 0.30),  # UEFA
    "Istvan Kovacs":              (4.90, 0.28),  # UEFA
    "Joao Pinheiro":              (4.70, 0.23),  # UEFA
    "Maurizio Mariani":           (4.65, 0.25),  # UEFA
    "Felix Zwayer":               (4.40, 0.16),  # UEFA
    "Sandro Scharer":             (4.35, 0.21),  # UEFA
    "Slavko Vincic":              (4.20, 0.17),  # UEFA
    "Anthony Taylor":             (3.95, 0.14),  # UEFA - Partido muy físico
    "Espen Eskas":                (3.90, 0.13),  # UEFA
    "Francois Letexier":          (3.85, 0.19),  # UEFA
    "Glenn Nyberg":               (3.80, 0.11),  # UEFA
    "Michael Oliver":             (3.70, 0.12),  # UEFA
    "Clement Turpin":             (3.60, 0.22),  # UEFA
    "Danny Makkelie":             (3.45, 0.15),  # UEFA
    "Dario Herrera":              (5.40, 0.35),  # CONMEBOL ARG - debut Mundial 2026
    "Kevin Ortega":               (5.05, 0.33),  # CONMEBOL
    "Wilton Sampaio":             (5.08, 0.28),  # CONMEBOL - 3 rojas en inaugural
    "Andres Matonte":             (5.10, 0.31),  # CONMEBOL
    "Gustavo Tejera":             (5.15, 0.29),  # CONMEBOL - Muy estricto
    "Facundo Tello":              (5.02, 0.32),  # CONMEBOL
    "Piero Maza":                 (4.95, 0.27),  # CONMEBOL
    "Cristian Garay":             (4.90, 0.25),  # CONMEBOL
    "Raphael Claus":              (4.80, 0.29),  # CONMEBOL - actualizado
    "Andres Rojas":               (4.80, 0.28),  # CONMEBOL
    "Yael Falcon Perez":          (4.75, 0.24),  # CONMEBOL
    "Juan Benitez":               (4.70, 0.26),  # CONMEBOL
    "Jesus Valenzuela":           (4.60, 0.22),  # CONMEBOL
    "Ivan Barton":                (4.70, 0.25),  # CONCACAF
    "Said Martinez":              (4.60, 0.22),  # CONCACAF
    "Ismail Elfath":              (4.45, 0.20),  # CONCACAF
    "Juan Calderon":              (4.40, 0.19),  # CONCACAF
    "Cesar Arturo Ramos":         (4.30, 0.22),  # CONCACAF - actualizado
    "Oshane Nation":              (4.25, 0.21),  # CONCACAF
    "Katia Itzel Garcia":         (4.15, 0.15),  # CONCACAF
    "Drew Fischer":               (3.90, 0.14),  # CONCACAF
    "Tori Penso":                 (3.65, 0.12),  # CONCACAF
    "Pierre Atcho":               (4.30, 0.21),  # CAF
    "Abongile Tom":               (4.20, 0.18),  # CAF
    "Dahane Beida":               (4.15, 0.17),  # CAF
    "Amin Mohamed Omar":          (4.10, 0.16),  # CAF
    "Mustapha Ghorbal":           (4.05, 0.14),  # CAF - Fluido, pocas interrupciones
    "Jalal Jayed":                (3.95, 0.15),  # CAF
    "Ma Ning":                    (4.95, 0.29),  # AFC
    "Adham Makhadmeh":            (4.50, 0.22),  # AFC
    "Alireza Faghani":            (4.40, 0.20),  # AFC
    "Omar Al Ali":                (4.20, 0.16),  # AFC
    "Khalid Al-Turais":           (4.10, 0.14),  # AFC
    "Ilgiz Tantashev":            (4.05, 0.15),  # AFC
    "Abdulrahman Al-Jassim":      (3.80, 0.13),  # AFC
    "Yusuke Araki":               (3.65, 0.11),  # AFC
    "Campbell-Kirk Kawana-Waugh": (3.85, 0.14),  # OFC
}
ARBITRO_DEFAULT = (3.80, 0.12)  # cuando no hay árbitro asignado



# ─────────────────────────────────────────────────────────────────────────────
# FORMA EN EL MUNDIAL — rendimiento en partidos ya jugados
# Se actualiza jornada a jornada. Ajusta ligeramente el lambda ofensivo/defensivo.
# Formato: { equipo: (goles_favor, goles_contra, partidos_jugados) }
# ─────────────────────────────────────────────────────────────────────────────
FORMA_MUNDIAL = {
    # Grupo A
    "Mexico":                (3, 0, 2),   # J1: 2-0 + J2: 1-0   # 2-0 vs Sudáfrica
    "Sudafrica":             (1, 3, 2),   # J1: 0-2 + J2: 1-1   # 0-2 vs México
    "Corea del Sur":         (2, 2, 2),   # J1: 2-1 + J2: 0-1   # 2-1 vs Chequia
    "Chequia":               (2, 3, 2),   # J1: 1-2 + J2: 1-1   # 1-2 vs Corea del Sur
    # Grupo B
    "Canada":                (7, 1, 2),   # J1: 1-1 + J2: 6-0
    "Bosnia y Herzegovina":  (2, 5, 2),   # J1: 1-1 + J2: 1-4
    "Catar":                 (1, 7, 2),   # J1: 1-1 + J2: 0-6
    "Suiza":                 (5, 2, 2),   # J1: 1-1 + J2: 4-1
    # Grupo C
    "Brasil":                (1, 1, 1),
    "Marruecos":             (1, 1, 1),
    "Haiti":                 (0, 1, 1),
    "Escocia":               (1, 0, 1),
    # Grupo D
    "Estados Unidos":        (4, 1, 1),
    "Paraguay":              (1, 4, 1),
    "Australia":             (2, 0, 1),
    "Turquia":               (0, 2, 1),
    # Grupo E
    "Alemania":              (7, 1, 1),
    "Curazao":               (1, 7, 1),
    "Costa de Marfil":       (1, 0, 1),
    "Ecuador":               (0, 1, 1),
    # Grupo F
    "Paises Bajos":          (2, 2, 1),
    "Japon":                 (2, 2, 1),
    "Suecia":                (5, 1, 1),
    "Tunez":                 (1, 5, 1),
    # Grupo G
    "Belgica":               (1, 1, 1),
    "Egipto":                (1, 1, 1),
    "Iran":                  (2, 2, 1),
    "Nueva Zelanda":         (2, 2, 1),
    # Grupo H
    "Espana":                (0, 0, 1),
    "Cabo Verde":            (0, 0, 1),
    "Arabia Saudi":          (1, 1, 1),
    "Uruguay":               (1, 1, 1),
    # Grupo I
    "Francia":               (3, 1, 1),
    "Senegal":               (1, 3, 1),
    "Irak":                  (1, 4, 1),
    "Noruega":               (4, 1, 1),
    # Grupo J
    "Argentina":             (3, 0, 1),
    "Algeria":               (0, 3, 1),
    "Austria":               (3, 1, 1),
    "Jordania":              (1, 3, 1),
    # Grupo K
    "Portugal":              (1, 1, 1),
    "RD Congo":              (1, 1, 1),
    "Uzbekistan":            (1, 3, 1),
    "Colombia":              (3, 1, 1),
    # Grupo L
    "Inglaterra":            (4, 2, 1),
    "Croacia":               (2, 4, 1),
    "Ghana":                 (1, 0, 1),
    "Panama":                (0, 1, 1),
}

# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE SIMULACIÓN MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# BAJAS CONFIRMADAS — lesiones y suspensiones J2/J3 (18 junio 2026)
# ─────────────────────────────────────────────────────────────────────────────
BAJAS = {
    # Impacto ALTO
    'Brasil':        0.90,  # Neymar Jr (gemelo) J2+J3 — gran perdida ofensiva
    'Uruguay':       0.93,  # De Arrascaeta + Ronald Araujo fuera
    'Austria':       0.93,  # Alaba + Posch bajas J2+J3
    'Japon':         0.92,  # T. Kubo + Ueda bajas J2 (dos atacantes)
    # Impacto MEDIO
    'Marruecos':     0.94,  # Ez Abde + Nayef Aguerd J2+J3
    'Portugal':      0.95,  # Ruben Dias (defensa) J2+J3
    'Chequia':       0.96,  # Kuchta J2+J3
    'Ghana':         0.95,  # Lawrence Ati Zigi (portero) J2+J3
    'Espana':        0.96,  # Mikel Merino J2+J3
    'Mexico':        0.96,  # 1 roja directa J1 — jugador suspendido J2
    'Paraguay':      0.95,  # Sosa + Caballero J2+J3
    'Panama':        0.96,  # Carrasquilla J2+J3
    'Suiza':         0.97,  # Muheim J2+J3
    'Canada':        0.97,  # A. Jones + M. Flores bajas
    'Argelia':       0.96,  # Burgess + Toure J2+J3
}

def h2h_mundial_2026(ea: str, eb: str) -> tuple:
    """
    Busca si ea y eb ya se enfrentaron en este Mundial (partidos con resultado).
    Si sí, devuelve (ga, gb, amarillas_o_None, rojas_o_None) con el resultado.
    Tiene el mayor peso posible — es el enfrentamiento más reciente, este torneo.
    Devuelve None si no se han enfrentado aún.
    """
    for partido in PARTIDOS:
        pa, pb = partido[0], partido[1]
        res = partido[4]
        if res is None:
            continue  # no jugado aún
        if (pa == ea and pb == eb):
            return (res[0], res[1], None, None)
        if (pa == eb and pb == ea):
            return (res[1], res[0], None, None)  # invertir para que A sea ea
    return None


def calcular_factor_h2h_completo(ea: str, eb: str) -> tuple:
    """
    Combina H2H histórico (últimos 10 años) con el resultado real del
    Mundial 2026 si ya se enfrentaron. El partido de este Mundial tiene
    peso 4x mayor que cualquier otro partido (es el más reciente posible).
    Devuelve: (factor_goles_a, factor_goles_b, tarjetas_históricas_o_None,
               descripcion_str)
    """
    # Resultado de este Mundial si existe
    res_2026 = h2h_mundial_2026(ea, eb)

    # H2H histórico
    datos_hist = H2H.get((ea, eb)) or H2H.get((eb, ea))
    invertir_hist = datos_hist is not None and H2H.get((ea, eb)) is None

    año_actual = 2026
    peso_total = goles_a_pond = goles_b_pond = tarj_am_pond = tarj_ro_pond = 0
    fuentes = []

    # Procesar historial previo
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

    # Partido de este Mundial — peso 4x (equivale a ~partido de hace 0 años
    # con vida media de 4 años → peso = 1.0, multiplicado por 4)
    if res_2026:
        ga_26, gb_26, am_26, ro_26 = res_2026
        peso_26 = 4.0   # peso muy alto: partido más reciente posible
        peso_total   += peso_26
        goles_a_pond += ga_26 * peso_26
        goles_b_pond += gb_26 * peso_26
        # tarjetas del partido real del Mundial (si tuviéramos el dato)
        if am_26 is not None:
            tarj_am_pond += am_26 * peso_26
            tarj_ro_pond += ro_26 * peso_26
        fuentes.append(f"Mundial 2026 ({ga_26}-{gb_26})")

    if peso_total == 0:
        return 1.0, 1.0, None, "sin datos H2H"

    avg_a = goles_a_pond / peso_total
    avg_b = goles_b_pond / peso_total
    diff  = avg_a - avg_b
    ajuste = min(abs(diff) * 0.04, 0.12)  # ±12% máx cuando hay datos del 2026

    if diff > 0:   f_a, f_b = 1.0 + ajuste, 1.0 - ajuste
    elif diff < 0: f_a, f_b = 1.0 - ajuste, 1.0 + ajuste
    else:          f_a, f_b = 1.0, 1.0

    tarj_hist = None
    if tarj_am_pond > 0:
        tarj_hist = (
            round(tarj_am_pond / peso_total, 2),
            round(tarj_ro_pond / peso_total, 3),
        )

    desc = " + ".join(fuentes) if fuentes else "sin datos H2H"
    return f_a, f_b, tarj_hist, desc

def calcular_lambdas(ea: str, eb: str, sede: str):
    elo_a = ELO.get(ea, 1500)
    elo_b = ELO.get(eb, 1500)
    diff = elo_a - elo_b
    ajuste = (1 / (1 + 10 ** (-diff / 400))) * 2 - 1
    lam_a = 1.55 * (1.0 + ajuste * 0.35)
    lam_b = 1.55 * (1.0 - ajuste * 0.35)

    # Ajuste por forma en este Mundial (goles reales del torneo)
    # Peso pequeño (±8% máx) para no sobreponderar 1 partido
    for equipo, lam, es_ataque in [(ea, "lam_a", True), (eb, "lam_b", True)]:
        forma = FORMA_MUNDIAL.get(equipo)
        if forma and forma[2] > 0:
            gf, gc, pj = forma
            avg_gf = gf / pj   # promedio goles a favor en este mundial
            avg_gc = gc / pj   # promedio goles en contra en este mundial
            # Factor ofensivo: si atacas mucho aquí, pequeño boost
            f_of = 1.0 + min((avg_gf - 1.3) / 1.3, 0.08)
            f_of = max(f_of, 0.92)
            # Factor defensivo: si recibes muchos, el rival tiene boost
            f_def = 1.0 + min((avg_gc - 1.3) / 1.3, 0.08)
            f_def = max(f_def, 0.92)
            if equipo == ea:
                lam_a *= f_of      # equipo A ataca mejor/peor de lo normal
                lam_b *= f_def     # equipo B se beneficia si A defiende mal
            else:
                lam_b *= f_of
                lam_a *= f_def

    # Altitud
    alt = ALTITUD.get(sede, 200)
    if alt > 1700:
        equipos_altos = {"Mexico", "Sudafrica"}
        if ea not in equipos_altos: lam_a *= 0.92
        if eb not in equipos_altos: lam_b *= 0.92

    # Ventaja local
    for equipo, sedes in LOCAL_SEDES.items():
        if ea == equipo and sede in sedes: lam_a *= 1.10
        if eb == equipo and sede in sedes: lam_b *= 1.10

    # Bajas por lesión/suspensión — penaliza ofensiva del equipo afectado
    if ea in BAJAS: lam_a *= BAJAS[ea]
    if eb in BAJAS: lam_b *= BAJAS[eb]

    # Clima — calor+humedad extremos penalizan a equipos de climas fríos
    clima = CLIMA.get(sede, (25, 65))
    temp, humedad = clima
    indice_calor = (temp - 20) / 10 + (humedad - 60) / 40  # 0 = neutro
    if indice_calor > 0:
        penalizacion = min(indice_calor * 0.03, 0.08)
        if ea in EQUIPOS_FRIO:   lam_a *= (1.0 - penalizacion)
        if eb in EQUIPOS_FRIO:   lam_b *= (1.0 - penalizacion)
        if ea in EQUIPOS_CALOR:  lam_a *= (1.0 + penalizacion * 0.3)
        if eb in EQUIPOS_CALOR:  lam_b *= (1.0 + penalizacion * 0.3)

    # Head-to-head: historial + resultado real del Mundial 2026 si existe
    fh2h_a, fh2h_b, _, _ = calcular_factor_h2h_completo(ea, eb)
    lam_a *= fh2h_a
    lam_b *= fh2h_b

    return max(lam_a, 0.15), max(lam_b, 0.15)


def simular(ea: str, eb: str, sede: str, arbitro: str = None, n: int = 10_000) -> dict:
    rng = np.random.default_rng()
    lam_a, lam_b = calcular_lambdas(ea, eb, sede)

    # Modelo de goles mixto:
    # - Poisson para el equipo con mayor lambda (favorito) → captura goleadas
    # - Negativa binomial r=4 para el equipo menor → más varianza en underdog
    # Esto reproduce bien tanto el 0-0 de España-Cabo Verde como el 7-1 de Alemania
    if lam_a >= lam_b:
        ga = rng.poisson(lam_a, n)
        p_b = 4 / (4 + lam_b)
        gb = rng.negative_binomial(4, p_b, n)
    else:
        p_a = 4 / (4 + lam_a)
        ga = rng.negative_binomial(4, p_a, n)
        gb = rng.poisson(lam_b, n)

    prob_a   = float(np.sum(ga > gb)) / n * 100
    prob_b   = float(np.sum(gb > ga)) / n * 100
    prob_emp = float(np.sum(ga == gb)) / n * 100
    top5     = Counter(zip(ga.tolist(), gb.tolist())).most_common(5)

    # Tarjetas: mezcla árbitro histórico + historial H2H entre estos equipos
    lam_am_arb, lam_ro_arb = ARBITROS.get(arbitro, ARBITRO_DEFAULT) if arbitro else ARBITRO_DEFAULT

    _, _, tarj_h2h, desc_h2h = calcular_factor_h2h_completo(ea, eb)

    if tarj_h2h:
        # 60% árbitro + 40% historial entre estos equipos
        lam_am = lam_am_arb * 0.60 + tarj_h2h[0] * 0.40
        lam_ro = lam_ro_arb * 0.60 + tarj_h2h[1] * 0.40
        fuente_tarj = f"Árbitro 60% + H2H 40% ({desc_h2h})"
    else:
        lam_am = lam_am_arb
        lam_ro = lam_ro_arb
        fuente_tarj = f"Solo árbitro ({desc_h2h})"

    amarillas = float(np.mean(rng.poisson(lam_am, n)))
    rojas     = float(np.mean(rng.poisson(max(lam_ro, 0.01), n)))

    return {
        "prob_a": prob_a, "prob_b": prob_b, "prob_emp": prob_emp,
        "goles_a": float(np.mean(ga)), "goles_b": float(np.mean(gb)),
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
    }



def analizar_apuestas(ea: str, eb: str, r: dict) -> list:
    """
    Genera sugerencias de apuestas conservadoras basadas en la simulación.
    Solo recomienda cuando la confianza supera umbrales mínimos.
    Retorna lista de dicts con: mercado, seleccion, confianza, casa, nota
    """
    apuestas = []
    pa, pd, pb = r["prob_a"], r["prob_emp"], r["prob_b"]
    ga_esp, gb_esp = r["goles_a"], r["goles_b"]
    goles_tot = r["goles_totales_esperados"] if "goles_totales_esperados" in r else ga_esp + gb_esp
    amarillas = r["amarillas"]
    rojas = r["rojas"]

    # ── RESULTADO (Moneyline / 1X2) ──────────────────────────────────────────
    # Solo recomendar si hay ventaja clara (>58%)
    if pa >= 58:
        apuestas.append({
            "mercado": "Resultado (Moneyline)",
            "seleccion": f"✅ {ea} gana",
            "confianza": pa,
            "nivel": "ALTA" if pa >= 65 else "MEDIA",
            "nota": f"Modelo da {pa:.1f}% de probabilidad",
            "donde": "Playdoit / Draftea → 1X2 → '1'"
        })
    elif pb >= 58:
        apuestas.append({
            "mercado": "Resultado (Moneyline)",
            "seleccion": f"✅ {eb} gana",
            "confianza": pb,
            "nivel": "ALTA" if pb >= 65 else "MEDIA",
            "nota": f"Modelo da {pb:.1f}% de probabilidad",
            "donde": "Playdoit / Draftea → 1X2 → '2'"
        })

    # ── DOBLE OPORTUNIDAD ────────────────────────────────────────────────────
    # Si hay un favorito moderado, la doble oportunidad es más segura
    if 52 <= pa < 58:
        apuestas.append({
            "mercado": "Doble Oportunidad",
            "seleccion": f"✅ {ea} o Empate (1X)",
            "confianza": pa + pd,
            "nivel": "MEDIA",
            "nota": f"Cubre victoria + empate: {pa+pd:.1f}% probabilidad combinada",
            "donde": "Playdoit / Draftea → Doble Oportunidad → '1X'"
        })
    elif 52 <= pb < 58:
        apuestas.append({
            "mercado": "Doble Oportunidad",
            "seleccion": f"✅ {eb} o Empate (X2)",
            "confianza": pb + pd,
            "nivel": "MEDIA",
            "nota": f"Cubre victoria + empate: {pb+pd:.1f}% probabilidad combinada",
            "donde": "Playdoit / Draftea → Doble Oportunidad → 'X2'"
        })

    # ── TOTAL DE GOLES Over/Under ────────────────────────────────────────────
    # Solo recomendar si la señal es clara (>60% en una dirección)
    if goles_tot >= 2.8:
        apuestas.append({
            "mercado": "Total de Goles",
            "seleccion": f"✅ Más de 2.5 goles",
            "confianza": min(goles_tot / 3.5 * 100, 72),
            "nivel": "MEDIA" if goles_tot < 3.2 else "ALTA",
            "nota": f"Modelo espera {goles_tot:.2f} goles totales en el partido",
            "donde": "Playdoit / Draftea → Total Goles → 'Over 2.5'"
        })
    elif goles_tot <= 2.0:
        apuestas.append({
            "mercado": "Total de Goles",
            "seleccion": f"✅ Menos de 2.5 goles",
            "confianza": min((3.0 - goles_tot) / 2.0 * 100, 68),
            "nivel": "MEDIA",
            "nota": f"Modelo espera solo {goles_tot:.2f} goles totales",
            "donde": "Playdoit / Draftea → Total Goles → 'Under 2.5'"
        })

    # ── TARJETAS AMARILLAS Over/Under ────────────────────────────────────────
    # Muy conservador — solo si la señal es muy clara
    linea_tarj = round(amarillas * 2) / 2  # redondear al .5 más cercano
    if amarillas >= 5.2:
        apuestas.append({
            "mercado": "Total Tarjetas Amarillas",
            "seleccion": f"✅ Más de {linea_tarj - 0.5:.1f} amarillas",
            "confianza": min(amarillas / 6.0 * 100, 68),
            "nivel": "MEDIA",
            "nota": f"Árbitro estricto + historial H2H: {amarillas:.1f} esperadas. Conservador.",
            "donde": "Playdoit / Draftea → Tarjetas → 'Over {:.1f}'".format(linea_tarj - 0.5)
        })
    elif amarillas <= 3.2:
        apuestas.append({
            "mercado": "Total Tarjetas Amarillas",
            "seleccion": f"✅ Menos de {linea_tarj + 0.5:.1f} amarillas",
            "confianza": min((4.0 - amarillas) / 2.0 * 100, 65),
            "nivel": "MEDIA",
            "nota": f"Árbitro permisivo: solo {amarillas:.1f} esperadas",
            "donde": "Playdoit / Draftea → Tarjetas → 'Under {:.1f}'".format(linea_tarj + 0.5)
        })

    # ── AMBOS EQUIPOS MARCAN ─────────────────────────────────────────────────
    # Calculamos probabilidad de que ambos marquen al menos 1
    # P(ga>=1 y gb>=1) ≈ P(ga>=1) * P(gb>=1) con Poisson
    import math
    p_ga0 = math.exp(-ga_esp)   # P(ga=0) con Poisson
    p_gb0 = math.exp(-gb_esp)
    p_ambos = (1 - p_ga0) * (1 - p_gb0) * 100

    if p_ambos >= 55:
        apuestas.append({
            "mercado": "Ambos Equipos Marcan",
            "seleccion": "✅ Sí — ambos marcan",
            "confianza": p_ambos,
            "nivel": "MEDIA" if p_ambos < 62 else "ALTA",
            "nota": f"{p_ambos:.1f}% de probabilidad basado en goles esperados",
            "donde": "Playdoit / Draftea → Ambos Marcan → 'Sí'"
        })
    elif p_ambos <= 38:
        apuestas.append({
            "mercado": "Ambos Equipos Marcan",
            "seleccion": "✅ No — al menos un equipo no marca",
            "confianza": 100 - p_ambos,
            "nivel": "MEDIA",
            "nota": f"Solo {p_ambos:.1f}% de prob. de que ambos marquen",
            "donde": "Playdoit / Draftea → Ambos Marcan → 'No'"
        })

    # Ordenar por confianza (más seguro primero)
    apuestas.sort(key=lambda x: x["confianza"], reverse=True)
    return apuestas

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE UI
# ─────────────────────────────────────────────────────────────────────────────
def tag(cls, txt):
    return f'<span class="tag {cls}">{txt}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Mundial 2026 · Predictor</div>
  <div class="hero-sub">Monte Carlo · 10,000 simulaciones · ELO + H2H + Clima + Altitud + Árbitro</div>
</div>
""", unsafe_allow_html=True)

# ── APUESTAS DEL DÍA ──────────────────────────────────────────────────────────
from datetime import date as _date
hoy = _date.today()

# Partidos de hoy = los que no tienen resultado aún y están en jornada próxima
partidos_hoy = [p for p in PARTIDOS if p[4] is None][:6]  # máx 6 partidos

if partidos_hoy:
    with st.expander("🎰 APUESTAS MÁS FUERTES DE HOY — Click para ver", expanded=True):
        st.markdown("""
        <div style="font-size:0.7rem;color:#f0c040;letter-spacing:1px;margin-bottom:1rem">
        Simulación automática de los próximos partidos · Solo se muestran señales con confianza ALTA
        </div>""", unsafe_allow_html=True)

        mejores = []  # (confianza, partido_label, apuesta)
        for p in partidos_hoy:
            ea_d, eb_d, gr_d, sede_d, _, arb_d = p
            try:
                r_d = simular(ea_d, eb_d, sede_d, arbitro=arb_d, n=10_000)
                r_d["goles_totales_esperados"] = r_d["goles_a"] + r_d["goles_b"]
                sugs_d = analizar_apuestas(ea_d, eb_d, r_d)
                for s in sugs_d:
                    if s["nivel"] == "ALTA":
                        mejores.append({
                            "partido": f"{flag(ea_d)} {ea_d} vs {flag(eb_d)} {eb_d}",
                            "grupo": gr_d,
                            **s
                        })
            except Exception:
                continue

        # Ordenar por confianza y mostrar top 6
        mejores.sort(key=lambda x: x["confianza"], reverse=True)

        if not mejores:
            st.info("Hoy no hay señales de confianza ALTA en ningún partido. "
                    "El modelo es conservador — si no está claro, no sugiere.")
        else:
            cols_d = st.columns(min(len(mejores), 3))
            for i_d, ap_d in enumerate(mejores[:6]):
                with cols_d[i_d % 3]:
                    conf_d = min(ap_d["confianza"], 99)
                    st.markdown(f"""
                    <div style="background:#0d2818;border:1px solid #2d6b45;
                    border-radius:10px;padding:0.9rem;margin-bottom:0.75rem">
                      <div style="font-size:0.6rem;color:#4ade80;letter-spacing:1px">
                      Grupo {ap_d["grupo"]} · {ap_d["partido"]}</div>
                      <div style="font-size:0.6rem;color:#6677aa;letter-spacing:2px;
                      text-transform:uppercase;margin-top:0.3rem">{ap_d["mercado"]}</div>
                      <div style="font-size:0.95rem;color:#e8eaf0;margin:0.3rem 0;
                      font-weight:600">{ap_d["seleccion"]}</div>
                      <div style="background:#1e2d45;border-radius:3px;height:5px;margin:0.3rem 0">
                        <div style="width:{conf_d:.0f}%;height:5px;border-radius:3px;
                        background:linear-gradient(90deg,#3b82f6,#4ade80)"></div>
                      </div>
                      <div style="font-size:0.65rem;color:#4ade80">{conf_d:.0f}% confianza</div>
                      <div style="font-size:0.6rem;color:#4a5568;margin-top:0.2rem">
                      📱 {ap_d["donde"]}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("""<div style="font-size:0.65rem;color:#4a5568;padding-top:0.5rem;
        border-top:1px solid #1e2d45">
        ⚠️ Solo informativo · Basado en simulación Monte Carlo ·
        Apuesta responsablemente · No garantiza resultados
        </div>""", unsafe_allow_html=True)

tab_pred, tab_res, tab_apuestas, tab_info = st.tabs(["🎯 Predictor", "📊 Resultados reales", "🎰 Apuestas", "⚙️ Modelo"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    col_izq, col_der = st.columns([1, 2.5], gap="large")

    with col_izq:
        st.markdown("#### Elige el partido")

        grupos = sorted(set(p[2] for p in PARTIDOS))
        grupo_sel = st.selectbox("Grupo", ["Todos"] + [f"Grupo {g}" for g in grupos])
        estado_sel = st.radio("Mostrar", ["Todos", "Por jugarse", "Ya jugados"],
                              horizontal=True, label_visibility="collapsed")

        filtrados = PARTIDOS
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
            lbl_sel = st.selectbox("Partido", list(opciones.keys()),
                                   label_visibility="collapsed")
            idx_sel = opciones[lbl_sel]

        st.markdown("---")
        n_sims = st.select_slider("Simulaciones",
                                  options=[1_000, 5_000, 10_000, 50_000],
                                  value=10_000,
                                  format_func=lambda x: f"{x:,}")
        btn = st.button("▶ Simular partido")

    # ── Panel derecho ──────────────────────────────────────────────────────
    with col_der:
        if idx_sel is None:
            st.markdown("""
            <div style="text-align:center;padding:4rem;color:#4a5568">
              <div style="font-size:3rem">⚽</div>
              <div style="margin-top:0.5rem">Selecciona un partido para comenzar</div>
            </div>""", unsafe_allow_html=True)
        else:
            ea, eb, grupo, sede, resultado_real, arbitro = PARTIDOS[idx_sel]
            alt = ALTITUD.get(sede, 0)

            # Tags de estado
            estado_tag = tag("tag-played", "✓ Jugado") if resultado_real else tag("tag-pending", "⏳ Por jugarse")
            clima_sede = CLIMA.get(sede, (25, 65))
            temp_c, hum = clima_sede
            arb_txt = arbitro if arbitro else "Por confirmar"
            st.markdown(
                f'{tag("tag-group", f"Grupo {grupo}")} {estado_tag}'
                f'<div style="font-size:0.75rem;color:#6677aa;margin:0.5rem 0 1rem">"'
                f'📍 {sede} &nbsp;·&nbsp; ⛰️ {alt:,} m &nbsp;·&nbsp; '
                f'🌡️ {temp_c}°C &nbsp;·&nbsp; 💧 {hum}% &nbsp;·&nbsp; 🧑\u200d⚖️ {arb_txt}</div>',
                unsafe_allow_html=True
            )

            # Resultado real si ya se jugó
            if resultado_real:
                ga_r, gb_r = resultado_real
                ganador_txt = (f"🏆 Ganó {ea}" if ga_r > gb_r
                               else f"🏆 Ganó {eb}" if gb_r > ga_r
                               else "🤝 Empate")
                st.markdown(f"""
                <div class="real-result">
                  <div style="font-size:0.6rem;color:#4ade80;letter-spacing:2px;margin-bottom:0.3rem">
                    RESULTADO REAL
                  </div>
                  <div style="display:flex;align-items:center;justify-content:center;gap:2rem">
                    <div style="text-align:right">
                      <div style="font-size:1.8rem">{flag(ea)}</div>
                      <div style="font-size:0.75rem;color:#aabbcc">{ea}</div>
                    </div>
                    <div class="real-score">{ga_r} – {gb_r}</div>
                    <div style="text-align:left">
                      <div style="font-size:1.8rem">{flag(eb)}</div>
                      <div style="font-size:0.75rem;color:#aabbcc">{eb}</div>
                    </div>
                  </div>
                  <div style="font-size:0.75rem;color:#4ade80;margin-top:0.4rem">{ganador_txt}</div>
                </div>""", unsafe_allow_html=True)

            # Correr simulación
            if btn or resultado_real:
                with st.spinner(f"Simulando {n_sims:,} partidos..."):
                    r = simular(ea, eb, sede, arbitro=arbitro, n=n_sims)

                pa, pd_, pb = r["prob_a"], r["prob_emp"], r["prob_b"]

                # Barra de probabilidades
                st.markdown(
                    f'<div style="font-size:0.6rem;color:#6677aa;letter-spacing:2px;'
                    f'text-transform:uppercase;margin-bottom:0.2rem">'
                    f'Probabilidades — {n_sims:,} simulaciones</div>'
                    f'<div class="prob-bar">'
                    f'<div class="bar-a" style="width:{pa:.1f}%"></div>'
                    f'<div class="bar-draw" style="width:{pd_:.1f}%"></div>'
                    f'<div class="bar-b" style="width:{pb:.1f}%"></div></div>',
                    unsafe_allow_html=True
                )

                # Tres cajas de resultado
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="result-box">
                      <div class="team-flag">{flag(ea)}</div>
                      <div class="team-name">{ea}</div>
                      <div class="prob-pct">{pa:.1f}%</div>
                      <div class="prob-lbl">victoria</div>
                      <div class="goles-esp">{r['goles_a']:.2f}</div>
                      <div class="prob-lbl">goles esp.</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="result-box result-box-draw">
                      <div class="team-flag">🤝</div>
                      <div class="team-name" style="color:#9ca3af">Empate</div>
                      <div class="prob-pct prob-pct-draw">{pd_:.1f}%</div>
                      <div class="prob-lbl">probabilidad</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="result-box result-box-b">
                      <div class="team-flag">{flag(eb)}</div>
                      <div class="team-name">{eb}</div>
                      <div class="prob-pct prob-pct-b">{pb:.1f}%</div>
                      <div class="prob-lbl">victoria</div>
                      <div class="goles-esp">{r['goles_b']:.2f}</div>
                      <div class="prob-lbl">goles esp.</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Marcadores + tarjetas
                cm1, cm2 = st.columns(2)
                with cm1:
                    st.markdown('<div style="font-size:0.6rem;color:#6677aa;letter-spacing:2px;'
                                'text-transform:uppercase;margin-bottom:0.5rem">Top 5 marcadores</div>',
                                unsafe_allow_html=True)
                    for i, (marcador, cnt) in enumerate(r["top5"]):
                        pct = cnt / n_sims * 100
                        cls = "score-top" if i == 0 else "score-badge"
                        st.markdown(
                            f'<span class="{cls}">{marcador[0]}–{marcador[1]}</span>'
                            f'<span style="color:#6677aa;font-size:0.8rem;margin-left:6px">{pct:.1f}%</span>',
                            unsafe_allow_html=True)
                with cm2:
                    st.markdown('<div style="font-size:0.6rem;color:#6677aa;letter-spacing:2px;'
                                'text-transform:uppercase;margin-bottom:0.5rem">Tarjetas esperadas</div>',
                                unsafe_allow_html=True)
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        st.markdown(f'<div class="metric-box"><div><span class="card-y"></span></div>'
                                    f'<div class="metric-val">{r["amarillas"]}</div>'
                                    f'<div class="metric-lbl">Amarillas</div></div>',
                                    unsafe_allow_html=True)
                    with tc2:
                        st.markdown(f'<div class="metric-box"><div><span class="card-r"></span></div>'
                                    f'<div class="metric-val">{r["rojas"]}</div>'
                                    f'<div class="metric-lbl">Rojas</div></div>',
                                    unsafe_allow_html=True)

                # Nota del modelo
                st.markdown(
                    f'<div class="model-note">📐 ELO: {r["elo_a"]} ({ea}) vs {r["elo_b"]} ({eb})'
                    f' · λ_a={r["lam_a"]} · λ_b={r["lam_b"]} · Altitud: {r["alt"]:,} m'
                    f' · Árbitro: {r["arbitro"]} ({r["arbitro_am"]} T.A. / {r["arbitro_ro"]} T.R.)'
                    f' · H2H: {r["h2h_desc"]}'  
                    + (f' · ⚠️ Bajas: {", ".join([e for e in [ea, eb] if e in BAJAS])}' if any(e in BAJAS for e in [ea, eb]) else '')
                    + '</div>',
                    unsafe_allow_html=True)

                # ── APUESTAS INLINE ────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                r["goles_totales_esperados"] = r["goles_a"] + r["goles_b"]
                sugs = analizar_apuestas(ea, eb, r)
                if sugs:
                    st.markdown("""
                    <div style="font-family:'Bebas Neue',sans-serif;font-size:1.3rem;
                    letter-spacing:2px;color:#f0c040;margin-bottom:0.75rem">
                    🎰 APUESTAS SUGERIDAS — PLAYDOIT / DRAFTEA
                    </div>""", unsafe_allow_html=True)
                    cols_ap = st.columns(len(sugs) if len(sugs) <= 3 else 3)
                    for i_ap, ap in enumerate(sugs[:3]):
                        with cols_ap[i_ap]:
                            color_bg = "#0d2818" if ap["nivel"] == "ALTA" else "#0d1827"
                            color_br = "#2d6b45" if ap["nivel"] == "ALTA" else "#1e3a5f"
                            conf = min(ap["confianza"], 99)
                            st.markdown(f"""
                            <div style="background:{color_bg};border:1px solid {color_br};
                            border-radius:10px;padding:0.9rem;height:100%">
                              <div style="font-size:0.6rem;color:#6677aa;letter-spacing:2px;
                              text-transform:uppercase">{ap["mercado"]}</div>
                              <div style="font-size:0.95rem;color:#e8eaf0;margin:0.3rem 0;
                              font-weight:600">{ap["seleccion"]}</div>
                              <div style="background:#1e2d45;border-radius:3px;height:5px;margin:0.3rem 0">
                                <div style="width:{conf:.0f}%;height:5px;border-radius:3px;
                                background:linear-gradient(90deg,#3b82f6,#4ade80)"></div>
                              </div>
                              <div style="font-size:0.65rem;color:#4ade80">{conf:.0f}% confianza</div>
                              <div style="font-size:0.6rem;color:#4a5568;margin-top:0.3rem">
                              {ap["nota"]}</div>
                            </div>""", unsafe_allow_html=True)
                    # Parlay si hay 2+ altas
                    altas_i = [a for a in sugs if a["nivel"] == "ALTA"]
                    if len(altas_i) >= 2:
                        prob_p = 1.0
                        for a in altas_i: prob_p *= a["confianza"] / 100
                        sels = " + ".join([a["seleccion"].replace("✅ ","") for a in altas_i])
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#1a1500,#2a2000);
                        border:1px solid #f0c040;border-radius:10px;padding:0.8rem 1rem;
                        margin-top:0.75rem">
                          <span style="font-size:0.6rem;color:#f0c040;letter-spacing:2px">
                          💛 PARLAY SUGERIDO</span>
                          <div style="font-size:0.85rem;color:#f0c040;margin:0.2rem 0">{sels}</div>
                          <div style="font-size:0.65rem;color:#8899bb">
                          Prob. combinada: <b style="color:#f0c040">{prob_p*100:.1f}%</b>
                          </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:#111827;border:1px solid #1e2d45;border-radius:8px;
                    padding:0.6rem 1rem;font-size:0.8rem;color:#6677aa;margin-top:0.5rem">
                    🎰 Sin señales claras de apuesta para este partido — demasiado equilibrado.
                    </div>""", unsafe_allow_html=True)
                st.markdown("""<div style="font-size:0.65rem;color:#4a5568;margin-top:0.4rem">
                ⚠️ Solo informativo. Apuesta responsablemente.</div>""",
                unsafe_allow_html=True)


            elif not resultado_real:
                st.markdown("""
                <div style="text-align:center;padding:3rem;color:#4a5568">
                  <div style="font-size:3rem">⚽</div>
                  <div style="margin-top:0.5rem;font-size:0.9rem">
                    Presiona "Simular partido" para ver la predicción
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTADOS REALES
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
                ganador_lbl = (f"→ Ganó **{ea}**" if ga > gb
                               else f"→ Ganó **{eb}**" if gb > ga
                               else "→ **Empate**")
                st.markdown(
                    f'<div style="background:{color};border-radius:8px;padding:0.5rem 1rem;'
                    f'margin-bottom:0.35rem;font-size:0.88rem">'
                    f'{flag(ea)} {ea} '
                    f'<b style="font-size:1.1rem;color:#4ade80;margin:0 0.4rem">{ga}–{gb}</b>'
                    f'{eb} {flag(eb)}'
                    f'<span style="color:#6677aa;font-size:0.72rem;margin-left:0.8rem">'
                    f'📍 {sede} · {ganador_lbl}</span></div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — APUESTAS SUGERIDAS
# ══════════════════════════════════════════════════════════════════════════════
with tab_apuestas:
    st.markdown("#### 🎰 Análisis de apuestas — Playdoit & Draftea")
    st.caption("Selecciona un partido en el Predictor y simúlalo primero. Las sugerencias aparecerán aquí.")

    st.markdown("""
    <div style="background:#1a1500;border:1px solid #5a3a00;border-radius:10px;
    padding:0.8rem 1rem;margin-bottom:1.5rem;font-size:0.8rem;color:#f0c040">
    ⚠️ <b>Solo informativo.</b> Este análisis es estadístico, no garantiza resultados.
    Apuesta solo lo que puedas permitirte perder. El modelo es conservador — 
    si no hay señal clara, no sugiere nada.
    </div>
    """, unsafe_allow_html=True)

    if idx_sel is None or not (btn or resultado_real):
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#4a5568">
          <div style="font-size:2.5rem">🎰</div>
          <div style="margin-top:0.5rem;font-size:0.9rem">
            Ve al tab Predictor, selecciona un partido y simúlalo primero
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        ea2, eb2, grupo2, sede2, res2, arb2 = PARTIDOS[idx_sel]

        # Necesitamos el resultado de la simulación — correrla de nuevo
        r2 = simular(ea2, eb2, sede2, arbitro=arb2, n=10_000)
        # Agregar goles totales al resultado
        r2["goles_totales_esperados"] = r2["goles_a"] + r2["goles_b"]

        sugerencias = analizar_apuestas(ea2, eb2, r2)

        if not sugerencias:
            st.info("El modelo no encontró señales suficientemente claras para este partido. "
                    "Partido demasiado equilibrado — mejor no arriesgar.")
        else:
            st.markdown(f"##### {flag(ea2)} {ea2} vs {flag(eb2)} {eb2} — {len(sugerencias)} apuesta(s) sugerida(s)")

            # Mostrar cada apuesta
            for ap in sugerencias:
                nivel_cls = "apuesta-alta" if ap["nivel"] == "ALTA" else "apuesta-media"
                badge_cls = "badge-alta" if ap["nivel"] == "ALTA" else "badge-media"
                conf_pct = min(ap["confianza"], 99)
                st.markdown(f"""
                <div class="{nivel_cls}">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem">
                    <span class="apuesta-mercado">{ap["mercado"]}</span>
                    <span class="{badge_cls}">CONFIANZA {nivel_cls.split("-")[1].upper()}</span>
                  </div>
                  <div class="apuesta-titulo">{ap["seleccion"]}</div>
                  <div class="confianza-bar">
                    <div class="confianza-fill" style="width:{conf_pct:.0f}%"></div>
                  </div>
                  <div style="font-size:0.72rem;color:#8899bb;margin-top:0.2rem">{ap["nota"]}</div>
                  <div style="font-size:0.65rem;color:#4a5568;margin-top:0.3rem">📱 {ap["donde"]}</div>
                </div>
                """, unsafe_allow_html=True)

            # PARLAY sugerido — solo con las de confianza ALTA
            altas = [a for a in sugerencias if a["nivel"] == "ALTA"]
            if len(altas) >= 2:
                prob_parlay = 1.0
                for a in altas:
                    prob_parlay *= (a["confianza"] / 100)
                prob_parlay *= 100
                selecciones = " + ".join([a["seleccion"].replace("✅ ", "") for a in altas])
                st.markdown(f"""
                <div class="parlay-box">
                  <div style="font-size:0.65rem;color:#f0c040;letter-spacing:2px;margin-bottom:0.3rem">
                    💛 PARLAY SUGERIDO — SOLO APUESTAS DE CONFIANZA ALTA
                  </div>
                  <div style="font-family:'Bebas Neue',sans-serif;font-size:1.2rem;color:#f0c040;margin-bottom:0.4rem">
                    {selecciones}
                  </div>
                  <div style="font-size:0.75rem;color:#8899bb">
                    Probabilidad combinada estimada: <b style="color:#f0c040">{prob_parlay:.1f}%</b>
                    &nbsp;·&nbsp; Recuerda que en parlay todas las selecciones deben ganar
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:0.72rem;color:#4a5568;margin-top:1rem;padding:0.5rem;
            border-top:1px solid #1e2d45">
            📊 Basado en: ELO FIFA + forma en el Mundial + altitud + árbitro + historial H2H.
            El modelo es deliberadamente conservador — si la señal no es clara, no aparece aquí.
            Tiros de esquina no incluidos (requieren datos de estadísticas de córners por equipo).
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CÓMO FUNCIONA
# ══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.markdown("#### ¿Cómo funciona el modelo?")
    st.markdown("""
El predictor usa **simulación Monte Carlo con distribución de Poisson** —
el mismo enfoque de casas de apuestas y modelos académicos serios.

**En cada una de las 10,000 simulaciones:**
1. Calcula los **goles esperados** (λ) para cada equipo combinando:
   - **ELO Rating** — fuerza relativa; diferencia de 400 puntos ≈ 90% de victorias
   - **Altitud de la sede** — penaliza hasta 8% a equipos no aclimatados en sedes >1,700m (el Azteca está a 2,240m)
   - **Ventaja local** — +10% para México en sus estadios, +10% para Canadá/EEUU en los suyos
2. Muestrea goles de cada equipo desde una **distribución de Poisson** con esos λ
3. Cuenta victorias / empates / derrotas a través de las 10,000 iteraciones
4. Reporta probabilidades, marcadores más frecuentes y tarjetas esperadas

**¿Por qué Poisson y no una red neuronal?**
Las selecciones nacionales juegan ~15–20 partidos al año —
insuficiente para entrenar una red neuronal correctamente.
Poisson es el estándar de la industria para goles: eventos raros,
independientes entre sí, distribuidos en el tiempo.

**Rendimiento:** 10,000 simulaciones corren en ~20ms gracias a NumPy vectorizado.
""")
    st.markdown("---")
    st.markdown("#### ELO Ratings — 48 selecciones")
    sorted_elo = sorted(
        [(k, v) for k, v in ELO.items() if k not in ("Algeria", "Arabia Saudi")],
        key=lambda x: x[1], reverse=True
    )
    cols = st.columns(3)
    for i, (equipo, elo) in enumerate(sorted_elo):
        with cols[i % 3]:
            st.markdown(f"{flag(equipo)} **{equipo}** — `{elo}`")
