"""
Módulo XGBoost para predicción de goles del Mundial 2026
=========================================================
Reemplaza calcular_lambdas() con un modelo entrenado en 25,000+
partidos internacionales (2000-2026).

El modelo aprende patrones que la fórmula manual no captura:
- Que los sudamericanos rinden mejor en calor y altitud
- Que los europeos bajan más de lo esperado en sedes de alta altitud
- El impacto real del torneo (un partido de Copa del Mundo
  genera patrones distintos a un amistoso)
- Interacciones entre confederaciones (CAF vs UEFA, etc.)
"""

import numpy as np
import json
import os

# ── Mapas de confederaciones ──────────────────────────────────────────────────
CONF_MAP = {'UEFA':0,'CONMEBOL':1,'CONCACAF':2,'CAF':3,'AFC':4,'OFC':5,'OTHER':6}

CONFEDERACIONES = {
    # UEFA
    'England':'UEFA','Germany':'UEFA','France':'UEFA','Spain':'UEFA',
    'Italy':'UEFA','Portugal':'UEFA','Netherlands':'UEFA','Belgium':'UEFA',
    'Croatia':'UEFA','Switzerland':'UEFA','Denmark':'UEFA','Sweden':'UEFA',
    'Norway':'UEFA','Scotland':'UEFA','Austria':'UEFA','Czech Republic':'UEFA',
    'Poland':'UEFA','Turkey':'UEFA','Romania':'UEFA','Serbia':'UEFA',
    'Bosnia and Herzegovina':'UEFA','Slovenia':'UEFA','Slovakia':'UEFA',
    # CONMEBOL
    'Brazil':'CONMEBOL','Argentina':'CONMEBOL','Uruguay':'CONMEBOL',
    'Colombia':'CONMEBOL','Chile':'CONMEBOL','Peru':'CONMEBOL',
    'Ecuador':'CONMEBOL','Paraguay':'CONMEBOL','Bolivia':'CONMEBOL',
    # CONCACAF
    'Mexico':'CONCACAF','United States':'CONCACAF','Canada':'CONCACAF',
    'Costa Rica':'CONCACAF','Honduras':'CONCACAF','Panama':'CONCACAF',
    'Haiti':'CONCACAF','Jamaica':'CONCACAF','Curaçao':'CONCACAF',
    # CAF
    'Morocco':'CAF','Senegal':'CAF','Nigeria':'CAF','Ghana':'CAF',
    'Cameroon':'CAF','Egypt':'CAF','Ivory Coast':'CAF','Algeria':'CAF',
    'Tunisia':'CAF','Mali':'CAF','South Africa':'CAF','DR Congo':'CAF',
    'Cape Verde':'CAF',
    # AFC
    'Japan':'AFC','South Korea':'AFC','Australia':'AFC','Iran':'AFC',
    'Saudi Arabia':'AFC','Qatar':'AFC','Iraq':'AFC','Uzbekistan':'AFC',
    'Jordan':'AFC',
    # OFC
    'New Zealand':'OFC',
}

ALTITUDES_SEDE = {
    'Mexico': 2240,      # Azteca — CDMX
    'Guadalajara': 1566, # Akron
    'Monterrey': 540,
    'Atlanta': 320,
    'Kansas City': 270,
    'Dallas': 180,
    'Los Angeles': 25,
    'Toronto': 76,
    'Boston': 65,
    'Philadelphia': 12,
    'Seattle': 10,
    'Houston': 14,
    'San Francisco': 11,
    'Miami': 3,
    'Nueva York': 2,
    'Vancouver': 2,
}

PESOS_TORNEO = {
    'FIFA World Cup': 3.0,
    'Copa América': 2.5,
    'UEFA Euro': 2.5,
    'African Cup of Nations': 2.5,
    'AFC Asian Cup': 2.5,
    'Gold Cup': 2.0,
    'UEFA Nations League': 1.8,
    'CONCACAF Nations League': 1.8,
    'FIFA World Cup qualification': 1.5,
    'Friendly': 0.6,
}

# Forma actualizada del Mundial 2026 (se actualiza conforme avanza el torneo)
FORMA_MUNDIAL_2026 = {
    # Jornada 1 y 2 completadas
    'Mexico':            {'gf': 1.5, 'gc': 0.0},
    'South Korea':       {'gf': 1.0, 'gc': 1.0},
    'Czech Republic':    {'gf': 1.0, 'gc': 1.5},
    'South Africa':      {'gf': 0.5, 'gc': 1.5},
    'Canada':            {'gf': 3.5, 'gc': 0.5},
    'Qatar':             {'gf': 0.5, 'gc': 3.5},
    'Switzerland':       {'gf': 2.5, 'gc': 1.0},
    'Bosnia and Herzegovina': {'gf': 1.0, 'gc': 2.5},
    'Brazil':            {'gf': 1.0, 'gc': 1.0},
    'Morocco':           {'gf': 1.0, 'gc': 1.0},
    'Haiti':             {'gf': 0.0, 'gc': 1.0},
    'Scotland':          {'gf': 1.0, 'gc': 0.0},
    'United States':     {'gf': 4.0, 'gc': 1.0},
    'Paraguay':          {'gf': 1.0, 'gc': 4.0},
    'Australia':         {'gf': 2.0, 'gc': 0.0},
    'Turkey':            {'gf': 0.0, 'gc': 2.0},
    'Germany':           {'gf': 7.0, 'gc': 1.0},
    'Curaçao':           {'gf': 1.0, 'gc': 7.0},
    'Ivory Coast':       {'gf': 1.0, 'gc': 0.0},
    'Ecuador':           {'gf': 0.0, 'gc': 1.0},
    'Netherlands':       {'gf': 2.0, 'gc': 2.0},
    'Japan':             {'gf': 2.0, 'gc': 2.0},
    'Sweden':            {'gf': 5.0, 'gc': 1.0},
    'Tunisia':           {'gf': 1.0, 'gc': 5.0},
    'Belgium':           {'gf': 1.0, 'gc': 1.0},
    'Egypt':             {'gf': 1.0, 'gc': 1.0},
    'Iran':              {'gf': 2.0, 'gc': 2.0},
    'New Zealand':       {'gf': 2.0, 'gc': 2.0},
    'Spain':             {'gf': 0.0, 'gc': 0.0},
    'Cape Verde':        {'gf': 0.0, 'gc': 0.0},
    'Saudi Arabia':      {'gf': 1.0, 'gc': 1.0},
    'Uruguay':           {'gf': 1.0, 'gc': 1.0},
    'France':            {'gf': 3.0, 'gc': 1.0},
    'Senegal':           {'gf': 1.0, 'gc': 3.0},
    'Iraq':              {'gf': 1.0, 'gc': 4.0},
    'Norway':            {'gf': 4.0, 'gc': 1.0},
    'Argentina':         {'gf': 3.0, 'gc': 0.0},
    'Algeria':           {'gf': 0.0, 'gc': 3.0},
    'Austria':           {'gf': 3.0, 'gc': 1.0},
    'Jordan':            {'gf': 1.0, 'gc': 3.0},
    'Portugal':          {'gf': 1.0, 'gc': 1.0},
    'DR Congo':          {'gf': 1.0, 'gc': 1.0},
    'Uzbekistan':        {'gf': 1.0, 'gc': 3.0},
    'Colombia':          {'gf': 3.0, 'gc': 1.0},
    'England':           {'gf': 4.0, 'gc': 2.0},
    'Croatia':           {'gf': 2.0, 'gc': 4.0},
    'Ghana':             {'gf': 1.0, 'gc': 0.0},
    'Panama':            {'gf': 0.0, 'gc': 1.0},
}

_model_home = None
_model_away = None
_elos = None

def _cargar_modelos():
    global _model_home, _model_away, _elos
    if _model_home is not None:
        return True
    try:
        from xgboost import XGBRegressor
        base = os.path.dirname(os.path.abspath(__file__))
        _model_home = XGBRegressor()
        _model_away = XGBRegressor()
        _model_home.load_model(os.path.join(base, 'model_home.json'))
        _model_away.load_model(os.path.join(base, 'model_away.json'))
        with open(os.path.join(base, 'elos_2026.json')) as f:
            _elos = json.load(f)
        return True
    except Exception as e:
        return False


def calcular_lambdas_xgb(
    equipo_a: str,
    equipo_b: str,
    sede: str,
    es_neutral: bool = True,
    torneo: str = 'FIFA World Cup',
    mes: int = 6,
    bajas_a: float = 1.0,
    bajas_b: float = 1.0,
) -> tuple:
    """
    Calcula los goles esperados usando XGBoost entrenado en 25,000+ partidos.
    Devuelve (lam_a, lam_b, info_dict) igual que calcular_lambdas() del app.py
    """
    if not _cargar_modelos():
        return None, None, {}

    # ELO del dataset histórico (más preciso que los manuales)
    elo_h = _elos.get(equipo_a, 1500.0)
    elo_a = _elos.get(equipo_b, 1500.0)
    elo_diff = elo_h - elo_a

    conf_h = CONF_MAP.get(CONFEDERACIONES.get(equipo_a, 'OTHER'), 6)
    conf_a = CONF_MAP.get(CONFEDERACIONES.get(equipo_b, 'OTHER'), 6)
    mismo_conf = int(conf_h == conf_a)

    alt = ALTITUDES_SEDE.get(sede, 100)
    alt_factor = max(0, (alt - 1700)) / 1000

    peso = PESOS_TORNEO.get(torneo, 1.5)

    forma_h = FORMA_MUNDIAL_2026.get(equipo_a, {'gf': 1.5, 'gc': 1.2})
    forma_a = FORMA_MUNDIAL_2026.get(equipo_b, {'gf': 1.5, 'gc': 1.2})

    feats = np.array([[
        elo_diff, elo_h, elo_a,
        forma_h['gf'], forma_h['gc'],
        forma_a['gf'], forma_a['gc'],
        int(es_neutral), conf_h, conf_a, mismo_conf,
        alt_factor, peso, mes
    ]])

    lam_a = float(_model_home.predict(feats)[0]) * bajas_a
    lam_b = float(_model_away.predict(feats)[0]) * bajas_b

    # Clip a valores razonables
    lam_a = max(min(lam_a, 5.0), 0.1)
    lam_b = max(min(lam_b, 5.0), 0.1)

    info = {
        'elo_a': round(elo_h, 0),
        'elo_b': round(elo_a, 0),
        'elo_diff': round(elo_diff, 0),
        'alt': alt,
        'modelo': 'XGBoost (25k partidos)',
    }

    return lam_a, lam_b, info


def disponible() -> bool:
    return _cargar_modelos()
