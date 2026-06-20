"""
Módulo Dixon-Coles Bayesiano para predicción de fútbol
=======================================================
Parámetros entrenados con ADVI (Variational Inference) sobre 4,277 partidos
internacionales de selecciones del Mundial 2026 (2010-2026).

El modelo calcula:
  λ_a = exp(mu + ataque_a - defensa_b + ventaja_local)
  λ_b = exp(mu + ataque_b - defensa_a)

Con corrección Dixon-Coles ρ para scores bajos (0-0, 1-0, 0-1, 1-1)
que corrige la subestimación de Poisson en partidos muy defensivos.

El Monte Carlo bivariado usa la distribución de Poisson corregida
en vez de dos Poisson independientes.
"""

import json
import numpy as np
import os

# ── Parámetro de Dixon-Coles ρ ────────────────────────────────────────────────
# Calibrado empíricamente sobre datos 2010-2026
# Rango típico en literatura: 0.05-0.15
RHO = 0.08

# Mapa de nombres app → nombres en el dataset de entrenamiento
NOMBRES_DC = {
    "Mexico":               "Mexico",
    "Sudafrica":            "South Africa",
    "Corea del Sur":        "South Korea",
    "Chequia":              "Czech Republic",
    "Canada":               "Canada",
    "Bosnia y Herzegovina": "Bosnia and Herzegovina",
    "Catar":                "Qatar",
    "Suiza":                "Switzerland",
    "Brasil":               "Brazil",
    "Marruecos":            "Morocco",
    "Haiti":                "Haiti",
    "Escocia":              "Scotland",
    "Estados Unidos":       "United States",
    "Paraguay":             "Paraguay",
    "Australia":            "Australia",
    "Turquia":              "Turkey",
    "Alemania":             "Germany",
    "Curazao":              "Curaçao",
    "Costa de Marfil":      "Ivory Coast",
    "Ecuador":              "Ecuador",
    "Paises Bajos":         "Netherlands",
    "Japon":                "Japan",
    "Suecia":               "Sweden",
    "Tunez":                "Tunisia",
    "Belgica":              "Belgium",
    "Egipto":               "Egypt",
    "Iran":                 "IR Iran",
    "Nueva Zelanda":        "New Zealand",
    "Espana":               "Spain",
    "Cabo Verde":           "Cape Verde",
    "Arabia Saudi":         "Saudi Arabia",
    "Arabia Saudita":       "Saudi Arabia",
    "Uruguay":              "Uruguay",
    "Francia":              "France",
    "Senegal":              "Senegal",
    "Irak":                 "Iraq",
    "Noruega":              "Norway",
    "Argentina":            "Argentina",
    "Algeria":              "Algeria",
    "Argelia":              "Algeria",
    "Austria":              "Austria",
    "Jordania":             "Jordan",
    "Portugal":             "Portugal",
    "RD Congo":             "DR Congo",
    "Uzbekistan":           "Uzbekistan",
    "Colombia":             "Colombia",
    "Inglaterra":           "England",
    "Croacia":              "Croatia",
    "Ghana":                "Ghana",
    "Panama":               "Panama",
}

# ── Estado global del módulo ──────────────────────────────────────────────────
_params = None
_eq_idx = None


def _cargar():
    global _params, _eq_idx
    if _params is not None:
        return True
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        ruta = os.path.join(base, 'dixon_coles_params.json')
        with open(ruta) as f:
            _params = json.load(f)
        _eq_idx = {e: i for i, e in enumerate(_params['equipos'])}
        return True
    except Exception:
        return False


def _tau(x, y, lam_a, lam_b, rho=RHO):
    """Factor de corrección Dixon-Coles para scores bajos."""
    if x == 0 and y == 0:
        return max(1 - lam_a * lam_b * rho, 0.01)
    elif x == 1 and y == 0:
        return 1 + lam_b * rho
    elif x == 0 and y == 1:
        return 1 + lam_a * rho
    elif x == 1 and y == 1:
        return max(1 - rho, 0.01)
    return 1.0


def calcular_lambdas_dc(
    equipo_a: str,
    equipo_b: str,
    es_neutral: bool = True,
    forma_mundial_a: dict = None,
    forma_mundial_b: dict = None,
    bajas_a: float = 1.0,
    bajas_b: float = 1.0,
) -> tuple:
    """
    Calcula lambdas usando el modelo Dixon-Coles Bayesiano.

    Args:
        equipo_a: nombre del equipo local (en formato app)
        equipo_b: nombre del equipo visitante
        es_neutral: True si es sede neutral
        forma_mundial_a/b: dict con {'gf': x, 'gc': y, 'pj': n}
        bajas_a/b: factor de penalización por lesiones (0.85-1.0)

    Returns:
        (lam_a, lam_b, info_dict) o (None, None, {}) si no hay datos
    """
    if not _cargar():
        return None, None, {}

    # Mapear nombres
    nombre_a = NOMBRES_DC.get(equipo_a, equipo_a)
    nombre_b = NOMBRES_DC.get(equipo_b, equipo_b)

    idx_a = _eq_idx.get(nombre_a)
    idx_b = _eq_idx.get(nombre_b)

    if idx_a is None or idx_b is None:
        return None, None, {'error': f'Equipo no encontrado: {nombre_a if idx_a is None else nombre_b}'}

    mu      = _params['mu_base']
    ventaja = _params['ventaja_local']
    at_a    = _params['ataque_media'][idx_a]
    df_a    = _params['defensa_media'][idx_a]
    at_b    = _params['ataque_media'][idx_b]
    df_b    = _params['defensa_media'][idx_b]

    # Lambdas base Dixon-Coles
    lam_a = np.exp(mu + at_a - df_b + (0 if es_neutral else ventaja))
    lam_b = np.exp(mu + at_b - df_a)

    # Ajuste por forma en el Mundial (pesos progresivos)
    if forma_mundial_a and forma_mundial_a.get('pj', 0) > 0:
        pj = forma_mundial_a['pj']
        gf_prom = forma_mundial_a['gf'] / pj
        factor_forma = 0.10 * pj  # 10% por partido jugado, máx 30%
        factor_forma = min(factor_forma, 0.30)
        ratio = gf_prom / max(lam_a, 0.5)
        lam_a = lam_a * (1 - factor_forma) + lam_a * ratio * factor_forma

    if forma_mundial_b and forma_mundial_b.get('pj', 0) > 0:
        pj = forma_mundial_b['pj']
        gf_prom = forma_mundial_b['gf'] / pj
        factor_forma = min(0.10 * pj, 0.30)
        ratio = gf_prom / max(lam_b, 0.5)
        lam_b = lam_b * (1 - factor_forma) + lam_b * ratio * factor_forma

    # Aplicar bajas
    lam_a = max(lam_a * bajas_a, 0.15)
    lam_b = max(lam_b * bajas_b, 0.15)

    info = {
        'ataque_a':  round(at_a, 3),
        'defensa_a': round(df_a, 3),
        'ataque_b':  round(at_b, 3),
        'defensa_b': round(df_b, 3),
        'modelo':    'Dixon-Coles ADVI',
        'rho':       RHO,
    }

    return lam_a, lam_b, info


def simular_dc(
    lam_a: float,
    lam_b: float,
    n: int = 10_000_000,
    rho: float = RHO,
) -> tuple:
    """
    Simula n partidos usando distribución de Poisson bivariada Dixon-Coles.
    La corrección ρ se implementa por rejection sampling sobre scores bajos.

    Returns:
        (ga, gb) arrays de goles simulados
    """
    rng = np.random.default_rng()

    # Generar goles base con Poisson independiente
    ga = rng.poisson(lam_a, n)
    gb = rng.poisson(lam_b, n)

    # Aplicar corrección Dixon-Coles por rejection sampling
    # Para cada score bajo, aceptar/rechazar según τ
    # Esto modifica levemente la distribución de scores bajos

    # Máscara de scores bajos (los únicos que cambia DC)
    bajos = ((ga <= 1) & (gb <= 1))
    n_bajos = bajos.sum()

    if n_bajos > 0:
        ga_b = ga[bajos]
        gb_b = gb[bajos]

        # Calcular τ para cada uno
        taus = np.ones(n_bajos)
        m00 = (ga_b == 0) & (gb_b == 0)
        m10 = (ga_b == 1) & (gb_b == 0)
        m01 = (ga_b == 0) & (gb_b == 1)
        m11 = (ga_b == 1) & (gb_b == 1)

        taus[m00] = max(1 - lam_a * lam_b * rho, 0.01)
        taus[m10] = 1 + lam_b * rho
        taus[m01] = 1 + lam_a * rho
        taus[m11] = max(1 - rho, 0.01)

        # Normalizar τ a probabilidades de aceptación
        tau_max = max(taus.max(), 1.0)
        accept_prob = taus / tau_max
        u = rng.uniform(0, 1, n_bajos)
        rechazados = u > accept_prob

        # Re-muestrear los rechazados con Poisson normal
        n_rech = rechazados.sum()
        if n_rech > 0:
            ga_new = rng.poisson(lam_a, n_rech)
            gb_new = rng.poisson(lam_b, n_rech)
            idx_rech = np.where(bajos)[0][rechazados]
            ga[idx_rech] = ga_new
            gb[idx_rech] = gb_new

    return ga, gb


def disponible() -> bool:
    return _cargar()
