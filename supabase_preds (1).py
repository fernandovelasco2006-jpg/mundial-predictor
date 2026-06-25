"""
Módulo de predicciones persistentes via Supabase
=================================================
Guarda y recupera predicciones del Mundial 2026 en PostgreSQL (Supabase).
Funciona tanto en Streamlit Cloud como en versión local (desktop).
Las predicciones se guardan UNA VEZ por partido — no se sobreescriben.
"""

import requests
import json
from datetime import datetime, timezone, timedelta

TZ_MX = timezone(timedelta(hours=-6))

def _headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

def guardar_prediccion(url: str, key: str, pred: dict) -> bool:
    try:
        r_check = requests.get(
            f"{url}/rest/v1/predicciones",
            headers={**_headers(key), "Prefer": ""},
            params={"id": f"eq.{pred['id']}", "select": "id"},
            timeout=8
        )
        if r_check.status_code == 200 and r_check.json():
            return False
        r = requests.post(
            f"{url}/rest/v1/predicciones",
            headers=_headers(key),
            json=pred,
            timeout=8
        )
        return r.status_code in (200, 201)
    except Exception:
        return False


def cargar_predicciones(url: str, key: str, fecha: str = None) -> dict:
    try:
        params = {"select": "*", "order": "fecha_partido.desc,guardada_en.desc"}
        if fecha:
            params["fecha_partido"] = f"eq.{fecha}"
        r = requests.get(
            f"{url}/rest/v1/predicciones",
            headers={**_headers(key), "Prefer": ""},
            params=params,
            timeout=10
        )
        if r.status_code == 200:
            return {p["id"]: p for p in r.json()}
        return {}
    except Exception:
        return {}


def cargar_todas_predicciones(url: str, key: str) -> dict:
    return cargar_predicciones(url, key, fecha=None)


def simular_y_guardar_dia(url: str, key: str, partidos_hoy: list,
                           fn_simular, HORARIOS_PARTIDO: dict,
                           fn_toast=None) -> int:
    from datetime import datetime as dt
    tz = timezone(timedelta(hours=-6))
    ahora = dt.now(tz)
    nuevas = 0

    for p in partidos_hoy:
        ea, eb, grupo, sede, _, arbitro = p
        pred_id = f"pred_{ea}_{eb}".replace(" ", "_")
        horario = (HORARIOS_PARTIDO.get((ea, eb)) or
                   HORARIOS_PARTIDO.get((eb, ea), ""))
        fecha_partido = horario[:10] if horario else ahora.strftime("%Y-%m-%d")

        try:
            r = fn_simular(ea, eb, sede, arbitro=arbitro, n=500_000)
            pred = {
                "id":            pred_id,
                "ea":            ea,
                "eb":            eb,
                "grupo":         grupo,
                "guardada_en":   ahora.strftime("%Y-%m-%d %H:%M"),
                "tipo":          "automatica_mañana",
                "prob_a":        round(r["prob_a"], 1),
                "prob_emp":      round(r["prob_emp"], 1),
                "prob_b":        round(r["prob_b"], 1),
                "favorito":      ea if r["prob_a"] > r["prob_b"] else eb,
                "prob_fav":      round(max(r["prob_a"], r["prob_b"]), 1),
                "goles_a_esp":   round(r["goles_a"], 2),
                "goles_b_esp":   round(r["goles_b"], 2),
                "lam_a":         round(r.get("lam_a", 0), 3),
                "lam_b":         round(r.get("lam_b", 0), 3),
                "modelo":        r.get("modelo", "Dixon-Coles"),
                "arbitro":       arbitro or "desconocido",
                "prob_over05":   round(r.get("prob_over05", 0), 1),
                "prob_over15":   round(r.get("prob_over15", 0), 1),
                "prob_over25":   round(r.get("prob_over25", 0), 1),
                "amarillas_esp": round(r.get("amarillas", 0), 2),
                "corners_esp":   round(r.get("corners_esp", 0), 1),
                "fecha_partido": fecha_partido,
            }
            guardado = guardar_prediccion(url, key, pred)
            if guardado:
                nuevas += 1
        except Exception:
            continue

    return nuevas


def calcular_accuracy(predicciones: dict, partidos_con_resultado: list) -> dict:
    detalle = []
    acertados_g = 0
    acertados_o = 0

    resultados = {}
    for p in partidos_con_resultado:
        if p[4] is not None:
            resultados[(p[0], p[1])] = p[4]

    for pred_id, pred in predicciones.items():
        if not pred.get("ea") or pred.get("tipo") == "test":
            continue
        ea, eb = pred["ea"], pred["eb"]
        clave = (ea, eb)
        if clave not in resultados:
            continue

        ga, gb = resultados[clave]
        ganador_real = ea if ga > gb else (eb if gb > ga else "Empate")
        over25_real  = (ga + gb) > 2
        over25_pred  = pred.get("prob_over25", 0) > 50
        correcto_g = pred.get("favorito") == ganador_real
        correcto_o = over25_real == over25_pred

        if correcto_g: acertados_g += 1
        if correcto_o: acertados_o += 1

        detalle.append({
            "ea": ea, "eb": eb,
            "grupo": pred.get("grupo", "?"),
            "guardada_en": pred.get("guardada_en", "?"),
            "favorito": pred.get("favorito", "?"),
            "prob_fav": pred.get("prob_fav", 0),
            "resultado": f"{ga}-{gb}",
            "ganador_real": ganador_real,
            "correcto": correcto_g,
            "over25_real": over25_real,
            "over25_pred": over25_pred,
            "correcto_o25": correcto_o,
            "modelo": pred.get("modelo", "?"),
        })

    total = len(detalle)
    return {
        "total": total,
        "acertados_ganador": acertados_g,
        "acertados_over25": acertados_o,
        "accuracy_ganador": round(acertados_g / total * 100, 1) if total else 0,
        "accuracy_over25": round(acertados_o / total * 100, 1) if total else 0,
        "detalle": sorted(detalle, key=lambda x: x["guardada_en"], reverse=True),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HISTORIAL DE APUESTAS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_apuestas_dia(url: str, key: str, partidos_hoy: list,
                          fn_simular, fn_analizar, HORARIOS_PARTIDO: dict) -> int:
    from datetime import datetime as dt
    tz = timezone(timedelta(hours=-6))
    ahora = dt.now(tz)
    guardadas = 0

    for p in partidos_hoy:
        ea, eb, grupo, sede, _, arbitro = p
        horario = (HORARIOS_PARTIDO.get((ea, eb)) or
                   HORARIOS_PARTIDO.get((eb, ea), ""))
        fecha_partido = horario[:10] if horario else ahora.strftime("%Y-%m-%d")

        try:
            r = fn_simular(ea, eb, sede, arbitro=arbitro, n=500_000)
            r["goles_totales_esperados"] = r["goles_a"] + r["goles_b"]
            sugs = fn_analizar(ea, eb, r)

            for i, s in enumerate(sugs):
                if s["nivel"] != "ALTA":
                    continue

                ap_id = f"ap_{ea}_{eb}_{i}_{fecha_partido}".replace(" ", "_")
                ap = {
                    "id":             ap_id,
                    "ea":             ea,
                    "eb":             eb,
                    "grupo":          grupo,
                    "fecha_partido":  fecha_partido,
                    "guardada_en":    ahora.strftime("%Y-%m-%d %H:%M"),
                    "mercado":        s["mercado"],
                    "seleccion":      s["seleccion"].replace("✅ ", ""),
                    "confianza":      round(s["confianza"], 1),
                    "donde":          s.get("donde", "Playdoit / Draftea"),
                    "resultado_real": None,
                    "goles_a":        None,
                    "goles_b":        None,
                    "acierto":        None,
                }

                try:
                    r_check = requests.get(
                        f"{url}/rest/v1/apuestas_historial",
                        headers={**_headers(key), "Prefer": ""},
                        params={"id": f"eq.{ap_id}", "select": "id"},
                        timeout=8
                    )
                    if r_check.status_code == 200 and r_check.json():
                        continue
                    r_ins = requests.post(
                        f"{url}/rest/v1/apuestas_historial",
                        headers=_headers(key),
                        json=ap,
                        timeout=8
                    )
                    if r_ins.status_code in (200, 201):
                        guardadas += 1
                except Exception:
                    continue

        except Exception:
            continue

    return guardadas


def actualizar_aciertos(url: str, key: str, partidos_con_resultado: list) -> int:
    """
    Actualiza aciertos de apuestas para todos los partidos con resultado.
    Re-evalúa también las que tienen acierto=False por si estaban mal calculadas.
    """
    actualizadas = 0

    for p in partidos_con_resultado:
        ea, eb = p[0], p[1]
        resultado = p[4]
        if resultado is None:
            continue

        ga, gb = resultado

        try:
            # Buscar TODAS las apuestas de este partido (con y sin acierto)
            # para poder re-evaluar las que estaban mal
            import urllib.parse as _up
            _params = f"ea=eq.{_up.quote(ea)}&eb=eq.{_up.quote(eb)}&select=*"
            r = requests.get(
                f"{url}/rest/v1/apuestas_historial?{_params}",
                headers={**_headers(key), "Prefer": ""},
                timeout=10
            )
            if r.status_code != 200:
                continue

            apuestas = r.json()
            for ap in apuestas:
                acierto_nuevo = _evaluar_acierto(ap, ga, gb)
                resultado_str = f"{ga}-{gb}"

                # Solo actualizar si no tiene resultado aún O si el acierto cambió
                # (para corregir evaluaciones anteriores incorrectas)
                acierto_actual = ap.get("acierto")
                if acierto_actual is None or acierto_actual != acierto_nuevo:
                    requests.patch(
                        f"{url}/rest/v1/apuestas_historial",
                        headers={**_headers(key), "Prefer": ""},
                        params={"id": f"eq.{ap['id']}"},
                        json={
                            "acierto":        acierto_nuevo,
                            "goles_a":        ga,
                            "goles_b":        gb,
                            "resultado_real": resultado_str,
                        },
                        timeout=8
                    )
                    actualizadas += 1

        except Exception:
            continue

    return actualizadas


def _evaluar_acierto(ap: dict, ga: int, gb: int, am_reales: int = None, co_reales: int = None) -> bool:
    """
    Evalúa si una apuesta acertó dado el resultado real (ga-gb).

    Mercados soportados:
    - Total Goles: Over/Under 0.5, 1.5, 2.5, 3.5
    - Resultado (1X2): Gana A, Gana B, Empate
    - Doble Oportunidad: 1X, X2, 12
    - Ambos Marcan: Sí / No
    - Tarjetas Amarillas: Over/Under 1.5, 2.5, 3.5, 4.5
      ⚠️ Las tarjetas guardadas en Supabase NO incluyen el conteo real,
         así que necesitamos el campo "amarillas_reales" si existe,
         o la evaluamos como None (pendiente) hasta que se agregue.
    - Córners: Over/Under — misma situación, requiere dato externo.
    """
    sel  = ap.get("seleccion", "").lower()
    merc = ap.get("mercado", "")
    ea   = ap.get("ea", "")
    eb   = ap.get("eb", "")
    goles_tot = ga + gb

    # ── TOTAL GOLES ───────────────────────────────────────────────────────────
    if merc == "Total Goles":
        if "over 0.5"  in sel: return goles_tot >= 1
        if "over 1.5"  in sel: return goles_tot >= 2
        if "over 2.5"  in sel: return goles_tot >= 3
        if "over 3.5"  in sel: return goles_tot >= 4
        if "under 1.5" in sel: return goles_tot <= 1
        if "under 2.5" in sel: return goles_tot <= 2
        if "under 3.5" in sel: return goles_tot <= 3

    # ── RESULTADO 1X2 ─────────────────────────────────────────────────────────
    elif merc == "Resultado (1X2)":
        # sel contiene "gana {equipo}" — comparar en minúsculas
        ea_l = ea.lower()
        eb_l = eb.lower()
        if ea_l in sel and "gana" in sel: return ga > gb
        if eb_l in sel and "gana" in sel: return gb > ga
        if "empate" in sel:               return ga == gb

    # ── DOBLE OPORTUNIDAD ─────────────────────────────────────────────────────
    elif merc == "Doble Oportunidad":
        # Formato guardado: "Inglaterra o Empate (1X)" / "Ghana o Empate (X2)"
        if "(1x)" in sel: return ga >= gb   # local no pierde
        if "(x2)" in sel: return gb >= ga   # visitante no pierde
        if "(12)" in sel: return ga != gb   # no hay empate
        # Fallback por si el formato varía
        ea_l = ea.lower()
        eb_l = eb.lower()
        if ea_l in sel and "empate" in sel: return ga >= gb
        if eb_l in sel and "empate" in sel: return gb >= ga

    # ── AMBOS MARCAN ──────────────────────────────────────────────────────────
    elif merc == "Ambos Marcan":
        if "no" in sel or "al menos uno" in sel: return not (ga > 0 and gb > 0)
        return ga > 0 and gb > 0  # "Sí"

    # ── TARJETAS AMARILLAS ────────────────────────────────────────────────────
    # Las tarjetas reales no se guardan automáticamente en Supabase.
    # Se usan los campos opcionales "amarillas_reales" si existen.
    # Si no existen, retorna None → la apuesta queda como "pendiente manual".
    elif merc == "Tarjetas Amarillas":
        # Prioridad: parámetro directo > campo en Supabase
        if am_reales is None:
            am_reales = ap.get("amarillas_reales")
        if am_reales is None:
            return None  # sin dato, queda pendiente
        am_reales = int(am_reales)
        if "over 1.5" in sel: return am_reales >= 2
        if "over 2.5" in sel: return am_reales >= 3
        if "over 3.5" in sel: return am_reales >= 4
        if "over 4.5" in sel: return am_reales >= 5
        if "under 2.5" in sel: return am_reales <= 2
        if "under 3.5" in sel: return am_reales <= 3

    # ── CÓRNERS ───────────────────────────────────────────────────────────────
    elif merc == "Córners":
        # Prioridad: parámetro directo > campo en Supabase
        corners_reales = co_reales if co_reales is not None else ap.get("corners_reales")
        if corners_reales is None:
            return None  # sin dato, queda pendiente
        corners_reales = int(corners_reales)
        if "over 6.5"  in sel: return corners_reales >= 7
        if "over 7.5"  in sel: return corners_reales >= 8
        if "over 8.5"  in sel: return corners_reales >= 9
        if "over 9.5"  in sel: return corners_reales >= 10
        if "under 7.5" in sel: return corners_reales <= 7
        if "under 8.5" in sel: return corners_reales <= 8

    return None  # mercado no reconocido → pendiente


def cargar_historial_apuestas(url: str, key: str) -> list:
    try:
        r = requests.get(
            f"{url}/rest/v1/apuestas_historial",
            headers={**_headers(key), "Prefer": ""},
            params={"select": "*", "order": "fecha_partido.desc,guardada_en.desc"},
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
        return []
    except Exception:
        return []


def calcular_stats_apuestas(apuestas: list) -> dict:
    """
    Calcula estadísticas del historial de apuestas.
    Las apuestas con acierto=None (tarjetas/córners sin dato) van a pendientes.
    """
    evaluadas  = [a for a in apuestas if a.get("acierto") is not None]
    pendientes = [a for a in apuestas if a.get("acierto") is None]

    if not evaluadas:
        return {
            "total_evaluadas":  0,
            "total_pendientes": len(pendientes),
            "aciertos": 0, "fallos": 0, "accuracy": 0,
            "por_mercado": {}, "pendientes": pendientes, "evaluadas": [],
        }

    aciertos = sum(1 for a in evaluadas if a["acierto"] is True)
    fallos   = sum(1 for a in evaluadas if a["acierto"] is False)

    mercados = {}
    for a in evaluadas:
        m = a.get("mercado", "Otro")
        if m not in mercados:
            mercados[m] = {"total": 0, "aciertos": 0}
        mercados[m]["total"] += 1
        if a["acierto"] is True:
            mercados[m]["aciertos"] += 1

    for m in mercados:
        t = mercados[m]["total"]
        mercados[m]["accuracy"] = round(mercados[m]["aciertos"] / t * 100, 1)

    total_eval = aciertos + fallos  # solo las que tienen True/False
    return {
        "total_evaluadas":  total_eval,
        "total_pendientes": len(pendientes),
        "aciertos":         aciertos,
        "fallos":           fallos,
        "accuracy":         round(aciertos / total_eval * 100, 1) if total_eval else 0,
        "por_mercado":      mercados,
        "pendientes":       pendientes,
        "evaluadas":        evaluadas,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN AUXILIAR — actualizar tarjetas/córners manualmente
# ══════════════════════════════════════════════════════════════════════════════

def actualizar_datos_partido(url: str, key: str,
                              ea: str, eb: str,
                              amarillas_reales: int = None,
                              corners_reales: int = None) -> int:
    """
    Agrega datos reales de tarjetas y córners a las apuestas guardadas
    de un partido específico. Luego re-evalúa el acierto.

    Uso desde app.py (tab de admin o botón manual):
        actualizar_datos_partido(URL, KEY, "Croacia", "Panama",
                                  amarillas_reales=2, corners_reales=9)
    """
    actualizadas = 0
    try:
        r = requests.get(
            f"{url}/rest/v1/apuestas_historial",
            headers={**_headers(key), "Prefer": ""},
            params={"ea": f"eq.{ea}", "eb": f"eq.{eb}", "select": "*"},
            timeout=10
        )
        if r.status_code != 200:
            return 0

        for ap in r.json():
            patch_data = {}
            if amarillas_reales is not None:
                patch_data["amarillas_reales"] = amarillas_reales
            if corners_reales is not None:
                patch_data["corners_reales"] = corners_reales

            # Re-evaluar acierto con los nuevos datos
            ap_enriquecido = {**ap, **patch_data}
            ga = ap.get("goles_a")
            gb = ap.get("goles_b")
            if ga is not None and gb is not None:
                nuevo_acierto = _evaluar_acierto(ap_enriquecido, int(ga), int(gb))
                if nuevo_acierto is not None:
                    patch_data["acierto"] = nuevo_acierto

            if patch_data:
                requests.patch(
                    f"{url}/rest/v1/apuestas_historial",
                    headers={**_headers(key), "Prefer": ""},
                    params={"id": f"eq.{ap['id']}"},
                    json=patch_data,
                    timeout=8
                )
                actualizadas += 1
    except Exception:
        pass

    return actualizadas
