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
    """
    Guarda una predicción en Supabase.
    Usa ON CONFLICT DO NOTHING para no sobreescribir predicciones existentes.
    Retorna True si se guardó, False si ya existía o hubo error.
    """
    try:
        # Verificar si ya existe
        r_check = requests.get(
            f"{url}/rest/v1/predicciones",
            headers={**_headers(key), "Prefer": ""},
            params={"id": f"eq.{pred['id']}", "select": "id"},
            timeout=8
        )
        if r_check.status_code == 200 and r_check.json():
            return False  # ya existe

        # Insertar nueva predicción
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
    """
    Carga predicciones de Supabase.
    Si fecha es None, carga todas.
    Retorna dict {id: pred_dict}
    """
    try:
        params = {"select": "*", "order": "created_at.desc"}
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
    """Carga todas las predicciones del torneo."""
    return cargar_predicciones(url, key, fecha=None)


def simular_y_guardar_dia(url: str, key: str, partidos_hoy: list,
                           fn_simular, HORARIOS_PARTIDO: dict,
                           fn_toast=None) -> int:
    """
    Simula todos los partidos del día y guarda en Supabase.
    Solo guarda si no existe ya una predicción para ese partido.

    Args:
        url: Supabase project URL
        key: Supabase anon key
        partidos_hoy: lista de tuplas (ea, eb, grupo, sede, resultado, arbitro)
        fn_simular: función simular() de app.py
        HORARIOS_PARTIDO: dict de horarios
        fn_toast: función para mostrar notificaciones (opcional)

    Returns:
        Número de predicciones nuevas guardadas
    """
    from datetime import datetime as dt
    tz = timezone(timedelta(hours=-6))
    ahora = dt.now(tz)
    nuevas = 0

    for p in partidos_hoy:
        ea, eb, grupo, sede, _, arbitro = p
        pred_id = f"pred_{ea}_{eb}".replace(" ", "_")

        # Obtener horario del partido
        horario = (HORARIOS_PARTIDO.get((ea, eb)) or
                   HORARIOS_PARTIDO.get((eb, ea), ""))
        fecha_partido = horario[:10] if horario else ahora.strftime("%Y-%m-%d")

        try:
            # Simular con 500k iteraciones (rápido pero preciso)
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
    """
    Cruza predicciones guardadas con resultados reales.

    Returns:
        {
          "total": int,
          "acertados_ganador": int,
          "acertados_over25": int,
          "accuracy_ganador": float,
          "accuracy_over25": float,
          "detalle": list de dicts
        }
    """
    detalle = []
    acertados_g = 0
    acertados_o = 0

    # Mapa de resultados reales
    resultados = {}
    for p in partidos_con_resultado:
        if p[4] is not None:
            resultados[(p[0], p[1])] = p[4]

    for pred_id, pred in predicciones.items():
        # Saltar registros de control (no son partidos)
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

        if correcto_g:
            acertados_g += 1
        if correcto_o:
            acertados_o += 1

        detalle.append({
            "ea":           ea,
            "eb":           eb,
            "grupo":        pred.get("grupo", "?"),
            "guardada_en":  pred.get("guardada_en", "?"),
            "favorito":     pred.get("favorito", "?"),
            "prob_fav":     pred.get("prob_fav", 0),
            "resultado":    f"{ga}-{gb}",
            "ganador_real": ganador_real,
            "correcto":     correcto_g,
            "over25_real":  over25_real,
            "over25_pred":  over25_pred,
            "correcto_o25": correcto_o,
            "modelo":       pred.get("modelo", "?"),
        })

    total = len(detalle)
    return {
        "total":              total,
        "acertados_ganador":  acertados_g,
        "acertados_over25":   acertados_o,
        "accuracy_ganador":   round(acertados_g / total * 100, 1) if total else 0,
        "accuracy_over25":    round(acertados_o / total * 100, 1) if total else 0,
        "detalle":            sorted(detalle, key=lambda x: x["guardada_en"], reverse=True),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HISTORIAL DE APUESTAS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_apuestas_dia(url: str, key: str, partidos_hoy: list,
                          fn_simular, fn_analizar, HORARIOS_PARTIDO: dict) -> int:
    """
    Simula los partidos del día, analiza las apuestas sugeridas y las guarda.
    Solo guarda apuestas de nivel ALTA.
    Retorna número de apuestas guardadas.
    """
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
                    "id":           ap_id,
                    "ea":           ea,
                    "eb":           eb,
                    "grupo":        grupo,
                    "fecha_partido": fecha_partido,
                    "guardada_en":  ahora.strftime("%Y-%m-%d %H:%M"),
                    "mercado":      s["mercado"],
                    "seleccion":    s["seleccion"].replace("✅ ", ""),
                    "confianza":    round(s["confianza"], 1),
                    "donde":        s.get("donde", "Playdoit / Draftea"),
                    "resultado_real": None,
                    "goles_a":      None,
                    "goles_b":      None,
                    "acierto":      None,
                }

                try:
                    # Verificar si ya existe
                    r_check = requests.get(
                        f"{url}/rest/v1/apuestas_historial",
                        headers={**_headers(key), "Prefer": ""},
                        params={"id": f"eq.{ap_id}", "select": "id"},
                        timeout=8
                    )
                    if r_check.status_code == 200 and r_check.json():
                        continue  # ya existe

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
    Cuando un partido termina, actualiza las apuestas de ese partido
    para marcar si acertaron o no.
    Retorna número de apuestas actualizadas.
    """
    actualizadas = 0

    for p in partidos_con_resultado:
        ea, eb = p[0], p[1]
        resultado = p[4]
        if resultado is None:
            continue

        ga, gb = resultado
        goles_tot = ga + gb

        try:
            # Obtener apuestas de este partido sin resultado aún
            r = requests.get(
                f"{url}/rest/v1/apuestas_historial",
                headers={**_headers(key), "Prefer": ""},
                params={
                    "ea": f"eq.{ea}",
                    "eb": f"eq.{eb}",
                    "acierto": "is.null",
                    "select": "*"
                },
                timeout=10
            )
            if r.status_code != 200:
                continue

            apuestas = r.json()
            for ap in apuestas:
                acierto = _evaluar_acierto(ap, ga, gb)
                resultado_str = f"{ga}-{gb}"

                # Actualizar en Supabase
                requests.patch(
                    f"{url}/rest/v1/apuestas_historial",
                    headers={**_headers(key), "Prefer": ""},
                    params={"id": f"eq.{ap['id']}"},
                    json={
                        "acierto":       acierto,
                        "goles_a":       ga,
                        "goles_b":       gb,
                        "resultado_real": resultado_str,
                    },
                    timeout=8
                )
                actualizadas += 1

        except Exception:
            continue

    return actualizadas


def _evaluar_acierto(ap: dict, ga: int, gb: int) -> bool:
    """Evalúa si una apuesta acertó dado el resultado real."""
    sel = ap.get("seleccion", "").lower()
    merc = ap.get("mercado", "")
    goles_tot = ga + gb
    ea = ap.get("ea", "")
    eb = ap.get("eb", "")

    if merc == "Total Goles":
        if "over 0.5" in sel:   return goles_tot > 0
        if "over 1.5" in sel:   return goles_tot > 1
        if "over 2.5" in sel:   return goles_tot > 2
        if "over 3.5" in sel:   return goles_tot > 3
        if "under 1.5" in sel:  return goles_tot <= 1
        if "under 2.5" in sel:  return goles_tot <= 2
        if "under 3.5" in sel:  return goles_tot <= 3
    elif merc == "Resultado (1X2)":
        if f"gana {ea.lower()}" in sel: return ga > gb
        if f"gana {eb.lower()}" in sel: return gb > ga
        if "empate" in sel:             return ga == gb
    elif merc == "Doble Oportunidad":
        if ea.lower() in sel and "empate" in sel.replace(ea.lower(), ""): 
            return ga >= gb  # 1X
        if eb.lower() in sel and "empate" in sel.replace(eb.lower(), ""):
            return gb >= ga  # X2
        if ea.lower() in sel and eb.lower() in sel:
            return ga != gb  # 12
    elif merc == "Ambos Marcan":
        return ga > 0 and gb > 0
    return False


def cargar_historial_apuestas(url: str, key: str) -> list:
    """Carga todo el historial de apuestas ordenado por fecha."""
    try:
        r = requests.get(
            f"{url}/rest/v1/apuestas_historial",
            headers={**_headers(key), "Prefer": ""},
            params={"select": "*", "order": "created_at.desc"},
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
        return []
    except Exception:
        return []


def calcular_stats_apuestas(apuestas: list) -> dict:
    """Calcula estadísticas del historial de apuestas."""
    evaluadas = [a for a in apuestas if a.get("acierto") is not None]
    pendientes = [a for a in apuestas if a.get("acierto") is None]

    if not evaluadas:
        return {
            "total_evaluadas": 0, "total_pendientes": len(pendientes),
            "aciertos": 0, "fallos": 0, "accuracy": 0,
            "por_mercado": {}, "pendientes": pendientes
        }

    aciertos = sum(1 for a in evaluadas if a["acierto"])
    fallos   = len(evaluadas) - aciertos

    # Accuracy por mercado
    mercados = {}
    for a in evaluadas:
        m = a.get("mercado", "Otro")
        if m not in mercados:
            mercados[m] = {"total": 0, "aciertos": 0}
        mercados[m]["total"] += 1
        if a["acierto"]:
            mercados[m]["aciertos"] += 1

    for m in mercados:
        t = mercados[m]["total"]
        mercados[m]["accuracy"] = round(mercados[m]["aciertos"] / t * 100, 1)

    return {
        "total_evaluadas":  len(evaluadas),
        "total_pendientes": len(pendientes),
        "aciertos":         aciertos,
        "fallos":           fallos,
        "accuracy":         round(aciertos / len(evaluadas) * 100, 1),
        "por_mercado":      mercados,
        "pendientes":       pendientes,
        "evaluadas":        evaluadas,
    }
