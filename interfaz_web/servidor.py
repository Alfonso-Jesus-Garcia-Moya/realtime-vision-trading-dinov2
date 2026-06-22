"""
Servidor Web — Terminal de Trading con Detección de Patrones en Tiempo Real.

Ejecutar:
    python -m interfaz_web.servidor
    python -m interfaz_web.servidor --puerto 5000 --intervalo 10

Abre http://localhost:5000 para ver la interfaz.
"""

import os
import sys
import io
import json
import time
import base64
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

from flask import Flask, render_template, jsonify, request

# ── Asegurar que la raíz del proyecto esté en sys.path ──────────
DIRECTORIO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DIRECTORIO_RAIZ not in sys.path:
    sys.path.insert(0, DIRECTORIO_RAIZ)

import torch
import numpy as np
from PIL import Image

from captura_datos.configuracion import Configuracion
from captura_datos.capturador_pantalla import CapturadorPantalla
from captura_datos.tuberia_tensores import TuberiaTensores

# ── Logger ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-22s │ %(message)s",
    datefmt="%H:%M:%S",
)
registrador = logging.getLogger("interfaz_web")


# ══════════════════════════════════════════════════════════════════
#  Estado Global Thread-Safe
# ══════════════════════════════════════════════════════════════════

class EstadoGlobal:
    """Estado compartido entre el hilo de captura y Flask."""

    def __init__(self, max_historial: int = 200, max_logs: int = 500):
        self.candado = threading.Lock()

        # ── Imagen actual ─────────────────────────────────────────
        self.imagen_actual_b64: str = ""
        self.imagen_bytes_png: bytes = b""

        # ── Predicción GRÁFICO COMPLETO ───────────────────────────
        self.patron_actual: str = "Esperando..."
        self.confianza_actual: float = 0.0
        self.es_valido: bool = False
        self.timestamp_actual: str = ""
        self.tiempo_inferencia_ms: float = 0.0

        # ── Predicción MITAD DERECHA ──────────────────────────────
        self.patron_mitad: str = "Esperando..."
        self.confianza_mitad: float = 0.0
        self.es_valido_mitad: bool = False
        self.tiempo_inferencia_mitad_ms: float = 0.0

        # ── Historial ─────────────────────────────────────────────
        self.historial: deque = deque(maxlen=max_historial)

        # ── Logs del sistema ──────────────────────────────────────
        self.logs: deque = deque(maxlen=max_logs)

        # ── Métricas ─────────────────────────────────────────────
        self.total_capturas: int = 0
        self.total_patrones_validos: int = 0
        self.total_errores: int = 0
        self.dispositivo: str = "cpu"
        self.modelo_cargado: bool = False
        self.pesos_entrenados: bool = False
        self.ejecutando: bool = False
        self.fps_promedio: float = 0.0
        self.url_actual: str = ""

        # ── Rango e intervalo actuales ────────────────────────────
        self.rango_actual: str = "ytd"
        self.intervalo_actual: str = "1 min"

        # ── Señales de control ────────────────────────────────────
        self.solicitud_cambio_url: Optional[str] = None
        self.solicitud_cambio_rango: Optional[str] = None
        self.solicitud_cambio_intervalo: Optional[str] = None
        self.solicitud_detener: bool = False
        self.solicitud_reiniciar: bool = False

        # ── Estado de entrenamiento ───────────────────────────────
        self.entrenando: bool = False
        self.progreso_entrenamiento: float = 0.0
        self.mensaje_entrenamiento: str = ""

    def agregar_log(self, nivel: str, modulo: str, mensaje: str):
        """Agrega un log al buffer para mostrar en la UI."""
        with self.candado:
            self.logs.appendleft({
                "hora": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "nivel": nivel,
                "modulo": modulo,
                "mensaje": mensaje,
            })

    def actualizar_prediccion(
        self,
        imagen_b64: str,
        imagen_bytes: bytes,
        patron: str,
        confianza: float,
        es_valido: bool,
        tiempo_ms: float,
    ):
        with self.candado:
            ahora = datetime.now().strftime("%H:%M:%S")
            self.imagen_actual_b64 = imagen_b64
            self.imagen_bytes_png = imagen_bytes
            self.patron_actual = patron
            self.confianza_actual = confianza
            self.es_valido = es_valido
            self.timestamp_actual = ahora
            self.tiempo_inferencia_ms = tiempo_ms
            self.total_capturas += 1
            if es_valido:
                self.total_patrones_validos += 1

            self.historial.appendleft({
                "hora": ahora,
                "patron": patron.replace("_", " "),
                "confianza": round(confianza * 100, 1),
                "valido": es_valido,
                "inferencia_ms": round(tiempo_ms, 1),
            })

    def obtener_estado_completo(self) -> Dict[str, Any]:
        with self.candado:
            return {
                "imagen_b64": self.imagen_actual_b64,
                "patron": self.patron_actual.replace("_", " "),
                "confianza": round(self.confianza_actual * 100, 1),
                "es_valido": self.es_valido,
                "timestamp": self.timestamp_actual,
                "inferencia_ms": round(self.tiempo_inferencia_ms, 1),
                "total_capturas": self.total_capturas,
                "total_patrones": self.total_patrones_validos,
                "total_errores": self.total_errores,
                "dispositivo": self.dispositivo,
                "modelo_cargado": self.modelo_cargado,
                "pesos_entrenados": self.pesos_entrenados,
                "ejecutando": self.ejecutando,
                "fps": round(self.fps_promedio, 2),
                "url_actual": self.url_actual,
                "patron_mitad": self.patron_mitad.replace("_", " "),
                "confianza_mitad": round(self.confianza_mitad * 100, 1),
                "es_valido_mitad": self.es_valido_mitad,
                "inferencia_mitad_ms": round(self.tiempo_inferencia_mitad_ms, 1),
                "rango_actual": self.rango_actual,
                "intervalo_actual": self.intervalo_actual,
                "entrenando": self.entrenando,
                "progreso_entrenamiento": round(self.progreso_entrenamiento, 1),
                "mensaje_entrenamiento": self.mensaje_entrenamiento,
                "historial": list(self.historial)[:50],
                "logs": list(self.logs)[:100],
            }


estado = EstadoGlobal()


# ══════════════════════════════════════════════════════════════════
#  Handler de Logging que alimenta la UI
# ══════════════════════════════════════════════════════════════════

class ManejadorLogsUI(logging.Handler):
    """Captura logs de Python y los envía al estado global para la UI."""

    def emit(self, record):
        try:
            estado.agregar_log(
                nivel=record.levelname,
                modulo=record.name,
                mensaje=self.format(record),
            )
        except Exception:
            pass


# Agregar handler a la raíz del logging
manejador_ui = ManejadorLogsUI()
manejador_ui.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(manejador_ui)


# ══════════════════════════════════════════════════════════════════
#  Utilidades
# ══════════════════════════════════════════════════════════════════

def bytes_png_a_base64(datos_png: bytes) -> str:
    """Convierte bytes PNG a string base64 para incrustar en HTML."""
    return base64.b64encode(datos_png).decode("utf-8")


def tensor_a_base64(tensor_img: torch.Tensor) -> str:
    """Convierte tensor [C,H,W] float32 [0,1] a base64 PNG."""
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.squeeze(0)
    arreglo = (tensor_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arreglo)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ══════════════════════════════════════════════════════════════════
#  Hilo de Captura + Inferencia
# ══════════════════════════════════════════════════════════════════

def hilo_captura_inferencia(url_inicial: str, intervalo_seg: float = 10.0):
    """
    Hilo daemon que:
    1. Abre Chrome headless en la URL de Yahoo Finance.
    2. Cada N segundos captura el chart como bytes PNG.
    3. Convierte a tensor en memoria.
    4. Ejecuta inferencia DINOv2 para detectar patrón técnico.
    5. Actualiza el estado global para que Flask lo sirva.
    """
    global estado

    registrador.info("═" * 55)
    registrador.info("  INICIANDO MOTOR DE CAPTURA + INFERENCIA")
    registrador.info("═" * 55)

    # ── 1. Configurar capturador ──────────────────────────────────
    config_captura = Configuracion(url_grafico=url_inicial)
    capturador = CapturadorPantalla(config_captura)
    tuberia = TuberiaTensores(config_captura)

    estado.url_actual = url_inicial

    # ── 2. Inicializar detector de patrones ───────────────────────
    procesador = None
    try:
        from detector_patrones.configuracion_vision import ConfiguracionVision
        from detector_patrones.procesador_vision import ProcesadorVision

        config_vision = ConfiguracionVision()
        procesador = ProcesadorVision(config_vision)
        procesador.inicializar()

        estado.modelo_cargado = True
        estado.dispositivo = str(procesador.dispositivo)
        estado.pesos_entrenados = os.path.exists(config_vision.ruta_pesos_cabeza)

        registrador.info("✔ Detector de patrones DINOv2 listo en: %s", procesador.dispositivo)

    except Exception as error:
        registrador.error("✘ Error cargando detector: %s", error)
        registrador.warning("Continuando sin detección de patrones.")
        estado.modelo_cargado = False

    # ── 3. Iniciar navegador ──────────────────────────────────────
    try:
        capturador.iniciar()
        registrador.info("✔ Navegador Chrome iniciado correctamente")
    except Exception as error:
        registrador.error("✘ Error iniciando navegador: %s", error)
        estado.ejecutando = False
        estado.agregar_log("ERROR", "sistema", f"No se pudo iniciar Chrome: {error}")
        return

    estado.ejecutando = True
    tiempos_ciclo = deque(maxlen=20)

    # ── 4. Bucle principal ────────────────────────────────────────
    try:
        while estado.ejecutando and not estado.solicitud_detener:

            # ── Verificar si hay cambio de URL solicitado ─────────
            if estado.solicitud_cambio_url is not None:
                nueva_url = estado.solicitud_cambio_url
                estado.solicitud_cambio_url = None

                registrador.info("🔄 Cambiando URL a: %s", nueva_url)
                try:
                    capturador.detener()
                    config_captura = Configuracion(url_grafico=nueva_url)
                    capturador = CapturadorPantalla(config_captura)
                    capturador.iniciar()
                    estado.url_actual = nueva_url
                    registrador.info("✔ URL cambiada exitosamente")
                except Exception as error:
                    registrador.error("✘ Error cambiando URL: %s", error)
                    estado.total_errores += 1
                    continue

            # ── Verificar si hay cambio de RANGO solicitado ───────
            if estado.solicitud_cambio_rango is not None:
                nuevo_rango = estado.solicitud_cambio_rango
                estado.solicitud_cambio_rango = None
                registrador.info("🔄 Cambiando rango a: %s", nuevo_rango)
                exito = capturador.seleccionar_rango(nuevo_rango)
                if exito:
                    estado.rango_actual = nuevo_rango
                    registrador.info("✔ Rango cambiado a '%s'", nuevo_rango)
                else:
                    registrador.warning("✘ No se pudo cambiar rango a '%s'", nuevo_rango)

            # ── Verificar si hay cambio de INTERVALO solicitado ───
            if estado.solicitud_cambio_intervalo is not None:
                nuevo_intervalo = estado.solicitud_cambio_intervalo
                estado.solicitud_cambio_intervalo = None
                registrador.info("🔄 Cambiando intervalo a: %s", nuevo_intervalo)
                exito = capturador.seleccionar_intervalo(nuevo_intervalo)
                if exito:
                    estado.intervalo_actual = nuevo_intervalo
                    registrador.info("✔ Intervalo cambiado a '%s'", nuevo_intervalo)
                else:
                    registrador.warning("✘ No se pudo cambiar intervalo a '%s'", nuevo_intervalo)

            inicio_ciclo = time.time()

            try:
                # ── PASO 1: Capturar screenshot como bytes ────────
                datos_png = capturador.capturar_grafico_bytes()

                if datos_png is None:
                    registrador.warning("Captura vacía, reintentando en 3s...")
                    estado.total_errores += 1
                    time.sleep(3)
                    continue

                # ── PASO 2: Convertir a tensor en memoria ─────────
                tensor = tuberia.procesar_captura(datos_png)
                if tensor is None:
                    estado.total_errores += 1
                    continue

                # ── PASO 3: Imagen para la UI (base64) ────────────
                imagen_b64 = bytes_png_a_base64(datos_png)

                # ── PASO 4A: Inferencia GRÁFICO COMPLETO ──────────
                nombre_patron = "Sin modelo cargado"
                confianza = 0.0
                es_valido = False
                tiempo_inferencia_ms = 0.0

                # ── PASO 4B: Inferencia MITAD DERECHA ─────────────
                nombre_patron_mitad = "Sin modelo cargado"
                confianza_mitad = 0.0
                es_valido_mitad = False
                tiempo_inferencia_mitad_ms = 0.0

                if procesador is not None and procesador.esta_inicializado:
                    # Análisis 1: Gráfico completo
                    resultado_completo = procesador.detectar_patron(tensor)
                    if resultado_completo is not None:
                        nombre_patron = resultado_completo.nombre_patron
                        confianza = resultado_completo.confianza
                        es_valido = resultado_completo.es_patron_valido
                        tiempo_inferencia_ms = resultado_completo.tiempo_inferencia_ms
                    else:
                        nombre_patron = "Error en inferencia"
                        estado.total_errores += 1

                    # Análisis 2: Mitad derecha del gráfico
                    tensor_mitad = tuberia.recortar_mitad_derecha(datos_png)
                    if tensor_mitad is not None:
                        resultado_mitad = procesador.detectar_patron(tensor_mitad)
                        if resultado_mitad is not None:
                            nombre_patron_mitad = resultado_mitad.nombre_patron
                            confianza_mitad = resultado_mitad.confianza
                            es_valido_mitad = resultado_mitad.es_patron_valido
                            tiempo_inferencia_mitad_ms = resultado_mitad.tiempo_inferencia_ms

                    registrador.info(
                        "  🔎 Completo: %s (%.1f%%) │ Mitad der: %s (%.1f%%)",
                        nombre_patron.replace("_", " "), confianza * 100,
                        nombre_patron_mitad.replace("_", " "), confianza_mitad * 100,
                    )

                # ── PASO 5: Actualizar estado global ──────────────
                estado.actualizar_prediccion(
                    imagen_b64=imagen_b64,
                    imagen_bytes=datos_png,
                    patron=nombre_patron,
                    confianza=confianza,
                    es_valido=es_valido,
                    tiempo_ms=tiempo_inferencia_ms,
                )
                # Actualizar predicción mitad derecha
                with estado.candado:
                    estado.patron_mitad = nombre_patron_mitad
                    estado.confianza_mitad = confianza_mitad
                    estado.es_valido_mitad = es_valido_mitad
                    estado.tiempo_inferencia_mitad_ms = tiempo_inferencia_mitad_ms

                # ── FPS ───────────────────────────────────────────
                duracion = time.time() - inicio_ciclo
                tiempos_ciclo.append(duracion)
                if tiempos_ciclo:
                    estado.fps_promedio = 1.0 / (sum(tiempos_ciclo) / len(tiempos_ciclo))

                registrador.info(
                    "📸 #%d │ %s │ %.1f%% │ %.0fms │ buffer: %d",
                    estado.total_capturas,
                    nombre_patron.replace("_", " "),
                    confianza * 100,
                    tiempo_inferencia_ms,
                    tuberia.tamano_buffer,
                )

            except Exception as error:
                estado.total_errores += 1
                registrador.error("Error en ciclo de captura: %s", error)

            # ── Esperar para siguiente ciclo ──────────────────────
            tiempo_usado = time.time() - inicio_ciclo
            espera = max(0, intervalo_seg - tiempo_usado)
            # Esperar en pasos cortos para responder rápido a señales
            pasos = int(espera / 0.5)
            for _ in range(pasos):
                if not estado.ejecutando or estado.solicitud_detener:
                    break
                if (estado.solicitud_cambio_url is not None
                        or estado.solicitud_cambio_rango is not None
                        or estado.solicitud_cambio_intervalo is not None):
                    break
                time.sleep(0.5)
            residuo = espera - (pasos * 0.5)
            if residuo > 0 and estado.ejecutando:
                time.sleep(residuo)

    except Exception as error:
        registrador.error("Error fatal en hilo de captura: %s", error)

    finally:
        capturador.detener()
        if procesador is not None:
            procesador.liberar_recursos()
        estado.ejecutando = False
        registrador.info("Motor de captura finalizado. ✔")


# ══════════════════════════════════════════════════════════════════
#  Aplicación Flask
# ══════════════════════════════════════════════════════════════════

aplicacion = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "plantillas"),
    static_folder=os.path.join(os.path.dirname(__file__), "estaticos"),
)


@aplicacion.route("/")
def pagina_principal():
    """Página principal — Terminal de Trading."""
    return render_template("terminal.html")


@aplicacion.route("/api/estado")
def api_estado():
    """Estado completo del sistema en JSON."""
    return jsonify(estado.obtener_estado_completo())


@aplicacion.route("/api/cambiar-url", methods=["POST"])
def api_cambiar_url():
    """Cambia la URL del chart de Yahoo Finance en tiempo real."""
    datos = request.get_json(silent=True) or {}
    nueva_url = datos.get("url", "").strip()

    if not nueva_url:
        return jsonify({"error": "URL vacía"}), 400

    if "finance.yahoo.com" not in nueva_url and "yahoo" not in nueva_url.lower():
        return jsonify({"error": "URL debe ser de Yahoo Finance"}), 400

    estado.solicitud_cambio_url = nueva_url
    registrador.info("Solicitud de cambio de URL recibida: %s", nueva_url)
    return jsonify({"mensaje": f"Cambiando a: {nueva_url}", "url": nueva_url})


@aplicacion.route("/api/cambiar-rango", methods=["POST"])
def api_cambiar_rango():
    """Cambia el rango temporal del chart (3m, 6m, 1y, etc.)."""
    datos = request.get_json(silent=True) or {}
    rango = datos.get("rango", "").strip().lower()
    validos = ["1d", "5d", "1m", "3m", "6m", "ytd", "1y", "5y", "max"]
    if rango not in validos:
        return jsonify({"error": f"Rango inválido. Válidos: {validos}"}), 400
    estado.solicitud_cambio_rango = rango
    registrador.info("Solicitud de cambio de rango: %s", rango)
    return jsonify({"mensaje": f"Cambiando rango a: {rango}", "rango": rango})


@aplicacion.route("/api/cambiar-intervalo", methods=["POST"])
def api_cambiar_intervalo():
    """Cambia el intervalo del chart (1 min, 5 min, etc.)."""
    datos = request.get_json(silent=True) or {}
    intervalo = datos.get("intervalo", "").strip()
    if not intervalo:
        return jsonify({"error": "Intervalo vacío"}), 400
    estado.solicitud_cambio_intervalo = intervalo
    registrador.info("Solicitud de cambio de intervalo: %s", intervalo)
    return jsonify({"mensaje": f"Cambiando intervalo a: {intervalo}", "intervalo": intervalo})


@aplicacion.route("/api/reiniciar", methods=["POST"])
def api_reiniciar():
    """Reinicia TODO: detiene captura, reentrena modelo en GPU, reinicia."""
    if estado.entrenando:
        return jsonify({"error": "Ya hay un entrenamiento en curso"}), 409

    registrador.info("🔁 SOLICITUD DE REINICIO TOTAL recibida")

    def hilo_reinicio():
        """Ejecuta el reinicio + reentrenamiento en un hilo separado."""
        import subprocess

        estado.entrenando = True
        estado.mensaje_entrenamiento = "Deteniendo captura..."
        estado.progreso_entrenamiento = 5.0
        registrador.info("🔁 Paso 1/4: Deteniendo motor de captura...")

        # 1. Detener captura actual
        estado.solicitud_detener = True
        time.sleep(3)

        # 2. Limpiar VRAM antes de entrenar
        estado.mensaje_entrenamiento = "Limpiando VRAM para entrenamiento..."
        estado.progreso_entrenamiento = 15.0
        registrador.info("🔁 Paso 2/4: Limpiando VRAM...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 3. Reentrenar modelo en GPU
        estado.mensaje_entrenamiento = "Reentrenando modelo DINOv2 en GPU..."
        estado.progreso_entrenamiento = 25.0
        registrador.info("🔁 Paso 3/4: Reentrenando cabeza clasificadora en GPU...")

        try:
            resultado = subprocess.run(
                [sys.executable, "-m", "detector_patrones.entrenador_patrones",
                 "--forzar-gpu"],
                capture_output=True, text=True, timeout=600,
                cwd=DIRECTORIO_RAIZ,
            )
            if resultado.returncode == 0:
                estado.mensaje_entrenamiento = "✔ Entrenamiento completado con éxito"
                estado.progreso_entrenamiento = 80.0
                registrador.info("✔ Reentrenamiento completado exitosamente en GPU")
                for linea in resultado.stdout.strip().split("\n")[-10:]:
                    registrador.info("  [TRAIN] %s", linea)
            else:
                estado.mensaje_entrenamiento = f"⚠ Entrenamiento falló: {resultado.stderr[-200:]}"
                estado.progreso_entrenamiento = 80.0
                registrador.error("✘ Error en reentrenamiento: %s", resultado.stderr[-500:])
        except subprocess.TimeoutExpired:
            estado.mensaje_entrenamiento = "⚠ Entrenamiento excedió tiempo límite (10 min)"
            registrador.error("✘ Timeout en reentrenamiento")
        except Exception as error:
            estado.mensaje_entrenamiento = f"⚠ Error: {error}"
            registrador.error("✘ Error en reentrenamiento: %s", error)

        # 4. Reiniciar motor de captura con modelo actualizado
        estado.mensaje_entrenamiento = "Reiniciando motor de captura..."
        estado.progreso_entrenamiento = 90.0
        registrador.info("🔁 Paso 4/4: Reiniciando motor de captura con modelo actualizado...")

        estado.solicitud_detener = False
        estado.solicitud_reiniciar = False
        estado.total_capturas = 0
        estado.total_patrones_validos = 0
        estado.total_errores = 0

        url_reinicio = estado.url_actual or URL_POR_DEFECTO
        hilo_nuevo = threading.Thread(
            target=hilo_captura_inferencia,
            args=(url_reinicio, 10.0),
            daemon=True,
            name="motor-captura-reinicio",
        )
        hilo_nuevo.start()

        estado.progreso_entrenamiento = 100.0
        estado.mensaje_entrenamiento = "✔ Sistema reiniciado completamente"
        registrador.info("🔁 ✔ REINICIO TOTAL COMPLETADO — sistema activo con modelo actualizado")

        time.sleep(3)
        estado.entrenando = False
        estado.progreso_entrenamiento = 0.0
        estado.mensaje_entrenamiento = ""

    threading.Thread(target=hilo_reinicio, daemon=True, name="hilo-reinicio").start()
    return jsonify({"mensaje": "Reinicio total iniciado — reentrenando modelo en GPU..."})


@aplicacion.route("/api/detener", methods=["POST"])
def api_detener():
    """Detiene el sistema de captura."""
    estado.solicitud_detener = True
    return jsonify({"mensaje": "Deteniendo sistema..."})


@aplicacion.route("/api/logs")
def api_logs():
    """Últimos logs del sistema."""
    with estado.candado:
        return jsonify(list(estado.logs)[:200])


# ══════════════════════════════════════════════════════════════════
#  Punto de Entrada
# ══════════════════════════════════════════════════════════════════

URL_POR_DEFECTO = "https://finance.yahoo.com/chart/%5EGSPC"


def iniciar_servidor(puerto: int = 5000, intervalo: float = 10.0, url: str = URL_POR_DEFECTO):
    """Inicia el servidor web y el motor de captura."""
    registrador.info("╔═══════════════════════════════════════════════╗")
    registrador.info("║  TERMINAL DE TRADING — DETECCIÓN DE PATRONES  ║")
    registrador.info("╠═══════════════════════════════════════════════╣")
    registrador.info("║  http://localhost:%d                         ║", puerto)
    registrador.info("╚═══════════════════════════════════════════════╝")

    # Lanzar hilo de captura + inferencia
    hilo = threading.Thread(
        target=hilo_captura_inferencia,
        args=(url, intervalo),
        daemon=True,
        name="motor-captura",
    )
    hilo.start()

    # Iniciar Flask (bloquea el hilo principal)
    aplicacion.run(host="0.0.0.0", port=puerto, debug=False, use_reloader=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Terminal de Trading — Detección de Patrones en Tiempo Real"
    )
    parser.add_argument("--puerto", type=int, default=5000, help="Puerto del servidor (default: 5000)")
    parser.add_argument("--intervalo", type=float, default=10.0, help="Segundos entre capturas (default: 10)")
    parser.add_argument("--url", type=str, default=URL_POR_DEFECTO, help="URL del chart de Yahoo Finance")

    args = parser.parse_args()
    iniciar_servidor(args.puerto, args.intervalo, args.url)
