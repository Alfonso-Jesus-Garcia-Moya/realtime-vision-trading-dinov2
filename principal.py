"""
Orquestador Principal — Sistema de Captura y Detección en Tiempo Real.

Coordina tres subsistemas:
  1. Capturador de pantalla → screenshots del gráfico Yahoo Finance.
  2. Tubería de tensores    → conversión a tensores [3, 224, 224] en memoria.
  3. Procesador de visión   → inferencia DINOv2 para detectar patrones técnicos.

Flujo continuo cada N segundos:
  Screenshot (bytes) → Tensor (GPU/CPU) → DINOv2 → Patrón detectado + confianza

Uso:
    python principal.py
    python principal.py --intervalo 10 --capturas 50 --con-vision
    python principal.py --sin-vision --visible --debug
"""

import sys
import time
import signal
import logging
import argparse
from datetime import datetime
from typing import Optional

from captura_datos.configuracion import Configuracion
from captura_datos.capturador_pantalla import CapturadorPantalla
from captura_datos.tuberia_tensores import TuberiaTensores


# ══════════════════════════════════════════════════════════════════
#  Configuración del logging
# ══════════════════════════════════════════════════════════════════

def configurar_logging(nivel: int = logging.INFO) -> None:
    """Configura el sistema de logging con formato legible."""
    formato = (
        "%(asctime)s │ %(levelname)-8s │ %(name)-22s │ %(message)s"
    )
    logging.basicConfig(
        level=nivel,
        format=formato,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ══════════════════════════════════════════════════════════════════
#  Señal de interrupción (Ctrl+C)
# ══════════════════════════════════════════════════════════════════

ejecutando = True


def manejador_senal(sig, marco):
    """Maneja Ctrl+C para detener el bucle limpiamente."""
    global ejecutando
    ejecutando = False
    print("\n⚠  Señal de interrupción recibida. Deteniendo…")


signal.signal(signal.SIGINT, manejador_senal)


# ══════════════════════════════════════════════════════════════════
#  Argumentos de línea de comandos
# ══════════════════════════════════════════════════════════════════

def parsear_argumentos() -> argparse.Namespace:
    """Define y parsea los argumentos CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Captura de gráfico Yahoo Finance → Tensores en memoria "
            "→ Detección de patrones con DINOv2"
        ),
    )
    parser.add_argument(
        "--intervalo",
        type=int,
        default=10,
        help="Segundos entre capturas (default: 10)",
    )
    parser.add_argument(
        "--capturas",
        type=int,
        default=0,
        help="Número máximo de capturas (0 = infinito, default: 0)",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Ejecutar el navegador en modo visible (no headless)",
    )
    parser.add_argument(
        "--tamano",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("ALTO", "ANCHO"),
        help="Tamaño del tensor (default: 224 224)",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=100,
        help="Capacidad del buffer circular de tensores (default: 100)",
    )
    parser.add_argument(
        "--dispositivo",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo para la tubería de tensores (default: cpu)",
    )

    # ── Argumentos del detector de patrones ───────────────────────
    grupo_vision = parser.add_argument_group("Detector de Patrones (Visión)")
    grupo_vision.add_argument(
        "--con-vision",
        action="store_true",
        default=True,
        help="Activar detección de patrones con DINOv2 (default: activado)",
    )
    grupo_vision.add_argument(
        "--sin-vision",
        action="store_true",
        help="Desactivar detección de patrones (solo captura + tensores)",
    )
    grupo_vision.add_argument(
        "--umbral-confianza",
        type=float,
        default=0.70,
        help="Umbral mínimo de confianza para aceptar un patrón (default: 0.70)",
    )
    grupo_vision.add_argument(
        "--fraccion-vram",
        type=float,
        default=0.40,
        help="Fracción máxima de VRAM a usar (default: 0.40 = 40%%)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activar logging en nivel DEBUG",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
#  Inicialización del procesador de visión (carga perezosa)
# ══════════════════════════════════════════════════════════════════

def inicializar_procesador_vision(
    umbral_confianza: float = 0.70,
    fraccion_vram: float = 0.40,
):
    """
    Carga perezosa del procesador de visión.
    Importa los módulos de detector_patrones solo si se necesitan,
    evitando la descarga del modelo DINOv2 si --sin-vision está activo.

    Returns:
        Instancia de ProcesadorVision inicializada, o None si falla.
    """
    registrador = logging.getLogger("principal")

    try:
        from detector_patrones.configuracion_vision import ConfiguracionVision
        from detector_patrones.procesador_vision import ProcesadorVision

        config_vision = ConfiguracionVision(
            umbral_confianza_minima=umbral_confianza,
            fraccion_vram_maxima=fraccion_vram,
        )

        registrador.info("─" * 55)
        registrador.info("  INICIALIZANDO DETECTOR DE PATRONES (DINOv2)")
        registrador.info("─" * 55)
        print("\n" + config_vision.resumen() + "\n")

        procesador = ProcesadorVision(config_vision)
        procesador.inicializar()

        return procesador

    except ImportError as error:
        registrador.error(
            "No se pudo importar detector_patrones: %s. "
            "Verificar que el módulo existe y las dependencias están instaladas.",
            error,
        )
        return None

    except Exception as error:
        registrador.error(
            "Error al inicializar procesador de visión: %s",
            error,
            exc_info=True,
        )
        return None


# ══════════════════════════════════════════════════════════════════
#  Bucle principal de captura + detección
# ══════════════════════════════════════════════════════════════════

def ejecutar_bucle_captura(
    config: Configuracion,
    capturador: CapturadorPantalla,
    tuberia: TuberiaTensores,
    procesador_vision=None,
    max_capturas: int = 0,
) -> None:
    """
    Bucle que cada `config.intervalo_captura_seg` segundos:
      1. Toma un screenshot del gráfico (bytes en memoria).
      2. Lo pasa por la tubería para convertirlo a tensor.
      3. (Opcional) Ejecuta inferencia DINOv2 para detectar patrón.
      4. Muestra estadísticas y resultado en consola.

    Se detiene con Ctrl+C o al alcanzar max_capturas (si > 0).
    """
    registrador = logging.getLogger("bucle_principal")
    numero_ciclo = 0
    vision_activa = procesador_vision is not None

    registrador.info("═" * 60)
    registrador.info("  INICIANDO BUCLE DE CAPTURA EN TIEMPO REAL")
    registrador.info("  Intervalo: %d seg | Máx capturas: %s | Visión: %s",
                     config.intervalo_captura_seg,
                     max_capturas if max_capturas > 0 else "∞",
                     "DINOv2 ACTIVO ✔" if vision_activa else "DESACTIVADO ✘")
    registrador.info("═" * 60)

    while ejecutando:
        numero_ciclo += 1
        inicio_ciclo = time.time()

        # ── Verificar límite de capturas ──────────────────────────
        if max_capturas > 0 and numero_ciclo > max_capturas:
            registrador.info("Límite de %d capturas alcanzado.", max_capturas)
            break

        registrador.info("─── Ciclo #%d ───", numero_ciclo)

        # ══════════════════════════════════════════════════════════
        #  PASO 1: Capturar screenshot como bytes
        # ══════════════════════════════════════════════════════════
        datos_png = capturador.capturar_grafico_bytes()

        if datos_png is None:
            registrador.warning("Captura fallida en ciclo #%d. Reintentando…",
                                numero_ciclo)
            _esperar_siguiente_ciclo(config.intervalo_captura_seg, inicio_ciclo)
            continue

        registrador.info(
            "  📸 Screenshot: %d bytes (%.1f KB)",
            len(datos_png),
            len(datos_png) / 1024,
        )

        # ══════════════════════════════════════════════════════════
        #  PASO 2: Convertir a tensor en memoria
        # ══════════════════════════════════════════════════════════
        tensor = tuberia.procesar_captura(datos_png)

        if tensor is not None:
            registrador.info(
                "  🧮 Tensor:     forma=%s  dtype=%s  dispositivo=%s",
                list(tensor.shape),
                tensor.dtype,
                tensor.device,
            )
            registrador.info(
                "  📊 Buffer:     %d/%d tensores almacenados",
                tuberia.tamano_buffer,
                config.capacidad_buffer,
            )
            registrador.info(
                "  📈 Valores:    min=%.4f  max=%.4f  media=%.4f",
                tensor.min().item(),
                tensor.max().item(),
                tensor.mean().item(),
            )

            # ══════════════════════════════════════════════════════
            #  PASO 3: Detectar patrón técnico con DINOv2
            # ══════════════════════════════════════════════════════
            if vision_activa:
                resultado = procesador_vision.detectar_patron(tensor)

                if resultado is not None:
                    # Resultado formateado para consola
                    registrador.info(
                        "  %s", resultado.formato_consola()
                    )

                    # Log adicional según validez del patrón
                    if resultado.es_patron_valido:
                        registrador.info(
                            "  ✅ Patrón válido detectado con %.1f%% de confianza.",
                            resultado.confianza * 100,
                        )
                    else:
                        registrador.info(
                            "  ❓ Sin Patrón Claro (confianza %.1f%% < umbral).",
                            resultado.confianza * 100,
                        )
                else:
                    registrador.warning(
                        "  ⚠ Inferencia fallida en ciclo #%d",
                        numero_ciclo,
                    )

        else:
            registrador.warning("  ⚠ Conversión a tensor fallida en ciclo #%d",
                                numero_ciclo)

        # ── Esperar hasta el siguiente ciclo ──────────────────────
        _esperar_siguiente_ciclo(config.intervalo_captura_seg, inicio_ciclo)

    # ══════════════════════════════════════════════════════════════
    #  Resumen final
    # ══════════════════════════════════════════════════════════════
    registrador.info("═" * 60)
    registrador.info("  RESUMEN FINAL")
    registrador.info("═" * 60)
    registrador.info("  Ciclos ejecutados:   %d", numero_ciclo - 1)
    registrador.info("  Capturas exitosas:   %d", capturador.capturas_realizadas)
    registrador.info("  Tensores creados:    %d", tuberia.total_procesados)
    registrador.info("  Errores de pipeline: %d", tuberia.total_errores)
    registrador.info("  Tensores en buffer:  %d", tuberia.tamano_buffer)

    # Mostrar info del último tensor si existe
    ultimo = tuberia.obtener_ultimo_tensor()
    if ultimo is not None:
        registrador.info("  Último tensor forma: %s", list(ultimo.shape))

    # Mostrar resumen de la tubería
    print("\n" + tuberia.resumen())

    # Mostrar resumen del procesador de visión si estaba activo
    if vision_activa:
        registrador.info("  Inferencias DINOv2:  %d", procesador_vision.total_inferencias)
        registrador.info("  Errores de visión:   %d", procesador_vision.total_errores)
        print("\n" + procesador_vision.resumen_estadisticas())


def _esperar_siguiente_ciclo(intervalo_seg: int, inicio_ciclo: float) -> None:
    """Espera el tiempo restante hasta completar el intervalo."""
    tiempo_procesamiento = time.time() - inicio_ciclo
    tiempo_espera = max(0, intervalo_seg - tiempo_procesamiento)

    if tiempo_espera > 0:
        registrador = logging.getLogger("bucle_principal")
        registrador.info(
            "  ⏳ Esperando %.1f seg para siguiente captura…",
            tiempo_espera,
        )
        # Dividir la espera para poder responder a Ctrl+C rápidamente
        pasos = int(tiempo_espera / 0.5)
        for _ in range(pasos):
            if not ejecutando:
                return
            time.sleep(0.5)
        # Esperar el residuo
        residuo = tiempo_espera - (pasos * 0.5)
        if residuo > 0 and ejecutando:
            time.sleep(residuo)


# ══════════════════════════════════════════════════════════════════
#  Punto de entrada
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    """Función principal que orquesta todo el sistema."""
    argumentos = parsear_argumentos()

    # Configurar logging
    nivel_log = logging.DEBUG if argumentos.debug else logging.INFO
    configurar_logging(nivel_log)

    registrador = logging.getLogger("principal")

    # ── Construir configuración de captura ────────────────────────
    config = Configuracion(
        intervalo_captura_seg=argumentos.intervalo,
        navegador_sin_cabeza=not argumentos.visible,
        tamano_tensor=tuple(argumentos.tamano),
        capacidad_buffer=argumentos.buffer,
        dispositivo=argumentos.dispositivo,
    )

    # Mostrar configuración de captura
    print("\n" + config.resumen() + "\n")

    # ── Inicializar módulos de captura ────────────────────────────
    capturador = CapturadorPantalla(config)
    tuberia = TuberiaTensores(config)

    # ── Inicializar procesador de visión (si está activo) ─────────
    procesador_vision = None
    activar_vision = argumentos.con_vision and not argumentos.sin_vision

    if activar_vision:
        procesador_vision = inicializar_procesador_vision(
            umbral_confianza=argumentos.umbral_confianza,
            fraccion_vram=argumentos.fraccion_vram,
        )
        if procesador_vision is None:
            registrador.warning(
                "Procesador de visión no disponible. "
                "Continuando solo con captura + tensores."
            )
    else:
        registrador.info("Detector de patrones DESACTIVADO (--sin-vision).")

    try:
        # Iniciar navegador
        registrador.info("Inicializando capturador de pantalla…")
        capturador.iniciar()

        # Ejecutar bucle de captura + detección
        ejecutar_bucle_captura(
            config=config,
            capturador=capturador,
            tuberia=tuberia,
            procesador_vision=procesador_vision,
            max_capturas=argumentos.capturas,
        )

    except KeyboardInterrupt:
        registrador.info("Interrupción por teclado.")

    except Exception as error:
        registrador.error("Error fatal: %s", error, exc_info=True)

    finally:
        # Siempre cerrar el navegador
        capturador.detener()

        # Liberar recursos de visión si estaban activos
        if procesador_vision is not None:
            procesador_vision.liberar_recursos()

        registrador.info("Sistema finalizado correctamente. ✔")


if __name__ == "__main__":
    main()
