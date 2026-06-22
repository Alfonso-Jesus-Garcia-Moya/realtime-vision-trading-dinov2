"""
Módulo Procesador de Visión — Inferencia en Producción.
Mantiene el modelo DINOv2 + cabeza de clasificación de forma persistente
en la VRAM de la GPU.  Expone una interfaz simple para detectar patrones
técnicos a partir de tensores generados por la tubería de captura.

Características de producción:
    - Limitación estricta de VRAM (40% máximo por proceso)
    - Transferencia eficiente CPU→GPU con non_blocking=True
    - Inferencia bajo torch.inference_mode() (sin gradientes)
    - Manejo robusto de OOM con fallback a CPU
    - Logging estructurado de resultados
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import torch

from .configuracion_vision import (
    ConfiguracionVision,
    MAPA_CLASES_PATRONES,
)
from .modelo_vision import ClasificadorPatronesVisuales

# ── Logger del módulo ─────────────────────────────────────────────
registrador = logging.getLogger("procesador_vision")


# ══════════════════════════════════════════════════════════════════
#  Estructura de resultado de detección
# ══════════════════════════════════════════════════════════════════

@dataclass
class ResultadoDeteccion:
    """Resultado estructurado de una detección de patrón."""
    marca_tiempo: str
    indice_clase: int
    nombre_patron: str
    confianza: float
    es_patron_valido: bool          # True si confianza >= umbral
    tiempo_inferencia_ms: float     # Milisegundos que tomó la inferencia
    dispositivo_usado: str          # "cuda" o "cpu"

    def formato_consola(self) -> str:
        """Formato legible para la consola del orquestador."""
        icono = "🎯" if self.es_patron_valido else "❓"
        nombre_mostrado = self.nombre_patron.replace("_", " ")
        return (
            f"[{self.marca_tiempo}] {icono} Patrón Detectado: "
            f"{nombre_mostrado} | Confianza: {self.confianza:.1%} | "
            f"Inferencia: {self.tiempo_inferencia_ms:.1f}ms "
            f"({self.dispositivo_usado})"
        )

    def a_diccionario(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            "marca_tiempo": self.marca_tiempo,
            "indice_clase": self.indice_clase,
            "nombre_patron": self.nombre_patron,
            "confianza": round(self.confianza, 4),
            "es_patron_valido": self.es_patron_valido,
            "tiempo_inferencia_ms": round(self.tiempo_inferencia_ms, 2),
            "dispositivo_usado": self.dispositivo_usado,
        }


# ══════════════════════════════════════════════════════════════════
#  Procesador de Visión (Clase de Producción)
# ══════════════════════════════════════════════════════════════════

class ProcesadorVision:
    """
    Procesador de visión de producción que mantiene el modelo DINOv2
    cargado persistentemente en GPU/CPU para inferencia en tiempo real.

    Flujo:
        1. Inicialización: carga backbone + cabeza entrenada en VRAM.
        2. detectar_patron(tensor) → ResultadoDeteccion
        3. Si confianza < umbral → "Sin_Patron"

    Protecciones:
        - Limita VRAM al 40% del total (configurable).
        - Captura OOM y hace fallback a CPU transparentemente.
        - inference_mode() elimina rastreo de gradientes.
    """

    def __init__(self, config: Optional[ConfiguracionVision] = None) -> None:
        self._config = config or ConfiguracionVision()
        self._modelo: Optional[ClasificadorPatronesVisuales] = None
        self._dispositivo: Optional[torch.device] = None
        self._esta_inicializado: bool = False

        # Estadísticas de producción
        self._total_inferencias: int = 0
        self._total_errores: int = 0
        self._total_patrones_validos: int = 0
        self._total_sin_patron: int = 0

    # ── Inicialización ───────────────────────────────────────────

    def inicializar(self) -> None:
        """
        Inicializa el modelo completo: backbone DINOv2 + cabeza de
        clasificación.  Aplica el límite de VRAM y carga los pesos
        entrenados si existen.

        Raises:
            RuntimeError: Si no se puede inicializar el modelo.
        """
        registrador.info("Inicializando Procesador de Visión…")

        # ── Paso 1: Configurar dispositivo con límite VRAM ────────
        self._dispositivo = self._configurar_dispositivo()

        # ── Paso 2: Construir modelo ──────────────────────────────
        try:
            registrador.info("Construyendo clasificador DINOv2 + cabeza MLP…")
            self._modelo = ClasificadorPatronesVisuales(self._config)

        except Exception as error:
            registrador.error(
                "Error al construir el modelo: %s", error, exc_info=True
            )
            raise RuntimeError(
                f"No se pudo construir el clasificador: {error}"
            ) from error

        # ── Paso 3: Cargar pesos entrenados de la cabeza ──────────
        ruta_pesos = self._config.ruta_pesos_cabeza
        if os.path.exists(ruta_pesos):
            try:
                self._modelo.cargar_pesos_cabeza(ruta_pesos)
                registrador.info(
                    "Pesos de la cabeza cargados desde: %s", ruta_pesos
                )
            except Exception as error:
                registrador.warning(
                    "No se pudieron cargar los pesos desde %s: %s. "
                    "Usando cabeza sin entrenar (predicciones aleatorias).",
                    ruta_pesos,
                    error,
                )
        else:
            registrador.warning(
                "No se encontraron pesos entrenados en: %s. "
                "La cabeza no está entrenada — las predicciones serán "
                "aleatorias. Ejecuta el entrenador primero:\n"
                "  python -m detector_patrones.entrenador_patrones",
                ruta_pesos,
            )

        # ── Paso 4: Mover modelo al dispositivo ──────────────────
        try:
            self._modelo.to(self._dispositivo)
            self._modelo.eval()  # Modo evaluación permanente
            registrador.info(
                "Modelo cargado en %s exitosamente.", self._dispositivo
            )
        except torch.cuda.OutOfMemoryError:
            registrador.error(
                "OOM al mover modelo a GPU. Intentando fallback a CPU…"
            )
            torch.cuda.empty_cache()
            self._dispositivo = torch.device("cpu")
            self._modelo.to(self._dispositivo)
            self._modelo.eval()
            registrador.info("Modelo cargado en CPU (fallback).")

        self._esta_inicializado = True

        # ── Mostrar resumen ───────────────────────────────────────
        registrador.info(self._modelo.resumen_modelo())
        registrador.info("Procesador de Visión inicializado correctamente. ✔")

    # ── Configuración de dispositivo ─────────────────────────────

    def _configurar_dispositivo(self) -> torch.device:
        """
        Configura el dispositivo de cómputo con protección de VRAM.

        Returns:
            torch.device configurado (cuda o cpu).
        """
        if (
            self._config.dispositivo_preferido == "cuda"
            and torch.cuda.is_available()
        ):
            try:
                # Aplicar límite estricto de VRAM ANTES de cualquier asignación
                torch.cuda.set_per_process_memory_fraction(
                    self._config.fraccion_vram_maxima
                )

                dispositivo = torch.device("cuda")
                nombre_gpu = torch.cuda.get_device_name(0)
                propiedades = torch.cuda.get_device_properties(0)
                vram_total_gb = propiedades.total_mem / (1024 ** 3)
                vram_limite_gb = vram_total_gb * self._config.fraccion_vram_maxima

                registrador.info("┌─ Hardware GPU ─────────────────────────────┐")
                registrador.info("│ GPU:         %s", nombre_gpu)
                registrador.info(
                    "│ VRAM total:  %.1f GB", vram_total_gb
                )
                registrador.info(
                    "│ VRAM límite: %.1f GB (%.0f%% del total)",
                    vram_limite_gb,
                    self._config.fraccion_vram_maxima * 100,
                )
                registrador.info("└────────────────────────────────────────────┘")

                return dispositivo

            except Exception as error:
                registrador.warning(
                    "Error al configurar GPU: %s. Fallback a CPU.", error
                )
                return torch.device("cpu")
        else:
            registrador.info(
                "Usando CPU (GPU %s).",
                "no disponible" if not torch.cuda.is_available()
                else "no preferida"
            )
            return torch.device("cpu")

    # ── Detección de patrones ────────────────────────────────────

    def detectar_patron(
        self, tensor_imagen: torch.Tensor
    ) -> Optional[ResultadoDeteccion]:
        """
        Detecta el patrón técnico presente en una imagen capturada.

        Args:
            tensor_imagen: Tensor de forma (3, 224, 224) con valores
                          en [0, 1], generado por TuberiaTensores.

        Returns:
            ResultadoDeteccion con la clase predicha, confianza y
            metadatos.  None si ocurre un error irrecuperable.
        """
        if not self._esta_inicializado:
            registrador.error(
                "Procesador no inicializado. Llamar a inicializar() primero."
            )
            return None

        marca_tiempo = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        inicio = time.perf_counter()

        try:
            # ── Mover tensor a GPU eficientemente ─────────────────
            tensor_gpu = tensor_imagen.to(
                self._dispositivo,
                non_blocking=True,
            )

            # ── Ejecutar inferencia ───────────────────────────────
            indice_clase, nombre_clase, confianza, probabilidades = (
                self._modelo.predecir(tensor_gpu)
            )

            tiempo_inferencia_ms = (time.perf_counter() - inicio) * 1000

            # ── Aplicar umbral de confianza ───────────────────────
            es_patron_valido = confianza >= self._config.umbral_confianza_minima

            if not es_patron_valido:
                nombre_clase = "Sin_Patron"
                indice_clase = self._config.indice_clase_sin_patron
                self._total_sin_patron += 1
            else:
                self._total_patrones_validos += 1

            self._total_inferencias += 1

            # ── Construir resultado ───────────────────────────────
            resultado = ResultadoDeteccion(
                marca_tiempo=marca_tiempo,
                indice_clase=indice_clase,
                nombre_patron=nombre_clase,
                confianza=confianza,
                es_patron_valido=es_patron_valido,
                tiempo_inferencia_ms=tiempo_inferencia_ms,
                dispositivo_usado=str(self._dispositivo),
            )

            return resultado

        except torch.cuda.OutOfMemoryError:
            self._total_errores += 1
            registrador.error(
                "⚠ OOM durante inferencia. Limpiando caché CUDA y "
                "reintentando en CPU…"
            )
            torch.cuda.empty_cache()

            # Reintentar en CPU como emergencia
            try:
                return self._inferencia_emergencia_cpu(
                    tensor_imagen, marca_tiempo, inicio
                )
            except Exception as error_cpu:
                registrador.error(
                    "Error también en CPU: %s", error_cpu
                )
                return None

        except Exception as error:
            self._total_errores += 1
            registrador.error(
                "Error en inferencia: %s", error, exc_info=True
            )
            return None

    # ── Inferencia de emergencia en CPU ──────────────────────────

    def _inferencia_emergencia_cpu(
        self,
        tensor_imagen: torch.Tensor,
        marca_tiempo: str,
        inicio: float,
    ) -> ResultadoDeteccion:
        """
        Ejecuta inferencia en CPU como fallback ante OOM en GPU.
        """
        registrador.warning("Ejecutando inferencia de emergencia en CPU…")

        tensor_cpu = tensor_imagen.to("cpu")
        modelo_cpu = self._modelo.to("cpu")

        indice_clase, nombre_clase, confianza, _ = modelo_cpu.predecir(tensor_cpu)

        # Devolver modelo a GPU si es posible
        try:
            self._modelo.to(self._dispositivo)
        except torch.cuda.OutOfMemoryError:
            registrador.warning(
                "No se pudo devolver modelo a GPU. Permanece en CPU."
            )
            self._dispositivo = torch.device("cpu")

        tiempo_inferencia_ms = (time.perf_counter() - inicio) * 1000
        es_patron_valido = confianza >= self._config.umbral_confianza_minima

        if not es_patron_valido:
            nombre_clase = "Sin_Patron"
            indice_clase = self._config.indice_clase_sin_patron

        return ResultadoDeteccion(
            marca_tiempo=marca_tiempo,
            indice_clase=indice_clase,
            nombre_patron=nombre_clase,
            confianza=confianza,
            es_patron_valido=es_patron_valido,
            tiempo_inferencia_ms=tiempo_inferencia_ms,
            dispositivo_usado="cpu (emergencia)",
        )

    # ── Liberación de recursos ───────────────────────────────────

    def liberar_recursos(self) -> None:
        """Libera el modelo de la VRAM y limpia la caché CUDA."""
        if self._modelo is not None:
            self._modelo.cpu()
            del self._modelo
            self._modelo = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._esta_inicializado = False
        registrador.info("Recursos del procesador de visión liberados. ✔")

    # ── Propiedades y estadísticas ───────────────────────────────

    @property
    def esta_inicializado(self) -> bool:
        return self._esta_inicializado

    @property
    def dispositivo(self) -> Optional[torch.device]:
        return self._dispositivo

    @property
    def total_inferencias(self) -> int:
        return self._total_inferencias

    @property
    def total_errores(self) -> int:
        return self._total_errores

    def resumen_estadisticas(self) -> str:
        """Devuelve un resumen de las estadísticas de producción."""
        lineas = [
            "┌──────────────────────────────────────────────────┐",
            "│   ESTADÍSTICAS DEL PROCESADOR DE VISIÓN           │",
            "├──────────────────────────────────────────────────┤",
            f"│ Inferencias totales:   {self._total_inferencias:>6}",
            f"│ Patrones válidos:      {self._total_patrones_validos:>6}",
            f"│ Sin patrón claro:      {self._total_sin_patron:>6}",
            f"│ Errores:               {self._total_errores:>6}",
            f"│ Dispositivo:           {self._dispositivo}",
            f"│ Umbral confianza:      {self._config.umbral_confianza_minima:.0%}",
            "└──────────────────────────────────────────────────┘",
        ]
        return "\n".join(lineas)
