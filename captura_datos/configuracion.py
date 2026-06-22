"""
Módulo de Configuración.
Centraliza todas las constantes y parámetros del sistema de captura.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Configuracion:
    """Parámetros globales del sistema de captura de datos."""

    # ── URL objetivo ──────────────────────────────────────────────
    url_grafico: str = (
        "https://finance.yahoo.com/chart/%5EGSPC"
    )

    # ── Temporización ─────────────────────────────────────────────
    intervalo_captura_seg: int = 10          # Cada cuántos segundos se captura
    tiempo_espera_carga_seg: int = 15        # Timeout para que cargue la página
    tiempo_espera_grafico_seg: int = 3       # Timeout por selector (rápido, itera lista)

    # ── Selectores CSS del elemento gráfico ───────────────────────
    # Yahoo Finance renderiza el chart dentro de un <canvas> envuelto
    # por estos contenedores.  Usamos una lista de selectores para
    # intentar en orden de especificidad (el primero que coincida gana).
    selectores_grafico: list = field(default_factory=lambda: [
        ".chartContainer canvas",                     # ✔ Selector confirmado Yahoo Finance
        "canvas.chartiq-chart",                       # Canvas principal del chart
        "[data-testid='chart-container'] canvas",     # Contenedor de pruebas
        ".chart-container canvas",                    # Contenedor genérico
        "#chartiq-chart-container canvas",            # ID directo
        "canvas",                                     # Último recurso: primer canvas
    ])

    # ── Navegador ─────────────────────────────────────────────────
    navegador_sin_cabeza: bool = True        # Headless (True) o visible (False)
    ancho_ventana: int = 1920
    alto_ventana: int = 1080

    # ── Tensor ────────────────────────────────────────────────────
    tamano_tensor: Tuple[int, int] = (224, 224)   # Alto x Ancho al redimensionar
    canales_color: int = 3                          # RGB
    normalizar_tensor: bool = True                  # Escalar valores a [0, 1]
    capacidad_buffer: int = 100                     # Máximo de tensores en el buffer circular

    # ── Dispositivo de cómputo ────────────────────────────────────
    dispositivo: str = "cpu"                         # "cpu" o "cuda"

    def resumen(self) -> str:
        """Devuelve un resumen legible de la configuración activa."""
        lineas = [
            "╔══════════════════════════════════════════╗",
            "║   CONFIGURACIÓN DEL SISTEMA DE CAPTURA   ║",
            "╠══════════════════════════════════════════╣",
            f"║ URL:              {self.url_grafico[:38]}",
            f"║ Intervalo:        cada {self.intervalo_captura_seg} segundos",
            f"║ Headless:         {self.navegador_sin_cabeza}",
            f"║ Ventana:          {self.ancho_ventana}×{self.alto_ventana}",
            f"║ Tamaño tensor:    {self.tamano_tensor}",
            f"║ Normalizar:       {self.normalizar_tensor}",
            f"║ Buffer máximo:    {self.capacidad_buffer} tensores",
            f"║ Dispositivo:      {self.dispositivo}",
            "╚══════════════════════════════════════════╝",
        ]
        return "\n".join(lineas)
