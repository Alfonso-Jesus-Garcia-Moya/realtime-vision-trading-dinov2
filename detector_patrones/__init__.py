"""
Paquete Detector de Patrones Técnicos.
Utiliza DINOv2 como extractor de características visuales y una cabeza
de clasificación ligera para identificar patrones de análisis técnico
en gráficos bursátiles capturados en tiempo real.
"""

from .configuracion_vision import ConfiguracionVision, MAPA_CLASES_PATRONES
from .modelo_vision import ClasificadorPatronesVisuales
from .procesador_vision import ProcesadorVision

__all__ = [
    "ConfiguracionVision",
    "MAPA_CLASES_PATRONES",
    "ClasificadorPatronesVisuales",
    "ProcesadorVision",
]
