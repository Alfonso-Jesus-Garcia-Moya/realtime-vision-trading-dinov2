"""
Módulo de Captura de Datos Financieros en Tiempo Real.
Captura screenshots del gráfico de Yahoo Finance y los convierte
a tensores en memoria (In-Memory Tensor Pipeline).
"""

from .configuracion import Configuracion
from .capturador_pantalla import CapturadorPantalla
from .tuberia_tensores import TuberiaTensores

__all__ = [
    "Configuracion",
    "CapturadorPantalla",
    "TuberiaTensores",
]
