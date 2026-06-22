"""
Módulo Tubería de Tensores en Memoria (In-Memory Tensor Pipeline).
Recibe bytes PNG de capturas de pantalla y los transforma directamente
a tensores PyTorch sin guardar nada en disco.
Mantiene un buffer circular de los últimos N tensores.
"""

import io
import time
import logging
from typing import Optional, List, Dict, Any
from collections import deque
from datetime import datetime

import numpy as np
from PIL import Image
import torch

from .configuracion import Configuracion

# ── Logger del módulo ─────────────────────────────────────────────
registrador = logging.getLogger("tuberia_tensores")


class TuberiaTensores:
    """
    Pipeline que convierte bytes PNG → PIL Image → NumPy → Tensor PyTorch.
    Todo ocurre en memoria.  Los tensores se almacenan en un buffer
    circular (deque) con capacidad configurable.
    """

    def __init__(self, config: Configuracion) -> None:
        self._config = config
        self._dispositivo = torch.device(config.dispositivo)

        # Buffer circular: cada entrada es un dict con tensor + metadatos
        self._buffer: deque = deque(maxlen=config.capacidad_buffer)

        # Estadísticas
        self._total_procesados: int = 0
        self._total_errores: int = 0

    # ── Procesamiento principal ───────────────────────────────────

    def procesar_captura(self, datos_png: bytes) -> Optional[torch.Tensor]:
        """
        Recibe bytes PNG en bruto y devuelve un tensor PyTorch listo
        para inferencia.  También lo almacena en el buffer interno.

        Pipeline:
          bytes PNG → PIL.Image (RGB) → resize → numpy → torch.Tensor
          → normalizar [0,1] → permutar a (C, H, W) → mover a dispositivo

        Retorna None si hay error en la conversión.
        """
        try:
            marca_tiempo = datetime.now()

            # 1. Bytes → Imagen PIL
            imagen_pil = self._bytes_a_imagen(datos_png)

            # 2. Redimensionar
            imagen_redim = self._redimensionar(imagen_pil)

            # 3. Imagen PIL → NumPy array
            arreglo_np = self._imagen_a_numpy(imagen_redim)

            # 4. NumPy → Tensor PyTorch
            tensor = self._numpy_a_tensor(arreglo_np)

            # 5. Almacenar en buffer con metadatos
            entrada = {
                "tensor": tensor,
                "marca_tiempo": marca_tiempo,
                "forma": tuple(tensor.shape),
                "indice": self._total_procesados,
            }
            self._buffer.append(entrada)
            self._total_procesados += 1

            registrador.info(
                "Tensor #%d creado — forma: %s — dispositivo: %s — "
                "buffer: %d/%d",
                self._total_procesados,
                tensor.shape,
                tensor.device,
                len(self._buffer),
                self._config.capacidad_buffer,
            )

            return tensor

        except Exception as error:
            self._total_errores += 1
            registrador.error(
                "Error al procesar captura #%d: %s",
                self._total_procesados + 1,
                error,
            )
            return None

    # ── Acceso al buffer ──────────────────────────────────────────

    def obtener_ultimo_tensor(self) -> Optional[torch.Tensor]:
        """Devuelve el tensor más reciente del buffer, o None."""
        if self._buffer:
            return self._buffer[-1]["tensor"]
        return None

    def obtener_ultimos_n(self, n: int) -> List[torch.Tensor]:
        """Devuelve los últimos n tensores como lista."""
        entradas = list(self._buffer)[-n:]
        return [e["tensor"] for e in entradas]

    def obtener_lote(self, n: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Devuelve un lote (batch) apilado de los últimos n tensores.
        Forma resultante: (N, C, H, W).
        """
        if not self._buffer:
            return None
        entradas = list(self._buffer) if n is None else list(self._buffer)[-n:]
        tensores = [e["tensor"] for e in entradas]
        return torch.stack(tensores, dim=0)

    def obtener_metadatos_buffer(self) -> List[Dict[str, Any]]:
        """Devuelve los metadatos de cada entrada en el buffer."""
        return [
            {
                "indice": e["indice"],
                "marca_tiempo": e["marca_tiempo"].isoformat(),
                "forma": e["forma"],
            }
            for e in self._buffer
        ]

    @property
    def tamano_buffer(self) -> int:
        return len(self._buffer)

    @property
    def total_procesados(self) -> int:
        return self._total_procesados

    @property
    def total_errores(self) -> int:
        return self._total_errores

    def limpiar_buffer(self) -> None:
        """Vacía el buffer de tensores."""
        self._buffer.clear()
        registrador.info("Buffer de tensores limpiado.")

    # ── Pasos internos del pipeline ───────────────────────────────

    def _bytes_a_imagen(self, datos_png: bytes) -> Image.Image:
        """Convierte bytes PNG a una imagen PIL en modo RGB."""
        flujo = io.BytesIO(datos_png)
        imagen = Image.open(flujo)
        if imagen.mode != "RGB":
            imagen = imagen.convert("RGB")
        return imagen

    def _redimensionar(self, imagen: Image.Image) -> Image.Image:
        """Redimensiona la imagen al tamaño configurado para el tensor."""
        alto_objetivo, ancho_objetivo = self._config.tamano_tensor
        return imagen.resize(
            (ancho_objetivo, alto_objetivo),
            Image.Resampling.LANCZOS,
        )

    def _imagen_a_numpy(self, imagen: Image.Image) -> np.ndarray:
        """Convierte imagen PIL a arreglo NumPy float32 (H, W, C)."""
        arreglo = np.array(imagen, dtype=np.float32)
        if self._config.normalizar_tensor:
            arreglo /= 255.0   # Escalar a [0, 1]
        return arreglo

    def _numpy_a_tensor(self, arreglo: np.ndarray) -> torch.Tensor:
        """
        Convierte arreglo NumPy (H, W, C) a tensor PyTorch (C, H, W)
        y lo mueve al dispositivo configurado.
        """
        tensor = torch.from_numpy(arreglo)
        # Permutar de (H, W, C) a (C, H, W) — formato estándar PyTorch
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.to(self._dispositivo)
        return tensor

    # ── Recorte de mitad derecha ──────────────────────────────────

    def recortar_mitad_derecha(self, datos_png: bytes) -> Optional[torch.Tensor]:
        """
        Toma bytes PNG, recorta la MITAD DERECHA de la imagen y
        devuelve un tensor listo para inferencia (C, H, W).
        No almacena en el buffer — es un análisis auxiliar.
        """
        try:
            imagen_pil = self._bytes_a_imagen(datos_png)
            ancho_original = imagen_pil.width
            # Recortar mitad derecha: desde la mitad hasta el final
            mitad_x = ancho_original // 2
            imagen_mitad = imagen_pil.crop((mitad_x, 0, ancho_original, imagen_pil.height))
            imagen_redim = imagen_mitad.resize(
                (self._config.tamano_tensor[1], self._config.tamano_tensor[0]),
                Image.Resampling.LANCZOS,
            )
            arreglo_np = self._imagen_a_numpy(imagen_redim)
            tensor = self._numpy_a_tensor(arreglo_np)
            registrador.debug(
                "Tensor mitad derecha creado — forma: %s",
                tensor.shape,
            )
            return tensor
        except Exception as error:
            registrador.error("Error al recortar mitad derecha: %s", error)
            return None

    # ── Resumen ───────────────────────────────────────────────────

    def resumen(self) -> str:
        """Devuelve un resumen del estado del pipeline."""
        lineas = [
            "┌──────────────────────────────────────────┐",
            "│   ESTADO DE LA TUBERÍA DE TENSORES        │",
            "├──────────────────────────────────────────┤",
            f"│ Procesados:  {self._total_procesados:>6}",
            f"│ Errores:     {self._total_errores:>6}",
            f"│ En buffer:   {len(self._buffer):>6} / {self._config.capacidad_buffer}",
            f"│ Dispositivo: {self._dispositivo}",
            f"│ Forma:       ({self._config.canales_color}, "
            f"{self._config.tamano_tensor[0]}, {self._config.tamano_tensor[1]})",
            "└──────────────────────────────────────────┘",
        ]
        return "\n".join(lineas)
