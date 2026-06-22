"""
Módulo del Modelo de Visión — Clasificador de Patrones Técnicos.
Implementa un clasificador basado en DINOv2 (backbone congelado) con
una cabeza de clasificación MLP ligera entrenada sobre patrones técnicos
de análisis bursátil.

Arquitectura:
    DINOv2-ViT-B/14 (congelado) → [CLS] embedding (768-d) → MLP → clases
"""

import os
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuracion_vision import ConfiguracionVision, MAPA_CLASES_PATRONES

# ── Logger del módulo ─────────────────────────────────────────────
registrador = logging.getLogger("modelo_vision")


class CabezaClasificacionMLP(nn.Module):
    """
    Cabeza de clasificación ligera: MLP con una capa oculta,
    activación ReLU, Dropout y proyección final al número de clases.

    Arquitectura:
        Linear(768 → 256) → ReLU → Dropout(0.3) → Linear(256 → N_clases)
    """

    def __init__(
        self,
        dimension_entrada: int,
        dimension_oculta: int,
        numero_clases: int,
        tasa_dropout: float,
    ) -> None:
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dimension_entrada, dimension_oculta),
            nn.ReLU(inplace=True),
            nn.Dropout(p=tasa_dropout),
            nn.Linear(dimension_oculta, numero_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Proyecta embeddings a logits de clases."""
        return self.red(x)


class CabezaClasificacionLineal(nn.Module):
    """
    Cabeza de clasificación simple: una sola capa lineal.
    Útil cuando los datos de entrenamiento son muy limitados.
    """

    def __init__(
        self,
        dimension_entrada: int,
        numero_clases: int,
    ) -> None:
        super().__init__()
        self.lineal = nn.Linear(dimension_entrada, numero_clases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lineal(x)


class ClasificadorPatronesVisuales(nn.Module):
    """
    Clasificador completo: DINOv2 (backbone congelado) + Cabeza MLP/Lineal.

    El backbone extrae embeddings visuales de alta calidad (768-d) a
    partir de imágenes 224×224.  La cabeza de clasificación, que es la
    única parte entrenable, proyecta esos embeddings a las clases de
    patrones técnicos definidos en la configuración.

    Parámetros del backbone: ~86M (congelados, no consumen grad memory)
    Parámetros de la cabeza:  ~200K (entrenables)
    """

    def __init__(self, config: ConfiguracionVision) -> None:
        super().__init__()
        self._config = config
        self._mapa_clases = MAPA_CLASES_PATRONES

        # ── Cargar backbone DINOv2 ────────────────────────────────
        registrador.info(
            "Cargando backbone DINOv2: %s desde %s…",
            config.nombre_modelo_base,
            config.repositorio_torch_hub,
        )
        try:
            self.backbone: nn.Module = torch.hub.load(
                config.repositorio_torch_hub,
                config.nombre_modelo_base,
                pretrained=True,
            )
        except Exception as error:
            registrador.error(
                "Error al cargar DINOv2 desde torch.hub: %s. "
                "Verificar conexión a internet en la primera descarga.",
                error,
            )
            raise RuntimeError(
                f"No se pudo cargar el backbone DINOv2: {error}"
            ) from error

        registrador.info("Backbone DINOv2 cargado exitosamente.")

        # ── Congelar backbone ─────────────────────────────────────
        if config.congelar_backbone:
            self._congelar_backbone()

        # ── Construir cabeza de clasificación ─────────────────────
        if config.usar_mlp:
            self.cabeza = CabezaClasificacionMLP(
                dimension_entrada=config.dimension_embedding,
                dimension_oculta=config.dimension_oculta_mlp,
                numero_clases=config.numero_clases,
                tasa_dropout=config.tasa_dropout,
            )
            registrador.info(
                "Cabeza MLP creada: %d → %d → %d clases (dropout=%.2f)",
                config.dimension_embedding,
                config.dimension_oculta_mlp,
                config.numero_clases,
                config.tasa_dropout,
            )
        else:
            self.cabeza = CabezaClasificacionLineal(
                dimension_entrada=config.dimension_embedding,
                numero_clases=config.numero_clases,
            )
            registrador.info(
                "Cabeza Lineal creada: %d → %d clases",
                config.dimension_embedding,
                config.numero_clases,
            )

        # ── Registrar normalización ImageNet ──────────────────────
        self.register_buffer(
            "media_imagenet",
            torch.tensor(config.media_imagenet).view(3, 1, 1),
        )
        self.register_buffer(
            "desviacion_imagenet",
            torch.tensor(config.desviacion_imagenet).view(3, 1, 1),
        )

    # ── Congelamiento del backbone ───────────────────────────────

    def _congelar_backbone(self) -> None:
        """Congela todos los parámetros del backbone DINOv2."""
        parametros_congelados = 0
        for parametro in self.backbone.parameters():
            parametro.requires_grad = False
            parametros_congelados += parametro.numel()

        # Poner en modo evaluación permanente
        self.backbone.eval()

        registrador.info(
            "Backbone congelado: %s parámetros (%.1f M) sin gradientes.",
            f"{parametros_congelados:,}",
            parametros_congelados / 1e6,
        )

    # ── Forward pass ─────────────────────────────────────────────

    def forward(self, imagenes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass completo: normalización → backbone → cabeza.

        Args:
            imagenes: Tensor de forma (B, 3, 224, 224) con valores en [0, 1].

        Returns:
            Logits de forma (B, numero_clases).
        """
        # Normalizar con media/desviación ImageNet
        imagenes_normalizadas = (imagenes - self.media_imagenet) / self.desviacion_imagenet

        # Extraer embeddings del backbone (CLS token)
        with torch.no_grad():
            embeddings = self.backbone(imagenes_normalizadas)  # (B, 768)

        # Pasar por la cabeza de clasificación
        logits = self.cabeza(embeddings)  # (B, numero_clases)
        return logits

    # ── Inferencia optimizada ────────────────────────────────────

    @torch.inference_mode()
    def predecir(
        self, imagen: torch.Tensor
    ) -> Tuple[int, str, float, torch.Tensor]:
        """
        Realiza inferencia sobre una sola imagen o un lote.
        Ejecuta bajo inference_mode() para eliminar el rastreo de
        gradientes y reducir el uso de VRAM.

        Args:
            imagen: Tensor de forma (3, 224, 224) o (B, 3, 224, 224)
                    con valores en [0, 1].

        Returns:
            Tupla con:
                - indice_clase (int): Índice de la clase predicha.
                - nombre_clase (str): Nombre legible de la clase.
                - confianza (float): Score de confianza [0, 1] (softmax).
                - probabilidades (Tensor): Vector completo de probabilidades.
        """
        # Asegurar dimensión de lote
        if imagen.dim() == 3:
            imagen = imagen.unsqueeze(0)  # (1, 3, 224, 224)

        # Forward
        logits = self.forward(imagen)  # (B, numero_clases)

        # Softmax para obtener probabilidades
        probabilidades = F.softmax(logits, dim=1)  # (B, numero_clases)

        # Obtener clase con mayor probabilidad
        confianza_max, indice_clase = probabilidades.max(dim=1)

        # Para una sola imagen, extraer escalares
        indice = indice_clase[0].item()
        confianza = confianza_max[0].item()
        nombre = self._mapa_clases.get(indice, "Desconocido")

        return indice, nombre, confianza, probabilidades[0]

    # ── Gestión de pesos de la cabeza ────────────────────────────

    def guardar_pesos_cabeza(self, ruta: Optional[str] = None) -> str:
        """
        Guarda SOLO los pesos de la cabeza de clasificación (no el backbone).

        Args:
            ruta: Ruta del archivo .pth. Si None, usa la configuración.

        Returns:
            Ruta donde se guardaron los pesos.
        """
        ruta_destino = ruta or self._config.ruta_pesos_cabeza

        # Crear directorio si no existe
        directorio = os.path.dirname(ruta_destino)
        if directorio:
            os.makedirs(directorio, exist_ok=True)

        estado = {
            "estado_cabeza": self.cabeza.state_dict(),
            "config": {
                "nombre_modelo_base": self._config.nombre_modelo_base,
                "dimension_embedding": self._config.dimension_embedding,
                "numero_clases": self._config.numero_clases,
                "dimension_oculta_mlp": self._config.dimension_oculta_mlp,
                "tasa_dropout": self._config.tasa_dropout,
                "usar_mlp": self._config.usar_mlp,
            },
            "mapa_clases": self._mapa_clases,
        }

        torch.save(estado, ruta_destino)
        registrador.info("Pesos de la cabeza guardados en: %s", ruta_destino)
        return ruta_destino

    def cargar_pesos_cabeza(self, ruta: Optional[str] = None) -> None:
        """
        Carga los pesos de la cabeza de clasificación desde un archivo .pth.

        Args:
            ruta: Ruta del archivo .pth. Si None, usa la configuración.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            RuntimeError: Si los pesos no son compatibles.
        """
        ruta_origen = ruta or self._config.ruta_pesos_cabeza

        if not os.path.exists(ruta_origen):
            raise FileNotFoundError(
                f"No se encontraron pesos de la cabeza en: {ruta_origen}"
            )

        estado = torch.load(ruta_origen, map_location="cpu", weights_only=True)
        self.cabeza.load_state_dict(estado["estado_cabeza"])
        registrador.info("Pesos de la cabeza cargados desde: %s", ruta_origen)

    # ── Información del modelo ───────────────────────────────────

    def contar_parametros(self) -> dict:
        """Cuenta y clasifica los parámetros del modelo."""
        total_backbone = sum(p.numel() for p in self.backbone.parameters())
        total_cabeza = sum(p.numel() for p in self.cabeza.parameters())
        entrenables = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {
            "backbone": total_backbone,
            "cabeza": total_cabeza,
            "total": total_backbone + total_cabeza,
            "entrenables": entrenables,
            "congelados": (total_backbone + total_cabeza) - entrenables,
        }

    def resumen_modelo(self) -> str:
        """Devuelve un resumen del modelo con conteo de parámetros."""
        params = self.contar_parametros()
        lineas = [
            "┌──────────────────────────────────────────────────┐",
            "│   RESUMEN DEL CLASIFICADOR DE PATRONES            │",
            "├──────────────────────────────────────────────────┤",
            f"│ Backbone:     {params['backbone']:>12,} parámetros (congelados)",
            f"│ Cabeza:       {params['cabeza']:>12,} parámetros (entrenables)",
            f"│ Total:        {params['total']:>12,} parámetros",
            f"│ Entrenables:  {params['entrenables']:>12,}",
            f"│ Congelados:   {params['congelados']:>12,}",
            "└──────────────────────────────────────────────────┘",
        ]
        return "\n".join(lineas)
