"""
Módulo de Configuración del Sistema de Visión por Computadora.
Define hiperparámetros del modelo DINOv2, mapeo de clases de patrones
técnicos y límites estrictos de hardware (VRAM) para ejecución local
segura en la RTX 5070 Ti.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List


# ══════════════════════════════════════════════════════════════════
#  Mapeo de Clases de Patrones Técnicos (29 clases)
#  Orden alfabético = ImageFolder de PyTorch
#  Basado en las carpetas de datos/patrones/
# ══════════════════════════════════════════════════════════════════

MAPA_CLASES_PATRONES: Dict[int, str] = {
    0:  "Bandera",
    1:  "Banderola",
    2:  "Canal_Alcista",
    3:  "Canal_Bajista",
    4:  "Canal_Lateral",
    5:  "Cuna",
    6:  "Divergencia_MACD",
    7:  "Divergencia_RSI",
    8:  "Doble_Suelo",
    9:  "Doble_Techo",
    10: "Formacion_V",
    11: "HCH_Ascendente",
    12: "HCH_Descendente",
    13: "Media_Movil_EMA200",
    14: "Media_Movil_EMA50",
    15: "Media_Movil_Ponderada",
    16: "Media_Movil_Simple",
    17: "Oscilador_K",
    18: "Oscilador_Momento",
    19: "Oscilador_ROC",
    20: "Oscilador_Williams",
    21: "Resistencia",
    22: "Sin_Patron",
    23: "Soporte",
    24: "Suelo_Redondeado",
    25: "Triangulo_Recto_Alcista",
    26: "Triangulo_Recto_Bajista",
    27: "Triangulo_Simetrico_Ascendente",
    28: "Triangulo_Simetrico_Descendente",
}

NUMERO_CLASES: int = len(MAPA_CLASES_PATRONES)

# Mapeo inverso: nombre → índice
MAPA_NOMBRE_A_INDICE: Dict[str, int] = {
    nombre: indice for indice, nombre in MAPA_CLASES_PATRONES.items()
}


# ══════════════════════════════════════════════════════════════════
#  Configuración del Modelo de Visión
# ══════════════════════════════════════════════════════════════════

@dataclass
class ConfiguracionVision:
    """
    Hiperparámetros y configuración del clasificador visual basado
    en DINOv2 + cabeza de clasificación lineal/MLP.
    """

    # ── Backbone DINOv2 ──────────────────────────────────────────
    nombre_modelo_base: str = "dinov2_vitb14"
    repositorio_torch_hub: str = "facebookresearch/dinov2"
    dimension_embedding: int = 768          # vitb14 produce embeddings de 768-d
    congelar_backbone: bool = True          # SIEMPRE congelar para ahorro de VRAM

    # ── Cabeza de Clasificación (MLP ligero) ─────────────────────
    numero_clases: int = NUMERO_CLASES
    dimension_oculta_mlp: int = 256         # Capa oculta del MLP
    tasa_dropout: float = 0.3               # Dropout para regularización
    usar_mlp: bool = True                   # True=MLP, False=Linear simple

    # ── Entrada ──────────────────────────────────────────────────
    tamano_entrada: Tuple[int, int] = (224, 224)   # (Alto, Ancho) — DINOv2 espera 224×224
    canales_entrada: int = 3

    # ── Normalización ImageNet (requerida por DINOv2) ────────────
    media_imagenet: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    desviacion_imagenet: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ── Hardware — Límites estrictos de VRAM ─────────────────────
    fraccion_vram_maxima: float = 0.40      # Máx 40% de VRAM para este proceso
    dispositivo_preferido: str = "cuda"     # "cuda" si disponible, fallback a "cpu"

    # ── Umbrales de Inferencia ───────────────────────────────────
    umbral_confianza_minima: float = 0.70   # Debajo de esto → "Sin Patrón Claro"
    indice_clase_sin_patron: int = -1       # Índice ficticio para "Sin Patrón"

    # ── Entrenamiento de la Cabeza ───────────────────────────────
    tasa_aprendizaje: float = 1e-3
    peso_decaimiento: float = 1e-4          # Weight decay para AdamW
    epocas_maximas: int = 50
    paciencia_early_stopping: int = 7       # Épocas sin mejora antes de parar
    tamano_lote_entrenamiento: int = 32
    fraccion_validacion: float = 0.2        # 20% para validación
    semilla_aleatoria: int = 42

    # ── Rutas de archivos ────────────────────────────────────────
    ruta_pesos_cabeza: str = "modelos_guardados/cabeza_patrones.pth"
    ruta_datos_entrenamiento: str = "datos/patrones"

    # ── Métodos utilitarios ──────────────────────────────────────

    def resumen(self) -> str:
        """Devuelve un resumen legible de la configuración de visión."""
        lineas = [
            "╔══════════════════════════════════════════════════╗",
            "║   CONFIGURACIÓN DEL DETECTOR DE PATRONES         ║",
            "╠══════════════════════════════════════════════════╣",
            f"║ Modelo base:       {self.nombre_modelo_base}",
            f"║ Embedding dim:     {self.dimension_embedding}",
            f"║ Backbone congelado: {self.congelar_backbone}",
            f"║ Clases:            {self.numero_clases}",
            f"║ MLP oculta:        {self.dimension_oculta_mlp}",
            f"║ Dropout:           {self.tasa_dropout}",
            f"║ Fracción VRAM máx: {self.fraccion_vram_maxima:.0%}",
            f"║ Dispositivo pref.: {self.dispositivo_preferido}",
            f"║ Umbral confianza:  {self.umbral_confianza_minima:.0%}",
            f"║ Pesos guardados:   {self.ruta_pesos_cabeza}",
            "╚══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lineas)
