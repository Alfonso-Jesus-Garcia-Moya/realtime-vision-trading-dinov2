"""
Script Organizador de Datos de Entrenamiento.
Lee las imágenes PNG de la carpeta 'PARA ENTRENAR/' y las copia
en la estructura de carpetas que espera el entrenador:
    datos/patrones/<nombre_clase>/imagen.png

Además, genera copias aumentadas (data augmentation agresiva)
para compensar que solo hay 1 imagen por clase.
Crea ~50 variantes por imagen original para un dataset efectivo.
"""

import os
import sys
import shutil
import logging
from typing import Dict, List, Tuple

from PIL import Image
from torchvision import transforms

# ── Logger ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
registrador = logging.getLogger("organizar_datos")


# ══════════════════════════════════════════════════════════════════
#  Mapeo: nombre de archivo PNG → nombre de clase
# ══════════════════════════════════════════════════════════════════

MAPEO_ARCHIVOS_CLASES: Dict[str, str] = {
    "GRAFICO DE UN CANAL ALCISTA.png":                          "Canal_Alcista",
    "GRAFICO DE UN CANAL BAJISTA.png":                          "Canal_Bajista",
    "GRAFICO DE UN CANAL LATERAL.png":                          "Canal_Lateral",
    "GRAFICO DE UN DOBLE TECHO.png":                            "Doble_Techo",
    "GRAFICO DE DOBLE SUELO.png":                               "Doble_Suelo",
    "GRAFICO DE FOMRACION V.png":                               "Formacion_V",
    "GRAFICO DE H-C-H ASCENDENTE.png":                          "HCH_Ascendente",
    "GRAFICO DE H-C-H DESENDENTE.png":                          "HCH_Descendente",
    "GRAFICO DE SUELO REDONDEADO.png":                          "Suelo_Redondeado",
    "GRAFICO DE TRIANGULO RECTO ALCISTA.png":                   "Triangulo_Alcista",
    "GRAFICO DE TRIGUANGULO RECTO BAJISTA.png":                 "Triangulo_Bajista",
    "Gráfico de un triángulo simétrico ascendente.png":         "Triangulo_Simetrico_Alcista",
    "Gráfico de un triángulo simétrico descendente.png":        "Triangulo_Simetrico_Bajista",
    "GRAFICO DE UN SOPORTE.png":                                "Soporte",
    "GRAFICO DE UNA RESISTENCIA.png":                           "Resistencia",
    "Gráfico de una bandera.png":                               "Bandera",
    "Gráfico de una banderola.png":                             "Banderola",
    "Gráfico de una cuña.png":                                  "Cuna",
    "Gráfico de una divergencia alcista en el oscilador MACD.png":  "Divergencia_MACD",
    "Gráfico de una divergencia alcista en el oscilador RSI.png":   "Divergencia_RSI",
    "Gráfico de una media móvil exponencial de 50 periodos.png":    "Media_Movil_EMA50",
    "Gráfico de una media móvil exponencial de 200 periodos.png":   "Media_Movil_EMA200",
    "Gráfico de una media móvil ponderada.png":                     "Media_Movil_Ponderada",
    "Gráfico de una media móvil simple.png":                        "Media_Movil_Simple",
    "Gráfico del oscilador %K.png":                                 "Oscilador_K",
    "Gráfico del oscilador de momento.png":                         "Oscilador_Momento",
    "Gráfico del oscilador de Williams .png":                       "Oscilador_Williams",
    "Gráfico del oscilador ROC o tasa de cambio.png":               "Oscilador_ROC",
}


# ══════════════════════════════════════════════════════════════════
#  Transformaciones de augmentación agresiva
# ══════════════════════════════════════════════════════════════════

VARIANTES_POR_IMAGEN = 50  # Generar 50 variantes por cada imagen original

def crear_aumentador() -> transforms.Compose:
    """Crea transformaciones de augmentación agresiva para generar variantes."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])


# ══════════════════════════════════════════════════════════════════
#  Funciones principales
# ══════════════════════════════════════════════════════════════════

def organizar_y_aumentar(
    ruta_origen: str = "PARA ENTRENAR",
    ruta_destino: str = "datos/patrones",
    variantes: int = VARIANTES_POR_IMAGEN,
) -> None:
    """
    1. Lee cada PNG de ruta_origen.
    2. Lo mapea a su clase según MAPEO_ARCHIVOS_CLASES.
    3. Crea la carpeta de clase en ruta_destino.
    4. Copia el original y genera N variantes aumentadas.
    """
    if not os.path.exists(ruta_origen):
        registrador.error("Carpeta origen no existe: %s", ruta_origen)
        sys.exit(1)

    # Limpiar destino previo si existe
    if os.path.exists(ruta_destino):
        registrador.info("Limpiando directorio previo: %s", ruta_destino)
        shutil.rmtree(ruta_destino)

    os.makedirs(ruta_destino, exist_ok=True)

    aumentador = crear_aumentador()
    total_imagenes = 0
    total_variantes = 0
    clases_creadas = set()

    registrador.info("═" * 60)
    registrador.info("  ORGANIZANDO DATOS DE ENTRENAMIENTO")
    registrador.info("  Origen: %s", ruta_origen)
    registrador.info("  Destino: %s", ruta_destino)
    registrador.info("  Variantes por imagen: %d", variantes)
    registrador.info("═" * 60)

    for nombre_archivo, nombre_clase in MAPEO_ARCHIVOS_CLASES.items():
        ruta_archivo = os.path.join(ruta_origen, nombre_archivo)

        if not os.path.exists(ruta_archivo):
            registrador.warning("  ⚠ No encontrado: %s", nombre_archivo)
            continue

        # Crear carpeta de clase
        carpeta_clase = os.path.join(ruta_destino, nombre_clase)
        os.makedirs(carpeta_clase, exist_ok=True)
        clases_creadas.add(nombre_clase)

        # Abrir imagen original
        try:
            imagen = Image.open(ruta_archivo).convert("RGB")
        except Exception as error:
            registrador.error("  ✘ Error al abrir %s: %s", nombre_archivo, error)
            continue

        # Copiar original (redimensionado a 224x224)
        imagen_redim = imagen.resize((224, 224), Image.Resampling.LANCZOS)
        ruta_copia = os.path.join(carpeta_clase, "original.png")
        imagen_redim.save(ruta_copia)
        total_imagenes += 1

        # Generar variantes aumentadas
        for i in range(variantes):
            try:
                variante = aumentador(imagen)
                # Asegurar tamaño 224x224
                if variante.size != (224, 224):
                    variante = variante.resize((224, 224), Image.Resampling.LANCZOS)
                ruta_variante = os.path.join(
                    carpeta_clase, f"variante_{i+1:03d}.png"
                )
                variante.save(ruta_variante)
                total_variantes += 1
            except Exception as error:
                registrador.warning(
                    "  ⚠ Error en variante %d de %s: %s",
                    i + 1, nombre_clase, error
                )

        registrador.info(
            "  ✔ %s → %d variantes + 1 original",
            nombre_clase, variantes
        )

    # ── Resumen ───────────────────────────────────────────────────
    registrador.info("═" * 60)
    registrador.info("  RESUMEN DE ORGANIZACIÓN")
    registrador.info("═" * 60)
    registrador.info("  Clases creadas:     %d", len(clases_creadas))
    registrador.info("  Imágenes originales: %d", total_imagenes)
    registrador.info("  Variantes generadas: %d", total_variantes)
    registrador.info("  Total imágenes:      %d", total_imagenes + total_variantes)
    registrador.info("  Directorio:          %s", os.path.abspath(ruta_destino))

    # Listar clases creadas
    registrador.info("\n  Clases:")
    for idx, clase in enumerate(sorted(clases_creadas)):
        carpeta = os.path.join(ruta_destino, clase)
        n_imgs = len(os.listdir(carpeta))
        registrador.info("    %2d. %-35s (%d imágenes)", idx, clase, n_imgs)

    registrador.info("\n  ✔ Datos listos para entrenamiento.")
    registrador.info("  Ejecutar: python -m detector_patrones.entrenador_patrones")


# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Organizador de datos de entrenamiento")
    parser.add_argument("--origen", default="PARA ENTRENAR", help="Carpeta con PNGs")
    parser.add_argument("--destino", default="datos/patrones", help="Carpeta destino")
    parser.add_argument("--variantes", type=int, default=50, help="Variantes por imagen")
    args = parser.parse_args()

    organizar_y_aumentar(args.origen, args.destino, args.variantes)
