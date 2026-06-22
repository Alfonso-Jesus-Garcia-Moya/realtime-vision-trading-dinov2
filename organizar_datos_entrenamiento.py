"""
Script para organizar las imágenes de PARA ENTRENAR/ en la estructura
de carpetas que necesita el entrenador:
    datos/patrones/Nombre_Clase/imagen.png

Mapea cada archivo PNG a su clase correspondiente del MAPA_CLASES_PATRONES.
"""

import os
import shutil
from pathlib import Path

# ── Directorio raíz del proyecto ─────────────────────────────────
RAIZ = os.path.dirname(os.path.abspath(__file__))
ORIGEN = os.path.join(RAIZ, "PARA ENTRENAR")
DESTINO = os.path.join(RAIZ, "datos", "patrones")

# ── Mapeo de archivos a clases ───────────────────────────────────
# Cada archivo se mapea a la clase del MAPA_CLASES_PATRONES
MAPEO_ARCHIVO_CLASE = {
    "GRAFICO DE UN CANAL ALCISTA.png": "Canal_Alcista",
    "GRAFICO DE UN CANAL BAJISTA.png": "Canal_Bajista",
    "GRAFICO DE UN CANAL LATERAL.png": "Canal_Lateral",
    "GRAFICO DE UN SOPORTE.png": "Soporte",
    "GRAFICO DE UNA RESISTENCIA.png": "Resistencia",
    "GRAFICO DE H-C-H ASCENDENTE.png": "HCH_Ascendente",
    "GRAFICO DE H-C-H DESENDENTE.png": "HCH_Descendente",
    "GRAFICO DE DOBLE SUELO.png": "Doble_Suelo",
    "GRAFICO DE UN DOBLE TECHO.png": "Doble_Techo",
    "GRAFICO DE FOMRACION V.png": "Formacion_V",
    "GRAFICO DE SUELO REDONDEADO.png": "Suelo_Redondeado",
    "GRAFICO DE TRIANGULO RECTO ALCISTA.png": "Triangulo_Recto_Alcista",
    "GRAFICO DE TRIGUANGULO RECTO BAJISTA.png": "Triangulo_Recto_Bajista",
    u"Gr\u00e1fico de un tri\u00e1ngulo sim\u00e9trico ascendente.png": "Triangulo_Simetrico_Ascendente",
    u"Gr\u00e1fico de un tri\u00e1ngulo sim\u00e9trico descendente.png": "Triangulo_Simetrico_Descendente",
    u"Gr\u00e1fico de una bandera.png": "Bandera",
    u"Gr\u00e1fico de una banderola.png": "Banderola",
    u"Gr\u00e1fico de una cu\u00f1a.png": "Cuna",
    u"Gr\u00e1fico de una divergencia alcista en el oscilador MACD.png": "Divergencia_MACD",
    u"Gr\u00e1fico de una divergencia alcista en el oscilador RSI.png": "Divergencia_RSI",
    u"Gr\u00e1fico de una media m\u00f3vil exponencial de 200 periodos.png": "Media_Movil_EMA200",
    u"Gr\u00e1fico de una media m\u00f3vil exponencial de 50 periodos.png": "Media_Movil_EMA50",
    u"Gr\u00e1fico de una media m\u00f3vil ponderada.png": "Media_Movil_Ponderada",
    u"Gr\u00e1fico de una media m\u00f3vil simple.png": "Media_Movil_Simple",
    u"Gr\u00e1fico del oscilador %K.png": "Oscilador_K",
    u"Gr\u00e1fico del oscilador de momento.png": "Oscilador_Momento",
    u"Gr\u00e1fico del oscilador de Williams .png": "Oscilador_Williams",
    u"Gr\u00e1fico del oscilador ROC o tasa de cambio.png": "Oscilador_ROC",
    u"Formaci\u00f3n de Taza (Cup  Rounding Bottom) con ruptura parab\u00f3lica.png": "Suelo_Redondeado",
    u"Patr\u00f3n Principal Reversi\u00f3n en V (V-Bottom  V-Shape Reversal.png": "Formacion_V",
    "Ruptura de resistencia macro seguida de un Throwback (retesteo).png": "Resistencia",
}


def organizar():
    """Crea la estructura de carpetas y copia las imágenes."""
    print("=" * 60)
    print("  ORGANIZANDO DATOS DE ENTRENAMIENTO")
    print("=" * 60)

    if not os.path.exists(ORIGEN):
        print(f"✘ Carpeta origen no encontrada: {ORIGEN}")
        return

    # Crear directorio destino
    os.makedirs(DESTINO, exist_ok=True)

    # También crear carpeta Sin_Patron (necesaria para el clasificador)
    sin_patron_dir = os.path.join(DESTINO, "Sin_Patron")
    os.makedirs(sin_patron_dir, exist_ok=True)

    archivos_procesados = 0
    archivos_no_mapeados = []

    for archivo in os.listdir(ORIGEN):
        if not archivo.lower().endswith(".png"):
            continue

        clase = MAPEO_ARCHIVO_CLASE.get(archivo)
        if clase is None:
            archivos_no_mapeados.append(archivo)
            continue

        # Crear subcarpeta de la clase
        carpeta_clase = os.path.join(DESTINO, clase)
        os.makedirs(carpeta_clase, exist_ok=True)

        # Copiar imagen
        origen_archivo = os.path.join(ORIGEN, archivo)
        destino_archivo = os.path.join(carpeta_clase, archivo)
        shutil.copy2(origen_archivo, destino_archivo)
        archivos_procesados += 1
        print(f"  ✔ {archivo}")
        print(f"    → {clase}/")

    # Generar imágenes sintéticas "Sin_Patron" a partir de capturas vacías
    # (placeholder: copiar una imagen y marcarla como Sin_Patron)
    print(f"\n{'=' * 60}")
    print(f"  Archivos organizados: {archivos_procesados}")
    print(f"  Clases creadas: {len(set(MAPEO_ARCHIVO_CLASE.values()))}")

    if archivos_no_mapeados:
        print(f"\n  ⚠ Archivos sin mapear ({len(archivos_no_mapeados)}):")
        for a in archivos_no_mapeados:
            print(f"    - {a}")

    # Listar estructura final
    print(f"\n  Estructura final en {DESTINO}:")
    for carpeta in sorted(os.listdir(DESTINO)):
        ruta = os.path.join(DESTINO, carpeta)
        if os.path.isdir(ruta):
            n = len([f for f in os.listdir(ruta) if f.endswith(".png")])
            print(f"    📁 {carpeta}/ ({n} imágenes)")

    print("=" * 60)


if __name__ == "__main__":
    organizar()
