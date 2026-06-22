"""
Script de Aumentación de Datos para el Entrenador de Patrones.
Genera múltiples variaciones de cada imagen para tener suficientes
muestras por clase para el entrenamiento.

Cada imagen original produce ~15 variaciones con:
- Rotaciones, flips, recortes, cambios de color, zoom, etc.
"""

import os
import random
from PIL import Image, ImageEnhance, ImageFilter

RAIZ = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(RAIZ, "datos", "patrones")
COPIAS_POR_IMAGEN = 15


def aumentar_imagen(ruta_imagen: str, carpeta_destino: str, indice_base: int):
    """Genera variaciones aumentadas de una imagen."""
    try:
        img_original = Image.open(ruta_imagen).convert("RGB")
    except Exception as e:
        print(f"  ✘ Error abriendo {ruta_imagen}: {e}")
        return 0

    nombre_base = os.path.splitext(os.path.basename(ruta_imagen))[0]
    generadas = 0

    for i in range(COPIAS_POR_IMAGEN):
        img = img_original.copy()

        # Rotación aleatoria (-15 a +15 grados)
        if random.random() > 0.3:
            angulo = random.uniform(-15, 15)
            img = img.rotate(angulo, expand=False, fillcolor=(255, 255, 255))

        # Flip horizontal
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Cambio de brillo
        if random.random() > 0.4:
            factor = random.uniform(0.7, 1.3)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Cambio de contraste
        if random.random() > 0.4:
            factor = random.uniform(0.7, 1.4)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # Cambio de saturación
        if random.random() > 0.5:
            factor = random.uniform(0.6, 1.5)
            img = ImageEnhance.Color(img).enhance(factor)

        # Cambio de nitidez
        if random.random() > 0.6:
            factor = random.uniform(0.5, 2.0)
            img = ImageEnhance.Sharpness(img).enhance(factor)

        # Recorte aleatorio (zoom)
        if random.random() > 0.4:
            w, h = img.size
            margen_x = int(w * random.uniform(0.02, 0.12))
            margen_y = int(h * random.uniform(0.02, 0.12))
            caja = (margen_x, margen_y, w - margen_x, h - margen_y)
            img = img.crop(caja).resize((w, h), Image.LANCZOS)

        # Blur ligero
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

        # Guardar variación
        nombre_salida = f"{nombre_base}_aug{indice_base + i:03d}.png"
        ruta_salida = os.path.join(carpeta_destino, nombre_salida)
        img.save(ruta_salida, "PNG")
        generadas += 1

    return generadas


def generar_imagenes_sin_patron(carpeta_destino: str, cantidad: int = 20):
    """
    Genera imágenes 'Sin Patron' usando ruido y gradientes aleatorios.
    Simula fondos de gráficos sin patrones técnicos claros.
    """
    os.makedirs(carpeta_destino, exist_ok=True)

    for i in range(cantidad):
        ancho, alto = 800, 600
        img = Image.new("RGB", (ancho, alto), (255, 255, 255))
        pixels = img.load()

        # Fondo con gradiente aleatorio
        color_base = (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255),
        )

        for y in range(alto):
            for x in range(ancho):
                ruido = random.randint(-15, 15)
                r = max(0, min(255, color_base[0] + ruido + int((y / alto) * 30)))
                g = max(0, min(255, color_base[1] + ruido + int((x / ancho) * 20)))
                b = max(0, min(255, color_base[2] + ruido))
                pixels[x, y] = (r, g, b)

        # Guardar
        nombre = f"sin_patron_{i:03d}.png"
        img.save(os.path.join(carpeta_destino, nombre), "PNG")

    print(f"  ✔ Generadas {cantidad} imágenes Sin_Patron")
    return cantidad


def main():
    print("=" * 60)
    print("  AUMENTACIÓN DE DATOS DE ENTRENAMIENTO")
    print(f"  {COPIAS_POR_IMAGEN} variaciones por imagen original")
    print("=" * 60)

    if not os.path.exists(DATOS):
        print(f"✘ No se encontró {DATOS}. Ejecuta primero organizar_datos_entrenamiento.py")
        return

    total_generadas = 0

    for clase in sorted(os.listdir(DATOS)):
        carpeta_clase = os.path.join(DATOS, clase)
        if not os.path.isdir(carpeta_clase):
            continue

        if clase == "Sin_Patron":
            n = generar_imagenes_sin_patron(carpeta_clase, cantidad=20)
            total_generadas += n
            continue

        imagenes = [f for f in os.listdir(carpeta_clase) if f.lower().endswith(".png") and "_aug" not in f]
        if not imagenes:
            print(f"  ⚠ {clase}/ — sin imágenes originales")
            continue

        print(f"\n  📁 {clase}/ ({len(imagenes)} originales)")
        idx = 0
        for img_nombre in imagenes:
            ruta = os.path.join(carpeta_clase, img_nombre)
            n = aumentar_imagen(ruta, carpeta_clase, idx)
            total_generadas += n
            idx += n
            print(f"    ✔ {img_nombre} → {n} variaciones")

    # Resumen final
    print(f"\n{'=' * 60}")
    print(f"  TOTAL IMÁGENES GENERADAS: {total_generadas}")
    print(f"\n  Estructura final:")
    for clase in sorted(os.listdir(DATOS)):
        carpeta = os.path.join(DATOS, clase)
        if os.path.isdir(carpeta):
            total = len([f for f in os.listdir(carpeta) if f.endswith(".png")])
            print(f"    📁 {clase}/ — {total} imágenes")
    print("=" * 60)


if __name__ == "__main__":
    main()
