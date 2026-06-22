"""
Script de Diagnóstico — Detecta los selectores CSS reales del gráfico
en Yahoo Finance y guarda un screenshot de referencia.
"""

import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ServicioCromo
from selenium.webdriver.chrome.options import Options as OpcionesCromo
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


def diagnosticar():
    print("═" * 60)
    print("  DIAGNÓSTICO DE SELECTORES — Yahoo Finance Chart")
    print("═" * 60)

    opciones = OpcionesCromo()
    opciones.add_argument("--headless=new")
    opciones.add_argument("--no-sandbox")
    opciones.add_argument("--disable-dev-shm-usage")
    opciones.add_argument("--disable-gpu")
    opciones.add_argument("--window-size=1920,1080")
    opciones.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )

    servicio = ServicioCromo(ChromeDriverManager().install())
    navegador = webdriver.Chrome(service=servicio, options=opciones)
    navegador.set_window_size(1920, 1080)

    url = "https://finance.yahoo.com/chart/%5EGSPC"
    print(f"\n1. Navegando a: {url}")
    navegador.get(url)

    print("2. Esperando 8 segundos para que el JS renderice…")
    time.sleep(8)

    # Guardar screenshot de referencia
    print("3. Guardando screenshot de referencia…")
    navegador.save_screenshot("diagnostico_screenshot.png")
    print("   → diagnostico_screenshot.png guardado")

    # Obtener título de la página
    print(f"\n4. Título de la página: {navegador.title}")
    print(f"   URL actual: {navegador.current_url}")

    # Buscar todos los canvas
    print("\n5. Buscando elementos <canvas>…")
    canvases = navegador.find_elements(By.TAG_NAME, "canvas")
    print(f"   → Encontrados: {len(canvases)} canvas")
    for i, c in enumerate(canvases):
        tamano = c.size
        clase = c.get_attribute("class") or "(sin clase)"
        id_elem = c.get_attribute("id") or "(sin id)"
        print(f"   Canvas #{i}: clase='{clase}' id='{id_elem}' "
              f"tamaño={tamano['width']}x{tamano['height']}")

    # Buscar selectores candidatos para el chart
    print("\n6. Probando selectores candidatos…")
    selectores_prueba = [
        "canvas.chartiq-chart",
        "[data-testid='chart-container']",
        "[data-testid='chart-container'] canvas",
        ".chart-container",
        ".chart-container canvas",
        "#chartiq-chart-container",
        "#chartiq-chart-container canvas",
        ".chartContainer",
        ".chartContainer canvas",
        "[data-testid='fullscreen-chart']",
        "[data-testid='fullscreen-chart'] canvas",
        ".main-chart",
        ".YDC-Lead",
        "#chart-container",
        "#lead-1-QuoteChart",
        "[data-testid='qsp-chart']",
        "[data-testid='qsp-chart'] canvas",
        ".chartRoot",
        ".chartRoot canvas",
        "div[class*='chart'] canvas",
        "div[class*='Chart'] canvas",
        "div[id*='chart'] canvas",
        "div[id*='Chart'] canvas",
        "section canvas",
        "main canvas",
        "canvas",
    ]

    encontrados = []
    for selector in selectores_prueba:
        try:
            elementos = navegador.find_elements(By.CSS_SELECTOR, selector)
            if elementos:
                for elem in elementos:
                    tamano = elem.size
                    if tamano["width"] > 0 and tamano["height"] > 0:
                        encontrados.append({
                            "selector": selector,
                            "tag": elem.tag_name,
                            "clase": elem.get_attribute("class") or "",
                            "id": elem.get_attribute("id") or "",
                            "ancho": tamano["width"],
                            "alto": tamano["height"],
                        })
                        print(f"   ✔ '{selector}' → {elem.tag_name} "
                              f"({tamano['width']}x{tamano['height']})")
        except Exception:
            pass

    # Buscar con JavaScript los contenedores del chart
    print("\n7. Explorando DOM con JavaScript…")
    js_resultado = navegador.execute_script("""
        const resultado = [];
        // Buscar divs grandes que podrían contener el chart
        const divs = document.querySelectorAll('div, section, main');
        for (const div of divs) {
            const rect = div.getBoundingClientRect();
            if (rect.width > 400 && rect.height > 200) {
                const canvases = div.querySelectorAll('canvas');
                if (canvases.length > 0) {
                    resultado.push({
                        tag: div.tagName,
                        clase: div.className.substring(0, 80),
                        id: div.id,
                        ancho: Math.round(rect.width),
                        alto: Math.round(rect.height),
                        num_canvas: canvases.length,
                        data_testid: div.getAttribute('data-testid') || ''
                    });
                }
            }
        }
        return resultado;
    """)

    print(f"   → Contenedores con canvas encontrados: {len(js_resultado)}")
    for i, r in enumerate(js_resultado[:10]):
        print(f"   [{i}] <{r['tag']}> clase='{r['clase'][:50]}' "
              f"id='{r['id']}' data-testid='{r['data_testid']}' "
              f"tamaño={r['ancho']}x{r['alto']} "
              f"canvas_dentro={r['num_canvas']}")

    # Guardar resultados
    with open("diagnostico_resultados.json", "w", encoding="utf-8") as f:
        json.dump({
            "titulo": navegador.title,
            "url": navegador.current_url,
            "num_canvas": len(canvases),
            "selectores_validos": encontrados,
            "contenedores_js": js_resultado,
        }, f, indent=2, ensure_ascii=False)
    print("\n   → diagnostico_resultados.json guardado")

    navegador.quit()
    print("\n" + "═" * 60)
    print("  DIAGNÓSTICO COMPLETADO")
    print("═" * 60)


if __name__ == "__main__":
    diagnosticar()
