"""
╔══════════════════════════════════════════════════════════════╗
║  INICIAR.PY — PUNTO DE ENTRADA ÚNICO DEL SISTEMA           ║
║                                                              ║
║  Ejecuta:  python iniciar.py                                 ║
║                                                              ║
║  Esto arranca TODO:                                          ║
║    1. Servidor Flask (interfaz web)                          ║
║    2. Motor de captura de Yahoo Finance (Chrome headless)    ║
║    3. Tubería de tensores en memoria                         ║
║    4. Detector de patrones DINOv2                            ║
║    5. Abre http://localhost:5000 en tu navegador             ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import webbrowser
import threading

# Asegurar que la raíz del proyecto esté en sys.path
DIRECTORIO_RAIZ = os.path.dirname(os.path.abspath(__file__))
if DIRECTORIO_RAIZ not in sys.path:
    sys.path.insert(0, DIRECTORIO_RAIZ)


def abrir_navegador_con_retardo(url: str, retardo_seg: float = 4.0):
    """Abre el navegador después de un retardo para que Flask esté listo."""
    time.sleep(retardo_seg)
    print(f"\n🌐 Abriendo {url} en tu navegador...\n")
    webbrowser.open(url)


def main():
    """Punto de entrada único — arranca todo el sistema."""
    puerto = 5000
    intervalo = 10.0
    url_chart = "https://finance.yahoo.com/chart/%5EGSPC"

    # Si hay argumentos CLI simples, parsearlos
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--puerto" and i + 1 < len(args):
            puerto = int(args[i + 1])
        elif arg == "--intervalo" and i + 1 < len(args):
            intervalo = float(args[i + 1])
        elif arg == "--url" and i + 1 < len(args):
            url_chart = args[i + 1]

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     SISTEMA DE DETECCIÓN DE PATRONES EN TIEMPO REAL     ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Puerto:    {puerto}                                        ║")
    print(f"║  Intervalo: {intervalo}s                                       ║")
    print(f"║  URL:       {url_chart[:45]}...  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Abrirá http://localhost:5000 automáticamente           ║")
    print("║  Presiona Ctrl+C para detener                           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Abrir navegador automáticamente después de 4 segundos
    hilo_navegador = threading.Thread(
        target=abrir_navegador_con_retardo,
        args=(f"http://localhost:{puerto}", 4.0),
        daemon=True,
    )
    hilo_navegador.start()

    # Importar e iniciar el servidor (esto bloquea)
    from interfaz_web.servidor import iniciar_servidor
    iniciar_servidor(puerto=puerto, intervalo=intervalo, url=url_chart)


if __name__ == "__main__":
    main()
