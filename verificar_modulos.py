"""Script de verificación de todos los módulos del sistema."""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  VERIFICACIÓN DE MÓDULOS DEL SISTEMA")
print("=" * 60)

# 1. Verificar configuracion_vision
try:
    from detector_patrones.configuracion_vision import (
        ConfiguracionVision, MAPA_CLASES_PATRONES, NUMERO_CLASES
    )
    config = ConfiguracionVision()
    print(f"\n[OK] configuracion_vision.py")
    print(f"     Clases definidas: {NUMERO_CLASES}")
    print(f"     Modelo base: {config.nombre_modelo_base}")
    print(f"     VRAM max: {config.fraccion_vram_maxima:.0%}")
    print(config.resumen())
except Exception as e:
    print(f"\n[ERROR] configuracion_vision.py: {e}")

# 2. Verificar modelo_vision (sin cargar DINOv2)
try:
    from detector_patrones.modelo_vision import (
        ClasificadorPatronesVisuales,
        CabezaClasificacionMLP,
        CabezaClasificacionLineal,
    )
    print(f"\n[OK] modelo_vision.py - Clases importadas correctamente")
    print(f"     ClasificadorPatronesVisuales: OK")
    print(f"     CabezaClasificacionMLP: OK")
    print(f"     CabezaClasificacionLineal: OK")
except Exception as e:
    print(f"\n[ERROR] modelo_vision.py: {e}")

# 3. Verificar procesador_vision
try:
    from detector_patrones.procesador_vision import (
        ProcesadorVision, ResultadoDeteccion
    )
    print(f"\n[OK] procesador_vision.py - Clases importadas correctamente")
    print(f"     ProcesadorVision: OK")
    print(f"     ResultadoDeteccion: OK")
except Exception as e:
    print(f"\n[ERROR] procesador_vision.py: {e}")

# 4. Verificar entrenador_patrones
try:
    from detector_patrones.entrenador_patrones import (
        EntrenadorPatrones, ParadaTemprana
    )
    print(f"\n[OK] entrenador_patrones.py - Clases importadas correctamente")
    print(f"     EntrenadorPatrones: OK")
    print(f"     ParadaTemprana: OK")
except Exception as e:
    print(f"\n[ERROR] entrenador_patrones.py: {e}")

# 5. Verificar __init__.py
try:
    from detector_patrones import (
        ConfiguracionVision, MAPA_CLASES_PATRONES,
        ClasificadorPatronesVisuales, ProcesadorVision
    )
    print(f"\n[OK] __init__.py - Exports correctos")
except Exception as e:
    print(f"\n[ERROR] __init__.py: {e}")

# 6. Verificar torch y CUDA
import torch
print(f"\n[INFO] PyTorch version: {torch.__version__}")
print(f"[INFO] CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"[INFO] VRAM total: {vram:.1f} GB")

# 7. Verificar torchvision
import torchvision
print(f"[INFO] TorchVision version: {torchvision.__version__}")

# 8. Verificar principal.py puede parsear argumentos
print(f"\n[INFO] Verificando principal.py...")
try:
    from principal import parsear_argumentos, inicializar_procesador_vision
    print(f"[OK] principal.py - Funciones importadas correctamente")
except Exception as e:
    print(f"[ERROR] principal.py: {e}")

print("\n" + "=" * 60)
print("  VERIFICACIÓN COMPLETADA")
print("=" * 60)
