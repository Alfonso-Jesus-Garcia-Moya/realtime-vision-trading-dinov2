# 📊 Sistema de Detección de Patrones Técnicos en Tiempo Real

## ¿Qué hace este sistema?

Captura screenshots del chart de Yahoo Finance (S&P 500 ^GSPC) cada 10 segundos, los convierte a tensores en memoria (sin guardar archivos) y usa un modelo DINOv2 + cabeza MLP para detectar **28 patrones técnicos** del análisis bursátil. Todo se visualiza en un dashboard web local en tiempo real.

---

## 🏗️ Arquitectura

```
Yahoo Finance (Chrome) → Screenshot cada 10s → Tensor en memoria → DINOv2 → Patrón detectado
                                                                      ↓
                                                              Dashboard Web (Flask)
                                                              http://localhost:5000
```

**Módulos:**
| Módulo | Descripción |
|--------|-------------|
| `captura_datos/` | Selenium screenshot → tensor PyTorch en memoria |
| `detector_patrones/` | DINOv2 backbone + MLP head → 28 clases de patrones |
| `interfaz_web/` | Dashboard Flask con polling cada 2s |

---

## 🚀 Guía Rápida (3 pasos)

### Paso 1: Organizar datos de entrenamiento

```bash
python organizar_datos.py
```

Esto lee las 28 imágenes PNG de `PARA ENTRENAR/`, las mapea a clases, y genera **50 variantes aumentadas** por cada una (total: ~1,428 imágenes en `datos/patrones/`).

### Paso 2: Entrenar la cabeza del modelo

```bash
python -m detector_patrones.entrenador_patrones
```

Entrena SOLO la cabeza MLP (~200K parámetros) sobre el backbone DINOv2 congelado. Duración estimada: 2-5 minutos en GPU, 10-15 min en CPU. Los pesos se guardan en `modelos_guardados/cabeza_patrones.pth`.

### Paso 3: Lanzar la interfaz web

```bash
python -m interfaz_web.servidor
```

Abre **http://localhost:5000** en tu navegador. Verás:
- 📸 **Screenshot en vivo** del chart S&P 500
- 🎯 **Patrón detectado** con porcentaje de confianza
- 📋 **Historial** de todas las detecciones
- 📈 **Métricas** (capturas totales, patrones detectados, FPS, dispositivo)

---

## 📂 Estructura del Proyecto

```
├── captura_datos/                 # Sprint 1: Captura de pantalla
│   ├── configuracion.py          # URLs, selectores, intervalos
│   ├── capturador_pantalla.py    # Selenium headless Chrome
│   └── tuberia_tensores.py       # Bytes → Tensor [3, 224, 224]
│
├── detector_patrones/             # Sprint 2: Visión por computadora
│   ├── configuracion_vision.py   # 28 clases, hiperparámetros
│   ├── modelo_vision.py          # DINOv2 + MLP head
│   ├── entrenador_patrones.py    # Entrenamiento de la cabeza
│   └── procesador_vision.py      # Inferencia con gestión VRAM
│
├── interfaz_web/                  # Sprint 3: Visualización
│   ├── servidor.py               # Flask + hilo de captura
│   └── plantillas/panel.html     # Dashboard dark-mode
│
├── PARA ENTRENAR/                 # Imágenes PNG de patrones
├── datos/patrones/                # (generado) Dataset organizado
├── modelos_guardados/             # (generado) Pesos entrenados
│
├── organizar_datos.py             # Script de preparación de datos
├── principal.py                   # Orquestador CLI
├── verificar_modulos.py           # Verificación de imports
├── requirements.txt               # Dependencias
└── INSTRUCCIONES.md               # Este archivo
```

---

## 🎯 28 Patrones que Detecta

| # | Patrón | # | Patrón |
|---|--------|---|--------|
| 0 | Bandera | 14 | Media Móvil EMA50 |
| 1 | Banderola | 15 | Media Móvil Ponderada |
| 2 | Canal Alcista | 16 | Media Móvil Simple |
| 3 | Canal Bajista | 17 | Oscilador %K |
| 4 | Canal Lateral | 18 | Oscilador Momento |
| 5 | Cuña | 19 | Oscilador ROC |
| 6 | Divergencia MACD | 20 | Oscilador Williams |
| 7 | Divergencia RSI | 21 | Resistencia |
| 8 | Doble Suelo | 22 | Soporte |
| 9 | Doble Techo | 23 | Suelo Redondeado |
| 10 | Formación V | 24 | Triángulo Alcista |
| 11 | HCH Ascendente | 25 | Triángulo Bajista |
| 12 | HCH Descendente | 26 | Triángulo Simétrico Alcista |
| 13 | Media Móvil EMA200 | 27 | Triángulo Simétrico Bajista |

---

## ⚙️ Opciones de Línea de Comandos

### Interfaz web:
```bash
python -m interfaz_web.servidor --puerto 5000 --intervalo 10.0
```

### Entrenador:
```bash
python -m detector_patrones.entrenador_patrones --ruta-datos datos/patrones --epocas 50
```

### Organizador de datos:
```bash
python organizar_datos.py --origen "PARA ENTRENAR" --destino datos/patrones --variantes 50
```

---

## 🖥️ Requisitos de Hardware

- **GPU:** NVIDIA RTX 5070 Ti (o cualquier GPU CUDA)
- **VRAM:** El sistema usa máximo 40% de la VRAM
- **CPU:** Fallback automático si no hay GPU
- **RAM:** Mínimo 8 GB
- **Chrome/Chromium** instalado (para Selenium)

### PyTorch con CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Todas las dependencias:
```bash
pip install -r requirements.txt
```

---

## 🔧 Solución de Problemas

| Problema | Solución |
|----------|----------|
| `CUDA disponible: False` | Reinstalar PyTorch CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall` |
| Chrome no inicia | Instalar Chrome o verificar chromedriver |
| `ModuleNotFoundError: flask` | `pip install flask` |
| Error de VRAM | Reducir `fraccion_vram_maxima` en `configuracion_vision.py` |
| Umbral muy alto | Bajar `umbral_confianza_minima` (default: 0.70) |
