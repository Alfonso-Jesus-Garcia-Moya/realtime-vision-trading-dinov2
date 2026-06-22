# 📊 REPORTE TÉCNICO COMPLETO
## Planificación de Sistemas Inteligentes en Tiempo Real
### Sistema de Detección de Patrones Bursátiles mediante Visión por Computadora

---

> **Versión:** 1.0 — Sprint Final  
> **Fecha:** 19 de Junio de 2026  
> **Arquitectura:** DINOv2 (Meta AI) + Selenium Headless + Flask + PyTorch  
> **Hardware objetivo:** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM) — CPU fallback  
> **Precisión alcanzada:** **100% validación** / **97.1% entrenamiento** (Época 29/30)

---

## 📑 Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Arquitectura General del Sistema](#2-arquitectura-general-del-sistema)
3. [Estructura de Directorios](#3-estructura-de-directorios)
4. [Módulo 1: Capa de Captura de Datos (`captura_datos/`)](#4-módulo-1-capa-de-captura-de-datos)
5. [Módulo 2: Detector de Patrones (`detector_patrones/`)](#5-módulo-2-detector-de-patrones)
6. [Módulo 3: Interfaz Web (`interfaz_web/`)](#6-módulo-3-interfaz-web)
7. [Pipeline de Datos: Flujo Completo](#7-pipeline-de-datos-flujo-completo)
8. [Pipeline de Entrenamiento](#8-pipeline-de-entrenamiento)
9. [Pipeline de Inferencia en Tiempo Real](#9-pipeline-de-inferencia-en-tiempo-real)
10. [Relación entre Archivos (Mapa de Dependencias)](#10-relación-entre-archivos)
11. [Stack Tecnológico](#11-stack-tecnológico)
12. [Resultados del Entrenamiento](#12-resultados-del-entrenamiento)
13. [Clases de Patrones Técnicos](#13-clases-de-patrones-técnicos)
14. [Scripts Auxiliares](#14-scripts-auxiliares)
15. [Consideraciones de Hardware y GPU](#15-consideraciones-de-hardware-y-gpu)
16. [Instrucciones de Ejecución](#16-instrucciones-de-ejecución)
17. [Conclusiones](#17-conclusiones)

---

## 1. Resumen Ejecutivo

Este proyecto implementa un **sistema inteligente de detección de patrones de análisis técnico bursátil en tiempo real**, inspirado en plataformas como **GBM Broker Plus**. El sistema captura automáticamente el gráfico del índice **S&P 500 (^GSPC)** desde Yahoo Finance cada 10 segundos, convierte la captura directamente a tensores en memoria (sin guardar archivos en disco), y utiliza un modelo de visión por computadora basado en **DINOv2 de Meta AI** para clasificar el patrón técnico visible en el gráfico.

### Características principales:
- 🔄 **Captura en tiempo real** del chart de Yahoo Finance cada 10 segundos
- 🧠 **In-Memory Tensor Pipeline** — las imágenes nunca se guardan como archivos
- 🎯 **29 clases de patrones técnicos** reconocidos (canales, HCH, triángulos, osciladores, etc.)
- 🖥️ **Interfaz web en tiempo real** estilo terminal de trading con diseño Aero Crystal
- 📊 **DINOv2 ViT-B/14** como backbone congelado + cabeza MLP entrenable
- 🔁 **Reentrenamiento desde la UI** con un solo clic
- 🇲🇽 **Todo en español** — variables, clases, funciones, logs y comentarios

---

## 2. Arquitectura General del Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARQUITECTURA DEL SISTEMA                        │
│              Planificación de Sistemas Inteligentes en TR               │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │   Yahoo Finance      │
    │   (finance.yahoo     │
    │    .com/chart/^GSPC) │
    └────────┬─────────────┘
             │ HTTPS (cada 10s)
             ▼
    ┌──────────────────────┐      ┌─────────────────────────┐
    │  CAPTURADOR PANTALLA │      │    CONFIGURACIÓN        │
    │  (Selenium Chrome    │◄─────│  (URL, intervalos,      │
    │   Headless)          │      │   selectores CSS)       │
    │  ► Screenshot del    │      └─────────────────────────┘
    │    elemento <canvas> │
    └────────┬─────────────┘
             │ bytes PNG en memoria
             ▼
    ┌──────────────────────┐
    │  TUBERÍA DE TENSORES │
    │  (In-Memory Pipeline)│
    │  ► PIL.Image.open()  │
    │  ► Resize 224×224    │
    │  ► ToTensor [0,1]    │
    │  ► Normalizar        │
    │    ImageNet μ/σ      │
    │  ► Buffer circular   │
    │    (100 tensores)    │
    └────────┬─────────────┘
             │ torch.Tensor [3, 224, 224]
             ▼
    ┌──────────────────────┐      ┌─────────────────────────┐
    │  PROCESADOR VISIÓN   │      │   MODELO VISIÓN         │
    │  ► DINOv2 backbone   │◄─────│  ► DINOv2 ViT-B/14     │
    │    (embedding 768-d) │      │    (86.6M params frozen)│
    │  ► Cabeza MLP        │      │  ► MLP: 768→256→29      │
    │    (768→256→29)      │      │    (204K params trained) │
    │  ► Softmax → clase   │      │  ► Pesos: cabeza_       │
    │  ► Análisis mitad    │      │    patrones.pth         │
    │    derecha           │      └─────────────────────────┘
    └────────┬─────────────┘
             │ {patron, confianza, predicciones}
             ▼
    ┌──────────────────────┐
    │  SERVIDOR FLASK      │
    │  ► API REST /api/    │
    │    estado             │
    │  ► WebSocket polling │
    │  ► Logs en tiempo    │
    │    real               │
    │  ► Cambio dinámico   │
    │    de URL            │
    │  ► Botón REINICIAR   │
    │    TODO              │
    └────────┬─────────────┘
             │ HTTP JSON + Base64 PNG
             ▼
    ┌──────────────────────┐
    │  NAVEGADOR CLIENTE   │
    │  ► Dashboard Aero    │
    │    Crystal           │
    │  ► Chart en vivo     │
    │  ► Historial         │
    │  ► Logs del sistema  │
    │  ► Panel de control  │
    └──────────────────────┘
```

---

## 3. Estructura de Directorios

```
PLANIFICACION DE SISTEMAS INTELIGENTES EN TIEMPO REAL/
│
├── 📄 iniciar.py                          # Punto de entrada principal
├── 📄 principal.py                        # Orquestador alternativo
├── 📄 requirements.txt                    # Dependencias Python
├── 📄 REPORTE_PROYECTO.md                 # ← Este reporte
├── 📄 INSTRUCCIONES.md                    # Instrucciones rápidas
│
├── 📁 captura_datos/                      # MÓDULO 1: Capa de datos
│   ├── __init__.py                        #   Exporta clases públicas
│   ├── configuracion.py                   #   Dataclass de configuración
│   ├── capturador_pantalla.py             #   Selenium Chrome headless
│   └── tuberia_tensores.py                #   Conversión PNG→Tensor en memoria
│
├── 📁 detector_patrones/                  # MÓDULO 2: IA / Visión
│   ├── __init__.py                        #   Exporta clases públicas
│   ├── configuracion_vision.py            #   Hiperparámetros + mapa de 29 clases
│   ├── modelo_vision.py                   #   DINOv2 + cabeza MLP (PyTorch)
│   ├── procesador_vision.py               #   Inferencia en tiempo real
│   └── entrenador_patrones.py             #   Entrenamiento de la cabeza MLP
│
├── 📁 interfaz_web/                       # MÓDULO 3: Frontend + API
│   ├── __init__.py                        #   Exporta servidor
│   ├── servidor.py                        #   Flask + hilo de captura/inferencia
│   ├── plantillas/                        #   Templates HTML (Jinja2)
│   │   └── terminal.html                  #   Dashboard Aero Crystal
│   └── estaticos/                         #   CSS/JS estáticos
│
├── 📁 datos/                              # Datos de entrenamiento organizados
│   └── patrones/                          #   29 subcarpetas (1 por clase)
│       ├── Bandera/          (16 imgs)    #   Original + 15 augmentaciones
│       ├── Canal_Alcista/    (16 imgs)
│       ├── HCH_Ascendente/  (16 imgs)
│       ├── Sin_Patron/      (20 imgs)    #   Generadas sintéticamente
│       └── ... (29 clases total)
│
├── 📁 modelos_guardados/                  # Pesos entrenados
│   └── cabeza_patrones.pth                #   ~800 KB — cabeza MLP
│
├── 📁 PARA ENTRENAR/                      # Imágenes originales (fuente)
│   ├── 31 imágenes PNG de patrones
│   └── 3 PDFs de referencia técnica
│
├── 📁 RESULTADOS/                         # Capturas de resultados
│   └── 1 CORRECTO .png
│
├── 📄 organizar_datos_entrenamiento.py    # Script: organiza imgs → carpetas
├── 📄 aumentar_datos.py                   # Script: data augmentation ×15
├── 📄 diagnostico_selectores.py           # Diagnóstico de selectores CSS
└── 📄 verificar_modulos.py                # Verificación de instalación
```

---

## 4. Módulo 1: Capa de Captura de Datos

### Responsabilidad
Capturar screenshots del gráfico S&P 500 de Yahoo Finance cada 10 segundos mediante Chrome headless, y convertir los bytes PNG directamente a tensores PyTorch en memoria sin guardar en disco.

---

### 4.1 `captura_datos/configuracion.py` (67 líneas)

**Propósito:** Centraliza TODA la configuración del sistema de captura en un `@dataclass` inmutable con valores por defecto.

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `url_grafico` | `https://finance.yahoo.com/chart/^GSPC` | URL del chart S&P 500 |
| `intervalo_captura_seg` | `10` | Segundos entre capturas |
| `tiempo_espera_carga_seg` | `15` | Timeout de carga de página |
| `selectores_grafico` | Lista de 6 CSS selectors | Selectores para encontrar el `<canvas>` |
| `navegador_sin_cabeza` | `True` | Chrome headless |
| `ancho_ventana` / `alto_ventana` | `1920 × 1080` | Resolución de la ventana |
| `tamano_tensor` | `(224, 224)` | Dimensión de salida (DINOv2 estándar) |
| `canales_color` | `3` | RGB |
| `capacidad_buffer` | `100` | Tensores máximos en buffer circular |
| `normalizar_tensor` | `True` | Normalizar con media/desv. ImageNet |

**Selectores CSS probados (en orden de prioridad):**
```python
selectores_grafico = [
    ".chartContainer canvas",      # Selector principal
    "canvas.cq-chart-canvas",      # Canvas del chart
    "#chartContainer canvas",      # Por ID
    ".chart-container canvas",     # Alternativo
    "canvas[class*='chart']",      # Wildcard
    "canvas",                      # Último recurso
]
```

---

### 4.2 `captura_datos/capturador_pantalla.py` (~180 líneas)

**Propósito:** Maneja el ciclo de vida completo de Chrome headless (iniciar, navegar, capturar, cerrar). Implementa Element-Specific Web Screenshotting.

**Clase principal:** `CapturadorPantalla`

```python
class CapturadorPantalla:
    def __init__(self, configuracion: Configuracion)
    def iniciar(self) -> None          # Inicia Chrome + navega a URL
    def capturar(self) -> bytes | None # Retorna bytes PNG del canvas
    def cerrar(self) -> None           # Cierra Chrome limpiamente
    def cambiar_url(self, url) -> None # Cambia URL sin reiniciar
```

**Flujo interno de `iniciar()`:**
1. Configura opciones de Chrome (headless, window-size, disable-gpu, no-sandbox)
2. Usa `webdriver_manager` para auto-descargar ChromeDriver compatible
3. Navega a la URL de Yahoo Finance
4. Intenta hacer clic en vista "YTD" (botón `#tab-YTD`)
5. Espera a que aparezca el elemento `<canvas>` del gráfico
6. Prueba los 6 selectores CSS en orden hasta encontrar uno que funcione
7. Guarda referencia al elemento encontrado

**Flujo interno de `capturar()`:**
1. Llama a `elemento_grafico.screenshot_as_png`
2. Retorna los bytes PNG crudos (sin guardar a disco)
3. Si falla, intenta re-encontrar el elemento con los selectores
4. Log de dimensiones capturadas (ej: "1598×826 px")

**Dependencias externas:**
- `selenium.webdriver.Chrome`
- `webdriver_manager.chrome.ChromeDriverManager`
- `selenium.webdriver.support.WebDriverWait`

---

### 4.3 `captura_datos/tuberia_tensores.py` (~120 líneas)

**Propósito:** Convierte bytes PNG crudos a tensores PyTorch normalizados **sin guardar nada en disco**. Mantiene un buffer circular de los últimos N tensores.

**Clase principal:** `TuberiaTensores`

```python
class TuberiaTensores:
    def __init__(self, configuracion: Configuracion)
    def procesar(self, bytes_png: bytes) -> torch.Tensor
    def obtener_ultimo(self) -> torch.Tensor | None
    def obtener_buffer_completo(self) -> list[torch.Tensor]
    def limpiar_buffer(self) -> None
```

**Pipeline de transformación (In-Memory Tensor Pipeline):**

```
bytes PNG (en memoria)
    │
    ▼ io.BytesIO(bytes_png)
PIL.Image.open(buffer)
    │
    ▼ .convert("RGB")
Imagen PIL RGB
    │
    ▼ transforms.Resize((224, 224))
Imagen 224×224
    │
    ▼ transforms.ToTensor()
torch.Tensor [3, 224, 224]  ← valores [0.0, 1.0]
    │
    ▼ transforms.Normalize(
    │     mean=[0.485, 0.456, 0.406],  ← Media ImageNet
    │     std=[0.229, 0.224, 0.225]    ← Desv. ImageNet
    │ )
torch.Tensor [3, 224, 224]  ← normalizado
    │
    ▼ buffer_circular.append(tensor)
Buffer deque(maxlen=100)
```

**Punto clave:** Nunca se escribe `imagen.save()` ni `torch.save()` al disco para las capturas en tiempo real. Todo el flujo es **estrictamente en memoria**.

---

### 4.4 `captura_datos/__init__.py`

```python
from .configuracion import Configuracion
from .capturador_pantalla import CapturadorPantalla
from .tuberia_tensores import TuberiaTensores
```

Exporta las 3 clases para uso externo: `from captura_datos import Configuracion, CapturadorPantalla, TuberiaTensores`

---

## 5. Módulo 2: Detector de Patrones

### Responsabilidad
Clasificar tensores de imágenes de gráficos bursátiles en una de 29 categorías de patrones de análisis técnico, usando DINOv2 como extractor de features y una cabeza MLP ligera como clasificador.

---

### 5.1 `detector_patrones/configuracion_vision.py` (130 líneas)

**Propósito:** Define el mapeo de 29 clases, hiperparámetros del modelo DINOv2 y límites de VRAM.

**Constantes globales:**

```python
MAPA_CLASES_PATRONES: Dict[int, str] = {
    0:  "Bandera",           1:  "Banderola",
    2:  "Canal_Alcista",     3:  "Canal_Bajista",
    4:  "Canal_Lateral",     5:  "Cuna",
    # ... 29 clases en total (ver sección 13)
    28: "Triangulo_Simetrico_Descendente",
}

NUMERO_CLASES = 29  # Automático desde len(MAPA)
```

**Dataclass `ConfiguracionVision`:**

| Hiperparámetro | Valor | Descripción |
|----------------|-------|-------------|
| `nombre_modelo_base` | `"dinov2_vitb14"` | Backbone de Meta AI |
| `dimension_embedding` | `768` | Dimensión del vector de features |
| `congelar_backbone` | `True` | Siempre congelado (ahorra VRAM) |
| `numero_clases` | `29` | Clases de patrones |
| `dimension_oculta_mlp` | `256` | Neuronas en capa oculta |
| `tasa_dropout` | `0.3` | Regularización |
| `tamano_entrada` | `(224, 224)` | Resolución DINOv2 estándar |
| `fraccion_vram_maxima` | `0.40` | Máximo 40% de VRAM |
| `umbral_confianza_minima` | `0.70` | Debajo = "Sin Patrón Claro" |
| `tasa_aprendizaje` | `1e-3` | Learning rate AdamW |
| `epocas_maximas` | `50` | Máximo de épocas |
| `paciencia_early_stopping` | `7` | Épocas sin mejora para parar |
| `tamano_lote_entrenamiento` | `32` | Batch size |
| `fraccion_validacion` | `0.2` | 80/20 split |

---

### 5.2 `detector_patrones/modelo_vision.py` (~200 líneas)

**Propósito:** Define la arquitectura del clasificador: DINOv2 ViT-B/14 (backbone congelado) + cabeza MLP entrenable.

**Clase principal:** `ModeloVisionPatrones(nn.Module)`

```python
class ModeloVisionPatrones(nn.Module):
    def __init__(self, configuracion: ConfiguracionVision)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    def guardar_pesos_cabeza(self, ruta: str) -> None
    def cargar_pesos_cabeza(self, ruta: str) -> None
```

**Arquitectura interna:**

```
Entrada: torch.Tensor [B, 3, 224, 224]
         │
         ▼
┌─────────────────────────────────┐
│  DINOv2 ViT-B/14 (Backbone)    │
│  86,580,480 parámetros          │
│  ❄ CONGELADOS (sin gradientes)  │
│                                 │
│  12 bloques Transformer         │
│  Patch size: 14×14              │
│  Embedding: 768-d               │
│  Atención: Multi-head (12)      │
└──────────┬──────────────────────┘
           │ [B, 768] (CLS token)
           ▼
┌─────────────────────────────────┐
│  Cabeza MLP (Entrenable)        │
│  204,317 parámetros             │
│  🔥 CON gradientes              │
│                                 │
│  Linear(768, 256)               │
│  ├── GELU activation            │
│  ├── Dropout(0.3)               │
│  └── Linear(256, 29)            │
└──────────┬──────────────────────┘
           │ [B, 29] (logits)
           ▼
       Softmax → predicción
```

**Desglose de parámetros:**

| Componente | Parámetros | Estado |
|------------|-----------|--------|
| DINOv2 ViT-B/14 (backbone) | 86,580,480 (86.6M) | ❄ Congelados |
| Cabeza MLP (clasificador) | 204,317 (~204K) | 🔥 Entrenables |
| **Total** | **86,784,797** | — |

---

### 5.3 `detector_patrones/procesador_vision.py` (~170 líneas)

**Propósito:** Ejecuta inferencia en tiempo real sobre tensores individuales. Implementa análisis dual (imagen completa + mitad derecha).

**Clase principal:** `ProcesadorVision`

```python
class ProcesadorVision:
    def __init__(self, configuracion: ConfiguracionVision)
    def procesar_tensor(self, tensor: torch.Tensor) -> dict
    def cargar_modelo(self) -> None
```

**Flujo de inferencia:**

```
tensor [3, 224, 224]
    │
    ├──► ANÁLISIS COMPLETO
    │    unsqueeze → [1, 3, 224, 224]
    │    modelo.forward() → logits [1, 29]
    │    softmax → probabilidades
    │    argmax → clase predicha
    │    ¿confianza > 70%? → patrón detectado / "Sin Patrón Claro"
    │
    └──► ANÁLISIS MITAD DERECHA
         tensor[:, :, 112:] → crop derecho
         resize → [3, 224, 224]
         modelo.forward() → logits [1, 29]
         softmax → probabilidades
         argmax → clase predicha (tendencia reciente)
```

**Retorno:**
```python
{
    "patron_completo": "Canal_Alcista",
    "confianza_completo": 0.87,
    "patron_mitad": "Resistencia",
    "confianza_mitad": 0.73,
    "tiempo_inferencia_ms": 95,
    "dispositivo": "cpu"
}
```

---

### 5.4 `detector_patrones/entrenador_patrones.py` (~350 líneas)

**Propósito:** Entrena la cabeza MLP sobre el dataset de patrones organizados. Implementa early stopping, logging detallado y guardado automático de pesos.

**Función principal:** `entrenar(ruta_datos, epocas, lote, paciencia, forzar_gpu)`

**Interfaz CLI:**
```bash
python -m detector_patrones.entrenador_patrones \
    --ruta-datos datos/patrones \
    --epocas 30 \
    --lote 16 \
    --paciencia 5 \
    --forzar-gpu
```

**Pipeline de entrenamiento:**

```
datos/patrones/
├── Bandera/ (16 imgs)
├── Canal_Alcista/ (16 imgs)
├── ...
└── Sin_Patron/ (20 imgs)
         │
         ▼ torchvision.datasets.ImageFolder
    Dataset (516 imágenes, 29 clases)
         │
         ▼ random_split (80/20)
    Train: 413 │ Val: 103
         │
         ▼ DataLoader (batch=16, shuffle=True)
    Batches de [16, 3, 224, 224]
         │
         ▼ Data Augmentation (train only):
         │   RandomHorizontalFlip(p=0.5)
         │   RandomRotation(±10°)
         │   ColorJitter(brightness=0.2, contrast=0.2)
         │   Resize(224×224)
         │   ToTensor + Normalize(ImageNet)
         │
         ▼ Forward pass:
         │   DINOv2 backbone → embedding [B, 768]
         │   MLP head → logits [B, 29]
         │
         ▼ Loss: CrossEntropyLoss
         ▼ Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
         ▼ Backward + step
         │
         ▼ Validación cada época
         ▼ Early stopping (paciencia=5)
         ▼ Guardar mejores pesos → cabeza_patrones.pth
```

---

## 6. Módulo 3: Interfaz Web

### Responsabilidad
Servir el dashboard web en tiempo real, orquestar el hilo de captura/inferencia, exponer API REST y manejar el estado global thread-safe.

---

### 6.1 `interfaz_web/servidor.py` (665 líneas)

**Propósito:** Módulo más grande del sistema. Es el "cerebro" que conecta captura, procesamiento e interfaz. Incluye Flask server, estado global, hilo de captura y API REST.

**Clases principales:**

#### `EstadoGlobal` (~128 líneas)
Estado compartido thread-safe entre el hilo de captura y Flask.

```python
class EstadoGlobal:
    # Imagen actual
    imagen_base64: str
    # Predicciones
    patron_completo: str, confianza_completo: float
    patron_mitad: str, confianza_mitad: float
    # Historial
    historial: deque(maxlen=50)
    # Logs
    logs: deque(maxlen=200)
    # Métricas
    total_capturas: int
    tiempo_inferencia_ms: float
    dispositivo: str
    tamano_buffer: int
    # Control
    senal_cambiar_url: str | None
    senal_detener: bool
    entrenando: bool
```

#### `ManejadorLogsUI` (logging.Handler)
Captura todos los logs de Python y los inyecta en el estado global para mostrarlos en la UI.

**API REST endpoints:**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Sirve `terminal.html` (dashboard) |
| `/api/estado` | GET | Estado completo (JSON): imagen, predicciones, historial, logs, métricas |
| `/api/cambiar-url` | POST | Cambia la URL del chart (recarga Chrome) |
| `/api/reiniciar` | POST | Detiene captura → reentrena modelo → reinicia todo |

**Hilo de captura/inferencia (`hilo_captura_inferencia`):**

```
while not senal_detener:
    │
    ├── ¿Hay señal de cambiar URL?
    │   └── Sí → capturador.cambiar_url(nueva_url)
    │
    ├── bytes_png = capturador.capturar()
    │
    ├── tensor = tuberia.procesar(bytes_png)
    │
    ├── resultado = procesador_vision.procesar_tensor(tensor)
    │
    ├── imagen_b64 = bytes_png_a_base64(bytes_png)
    │
    ├── estado_global.actualizar(resultado, imagen_b64)
    │
    └── time.sleep(intervalo_captura)  # 10 segundos
```

---

### 6.2 `interfaz_web/plantillas/terminal.html`

**Propósito:** Dashboard completo tipo terminal de trading con diseño **Aero Crystal** (efecto glassmorphism con backdrop-blur, gradientes animados y transparencias).

**Secciones del dashboard:**

```
┌──────────────────────────────────────────────────────────┐
│  🔴 BARRA SUPERIOR                                       │
│  [URL del chart] [CHART YTD ▼ ^GSPC] [REINICIAR] [SIN]  │
├──────────────┬───────────────────────────────────────────┤
│              │  ► PREDICCIÓN DUAL SÍNCRONA               │
│  CHART       │  ┌─────────────────────────┐              │
│  EN VIVO     │  │ ANÁLISIS CAPTURA COMPL. │              │
│  (iframe     │  │ ████████░░ Canal_Alcista│              │
│   Yahoo)     │  │ Confianza: 87.3%        │              │
│              │  └─────────────────────────┘              │
│              │  ┌─────────────────────────┐              │
│              │  │ ANÁLISIS MITAD DERECHA  │              │
│              │  │ ████████░░ Resistencia  │              │
│              │  │ Confianza: 73.1%        │              │
│              │  └─────────────────────────┘              │
│              │                                           │
│              │  📊 MÉTRICAS                              │
│              │  Capturas: 245 | Buffer: 100              │
│              │  Inferencia: 95ms | Dispositivo: CPU      │
│              │                                           │
│              │  📜 HISTORIAL (últimas 50)                │
│              │  16:23:06 Canal_Alcista 87% 95ms          │
│              │  16:22:56 Resistencia   73% 102ms         │
├──────────────┴───────────────────────────────────────────┤
│  🖥️ SYSTEM LOGS (últimas 200 líneas)                     │
│  16:23:06 │ INFO │ tuberia_tensores │ Tensor #228...     │
│  16:23:06 │ INFO │ interfaz_web │ 📸 #228 │ Canal...     │
└──────────────────────────────────────────────────────────┘
```

**Tecnologías frontend:**
- HTML5 + CSS3 (backdrop-filter, gradientes animados)
- JavaScript vanilla (fetch polling cada ~2s a `/api/estado`)
- Sin frameworks externos (zero dependencies)

---

## 7. Pipeline de Datos: Flujo Completo

### Diagrama de flujo end-to-end

```
[Yahoo Finance] ──HTTPS──► [Chrome Headless] ──screenshot_as_png──►
    │
    ▼ bytes PNG (memoria)
    │
[PIL.Image.open(BytesIO)] ──RGB──► [Resize 224×224] ──►
    │
    ▼ PIL Image
    │
[ToTensor()] ──[0,1]──► [Normalize(μ_ImageNet, σ_ImageNet)] ──►
    │
    ▼ torch.Tensor [3, 224, 224]
    │
[Buffer Circular (deque, 100)] ──último tensor──►
    │
    ▼ torch.Tensor [3, 224, 224]
    │
[DINOv2 ViT-B/14] ──embedding──► [768-d vector] ──►
    │
    ▼ torch.Tensor [768]
    │
[MLP Head (768→256→29)] ──logits──► [Softmax] ──►
    │
    ▼ Probabilidades [29]
    │
[argmax + umbral 70%] ──►  "Canal_Alcista" (87.3%)
    │
    ▼
[Flask API /api/estado] ──JSON──► [Dashboard Aero Crystal]
```

### Tiempos típicos del pipeline

| Etapa | Tiempo | Observación |
|-------|--------|-------------|
| Captura screenshot | ~50ms | Chrome headless, canvas 1598×826 |
| PNG → Tensor | ~5ms | PIL + torchvision transforms |
| DINOv2 forward | ~80ms (CPU) | ViT-B/14, batch=1 |
| MLP head | ~1ms | 204K params, muy rápido |
| **Total inferencia** | **~90-130ms** | Varía con carga de sistema |
| Intervalo entre capturas | 10,000ms | Configurable |

---

## 8. Pipeline de Entrenamiento

### Preparación de datos

#### Paso 1: Organización (`organizar_datos_entrenamiento.py`)

```
PARA ENTRENAR/                      datos/patrones/
├── GRAFICO DE UN CANAL ALCISTA.png ──► Canal_Alcista/
├── GRAFICO DE H-C-H ASCENDENTE.png ──► HCH_Ascendente/
├── GRAFICO DE UN DOBLE TECHO.png   ──► Doble_Techo/
├── ...                              ──► ...
└── (31 imágenes PNG)               ──► (29 carpetas-clase)
```

Mapeo de archivos con nombres descriptivos a carpetas de clase estandarizadas.

#### Paso 2: Aumentación (`aumentar_datos.py`)

Cada imagen original genera **15 variaciones** con transformaciones aleatorias:

| Transformación | Probabilidad | Rango |
|---------------|-------------|-------|
| Rotación | 70% | -15° a +15° |
| Flip horizontal | 50% | — |
| Brillo | 60% | ×0.7 a ×1.3 |
| Contraste | 60% | ×0.7 a ×1.4 |
| Saturación | 50% | ×0.6 a ×1.5 |
| Nitidez | 40% | ×0.5 a ×2.0 |
| Recorte (zoom) | 60% | 2-12% margen |
| Blur gaussiano | 30% | σ 0.3-1.2 |

**Clase `Sin_Patron`:** 20 imágenes sintéticas con gradientes y ruido aleatorio.

**Resultado final:** 516 imágenes en 29 clases (16 por clase + 20 Sin_Patron).

### Proceso de entrenamiento

```
Comando:
python -m detector_patrones.entrenador_patrones \
    --ruta-datos datos/patrones \
    --epocas 30 \
    --lote 16 \
    --paciencia 5

Configuración:
├── Dataset: 516 imágenes, 29 clases
├── Split: 413 train (80%) / 103 val (20%)
├── Backbone: DINOv2 ViT-B/14 (CONGELADO)
├── Cabeza: MLP 768→256→29 (ENTRENABLE, 204K params)
├── Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
├── Loss: CrossEntropyLoss
├── Early stopping: 5 épocas sin mejora
└── Dispositivo: CPU (PyTorch no soporta sm_120 aún)
```

---

## 12. Resultados del Entrenamiento

### Curva de aprendizaje completa

| Época | Pérd. Train | Pérd. Val | Prec. Train | Prec. Val | Tiempo |
|-------|------------|-----------|-------------|-----------|--------|
| 1 | 3.0929 | 2.4680 | 16.9% | 31.1% | 50.3s |
| 2 | 2.2228 | 1.6688 | 38.0% | 50.5% | 50.3s |
| 3 | 1.6063 | 1.2655 | 52.8% | 64.1% | 50.6s |
| 4 | 1.1896 | 0.9143 | 63.2% | 77.7% | 49.9s |
| 5 | 0.9353 | 0.6604 | 76.0% | 80.6% | 49.4s |
| 6 | 0.8696 | 0.6041 | 73.6% | 77.7% | 48.5s |
| 7 | 0.6471 | 0.5081 | 81.4% | 89.3% | 46.8s |
| 8 | 0.6426 | 0.4364 | 81.1% | 87.4% | 46.5s |
| 9 | 0.5504 | 0.3746 | 84.0% | 92.2% | 46.2s |
| 10 | 0.4567 | 0.2986 | 87.4% | 95.1% | 46.3s |
| 11 | 0.3719 | 0.2287 | 90.3% | 95.1% | 46.5s |
| 12 | 0.3371 | 0.2534 | 92.0% | 96.1% | 46.3s |
| 13 | 0.2888 | 0.2208 | 92.5% | 94.2% | 46.1s |
| 14 | 0.3241 | 0.2539 | 90.1% | 92.2% | 46.1s |
| 15 | 0.3107 | 0.1647 | 90.8% | 97.1% | 46.2s |
| 16 | 0.2630 | 0.2154 | 93.5% | 95.1% | 46.1s |
| 17 | 0.2534 | 0.1865 | 93.0% | 93.2% | 46.2s |
| 18 | 0.2403 | 0.1331 | 92.0% | 98.1% | 46.2s |
| 19 | 0.2282 | 0.1945 | 93.0% | 95.1% | 46.4s |
| 20 | 0.2511 | 0.1194 | 91.8% | 97.1% | 46.1s |
| 21 | 0.1862 | 0.1504 | 94.9% | 96.1% | 47.1s |
| 22 | 0.1776 | 0.1016 | 96.1% | 98.1% | 46.4s |
| 23 | 0.1739 | 0.1057 | 94.9% | 98.1% | 46.5s |
| 24 | 0.1688 | 0.1094 | 95.2% | 96.1% | 46.3s |
| 25 | 0.1831 | 0.0827 | 94.7% | 99.0% | 46.3s |
| 26 | 0.1480 | 0.1271 | 96.6% | 95.1% | 46.3s |
| 27 | 0.1452 | 0.1253 | 95.6% | 97.1% | 45.8s |
| 28 | 0.1372 | 0.1063 | 95.9% | 96.1% | 45.7s |
| **29** | **0.1068** | **0.0557** | **97.1%** | **100.0%** | 45.8s |
| 30 | 0.1392 | 0.0769 | 95.9% | 100.0% | 45.6s |

### Métricas finales

| Métrica | Valor |
|---------|-------|
| **Mejor precisión validación** | **100.0%** |
| **Mejor pérdida validación** | **0.0557** |
| **Precisión entrenamiento final** | **97.1%** |
| **Tiempo total de entrenamiento** | **1,408.6 segundos (~23.5 min)** |
| **Épocas completadas** | **30/30** |
| **Parámetros entrenados** | **204,317** |
| **Dispositivo** | **CPU** |
| **Tamaño pesos guardados** | **~800 KB** |

---

## 9. Pipeline de Inferencia en Tiempo Real

### Análisis dual simultáneo

El sistema ejecuta **dos inferencias por cada captura**:

1. **Análisis completo:** Tensor original [3, 224, 224] → identifica el patrón dominante del gráfico completo
2. **Análisis mitad derecha:** Crop del 50% derecho → identifica la tendencia más reciente

```python
# Pseudocódigo del análisis dual
tensor_completo = captura  # [3, 224, 224]
tensor_derecho = tensor_completo[:, :, 112:]  # [3, 224, 112] → resize a [3, 224, 224]

prediccion_completa = modelo(tensor_completo)  # "Canal_Alcista" 87%
prediccion_derecha = modelo(tensor_derecho)    # "Resistencia" 73%
```

### Umbral de confianza

- **≥ 70%** → Patrón detectado con nombre de clase
- **< 70%** → "Sin Patrón Claro" (sin suficiente certeza)

---

## 10. Relación entre Archivos

### Mapa de dependencias (imports)

```
iniciar.py
├── captura_datos.Configuracion
├── interfaz_web.crear_aplicacion()
└── interfaz_web.iniciar_servidor()

interfaz_web/servidor.py
├── captura_datos.Configuracion
├── captura_datos.CapturadorPantalla
├── captura_datos.TuberiaTensores
├── detector_patrones.ConfiguracionVision
├── detector_patrones.ModeloVisionPatrones
├── detector_patrones.ProcesadorVision
└── detector_patrones.entrenador_patrones (para reentrenamiento)

captura_datos/capturador_pantalla.py
└── captura_datos.Configuracion

captura_datos/tuberia_tensores.py
└── captura_datos.Configuracion

detector_patrones/modelo_vision.py
└── detector_patrones.ConfiguracionVision

detector_patrones/procesador_vision.py
├── detector_patrones.ConfiguracionVision
├── detector_patrones.ModeloVisionPatrones
└── detector_patrones.MAPA_CLASES_PATRONES

detector_patrones/entrenador_patrones.py
├── detector_patrones.ConfiguracionVision
├── detector_patrones.ModeloVisionPatrones
└── torchvision.datasets.ImageFolder
```

### Diagrama de flujo de datos entre módulos

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  captura_datos/  │     │detector_patrones│     │  interfaz_web/  │
│                 │     │                 │     │                 │
│ Configuracion ──┼──►──┤                 │     │                 │
│                 │     │                 │     │                 │
│ Capturador     │     │                 │     │                 │
│ Pantalla ──────┼──bytes PNG──►         │     │                 │
│                 │     │                 │     │                 │
│ Tuberia        │     │                 │     │                 │
│ Tensores ──────┼──tensor──► Procesador │     │                 │
│                 │     │   Vision ──────┼──resultado──► Servidor│
│                 │     │                 │     │   Flask         │
│                 │     │ Modelo Vision   │     │                 │
│                 │     │ (DINOv2+MLP)    │     │ terminal.html   │
│                 │     │                 │     │  (Dashboard)    │
│                 │     │ Entrenador      │     │                 │
│                 │     │ Patrones        │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 11. Stack Tecnológico

### Backend

| Tecnología | Versión | Uso |
|-----------|---------|-----|
| **Python** | 3.12 | Lenguaje principal |
| **PyTorch** | 2.x | Framework de deep learning |
| **DINOv2 (Meta AI)** | ViT-B/14 | Backbone de visión |
| **torchvision** | 0.x | Transforms + datasets |
| **Selenium** | 4.x | Chrome headless automation |
| **webdriver-manager** | — | Auto-descarga ChromeDriver |
| **Flask** | 3.x | Web server + API REST |
| **Pillow (PIL)** | 10.x | Procesamiento de imágenes |

### Frontend

| Tecnología | Uso |
|-----------|-----|
| **HTML5** | Estructura del dashboard |
| **CSS3** | Glassmorphism / Aero Crystal |
| **JavaScript** | Polling API + actualización dinámica |
| **Sin frameworks** | Zero dependencies frontend |

### Infraestructura

| Componente | Detalle |
|-----------|---------|
| **SO** | Windows 11 |
| **GPU** | NVIDIA GeForce RTX 5070 Ti (16 GB) |
| **CPU** | Usado actualmente (PyTorch no soporta sm_120) |
| **Navegador** | Chrome 149 (headless) |
| **Puerto** | localhost:5000 |

### Dependencias (`requirements.txt`)

```
torch
torchvision
selenium
webdriver-manager
flask
Pillow
numpy
```

---

## 13. Clases de Patrones Técnicos

Las 29 clases representan patrones de análisis técnico bursátil clásicos:

### Patrones de Tendencia (canales)
| Índice | Clase | Descripción |
|--------|-------|-------------|
| 2 | Canal_Alcista | Tendencia alcista con líneas paralelas |
| 3 | Canal_Bajista | Tendencia bajista con líneas paralelas |
| 4 | Canal_Lateral | Rango lateral / consolidación |

### Patrones de Reversión
| Índice | Clase | Descripción |
|--------|-------|-------------|
| 8 | Doble_Suelo | Doble mínimo (bullish reversal) |
| 9 | Doble_Techo | Doble máximo (bearish reversal) |
| 10 | Formacion_V | Reversión en V (sharp reversal) |
| 11 | HCH_Ascendente | Hombro-Cabeza-Hombro invertido |
| 12 | HCH_Descendente | Hombro-Cabeza-Hombro (bearish) |
| 24 | Suelo_Redondeado | Cup / Rounding bottom |

### Patrones de Continuación
| Índice | Clase | Descripción |
|--------|-------|-------------|
| 0 | Bandera | Flag pattern (consolidación breve) |
| 1 | Banderola | Pennant pattern |
| 5 | Cuña | Wedge pattern |
| 25 | Triangulo_Recto_Alcista | Right-angle triangle (bullish) |
| 26 | Triangulo_Recto_Bajista | Right-angle triangle (bearish) |
| 27 | Triangulo_Simetrico_Ascendente | Symmetrical triangle (up) |
| 28 | Triangulo_Simetrico_Descendente | Symmetrical triangle (down) |

### Soportes y Resistencias
| Índice | Clase | Descripción |
|--------|-------|-------------|
| 21 | Resistencia | Nivel de resistencia (techo) |
| 23 | Soporte | Nivel de soporte (piso) |

### Indicadores Técnicos
| Índice | Clase | Descripción |
|--------|-------|-------------|
| 6 | Divergencia_MACD | Divergencia en MACD |
| 7 | Divergencia_RSI | Divergencia en RSI |
| 13 | Media_Movil_EMA200 | EMA de 200 periodos visible |
| 14 | Media_Movil_EMA50 | EMA de 50 periodos visible |
| 15 | Media_Movil_Ponderada | WMA visible |
| 16 | Media_Movil_Simple | SMA visible |
| 17 | Oscilador_K | Estocástico %K |
| 18 | Oscilador_Momento | Indicador de momento |
| 19 | Oscilador_ROC | Rate of Change |
| 20 | Oscilador_Williams | Williams %R |

### Clase especial
| Índice | Clase | Descripción |
|--------|-------|-------------|
| 22 | Sin_Patron | Sin patrón técnico claro |

---

## 14. Scripts Auxiliares

### `organizar_datos_entrenamiento.py`
- **Propósito:** Mapea archivos PNG de `PARA ENTRENAR/` a subcarpetas de clase en `datos/patrones/`
- **Mapeo:** 31 imágenes → 29 clases (2 clases con imágenes duplicadas)
- **Ejecución:** `python organizar_datos_entrenamiento.py`

### `aumentar_datos.py`
- **Propósito:** Data augmentation offline — genera 15 variaciones por imagen original
- **Resultado:** 516 imágenes totales
- **Incluye:** Generación sintética de 20 imágenes "Sin_Patron"
- **Ejecución:** `python aumentar_datos.py`

### `diagnostico_selectores.py`
- **Propósito:** Prueba los selectores CSS de Yahoo Finance para verificar cuál funciona
- **Uso:** Debugging cuando Yahoo cambia su HTML

### `verificar_modulos.py`
- **Propósito:** Verifica que todas las dependencias Python estén instaladas correctamente

---

## 15. Consideraciones de Hardware y GPU

### Situación actual

| Componente | Estado |
|-----------|--------|
| **GPU** | NVIDIA RTX 5070 Ti (sm_120 Blackwell) |
| **PyTorch** | No soporta sm_120 aún (requiere nightly build) |
| **Modo actual** | CPU fallback automático |
| **VRAM configurada** | 40% máximo (cuando GPU disponible) |

### Impacto en rendimiento

| Operación | CPU | GPU (estimado) |
|-----------|-----|----------------|
| Inferencia DINOv2 | ~80-130ms | ~5-10ms |
| Entrenamiento (1 época) | ~46-50s | ~3-5s |
| Entrenamiento total (30 épocas) | ~23 min | ~2-3 min |

### Solución futura
```bash
# Cuando PyTorch soporte sm_120:
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

El código ya está preparado con `dispositivo_preferido: str = "cuda"` y fallback automático a CPU.

---

## 16. Instrucciones de Ejecución

### Primera vez (setup completo)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Organizar datos de entrenamiento
python organizar_datos_entrenamiento.py

# 3. Generar augmentaciones
python aumentar_datos.py

# 4. Entrenar el modelo
python -m detector_patrones.entrenador_patrones \
    --ruta-datos datos/patrones \
    --epocas 30 \
    --lote 16 \
    --paciencia 5

# 5. Iniciar el sistema
python iniciar.py

# 6. Abrir en navegador
start http://localhost:5000
```

### Ejecución normal (sistema ya entrenado)

```bash
python iniciar.py
# → Abre http://localhost:5000
```

### Reentrenar desde la UI
1. Abrir http://localhost:5000
2. Clic en botón **🔁 REINICIAR TODO** (barra superior)
3. El sistema detiene captura → reentrena → reinicia automáticamente

---

## 17. Conclusiones

### Logros técnicos

1. **In-Memory Tensor Pipeline:** El flujo completo de Yahoo Finance → tensor PyTorch ocurre **100% en memoria** sin escritura a disco, cumpliendo el requisito de Streaming de Tensión Directo.

2. **Transfer Learning eficiente:** Usando DINOv2 como backbone congelado (86.6M params), solo se entrenan 204K parámetros de la cabeza MLP, logrando **100% validación** con solo 516 imágenes de entrenamiento.

3. **Análisis dual:** El sistema analiza tanto el gráfico completo como la mitad derecha (tendencia reciente), proporcionando dos perspectivas complementarias.

4. **Modularidad:** Tres módulos independientes (`captura_datos`, `detector_patrones`, `interfaz_web`) con interfaces limpias y responsabilidades separadas.

5. **Todo en español:** Variables, funciones, clases, logs, comentarios y interfaz de usuario completamente en español.

6. **Ejecución local:** No depende de APIs externas de IA — todo corre localmente con PyTorch.

### Métricas clave

| Métrica | Valor |
|---------|-------|
| Precisión validación | **100%** |
| Clases detectadas | **29** patrones técnicos |
| Latencia inferencia | **~90-130ms** (CPU) |
| Intervalo captura | **10 segundos** |
| Parámetros entrenados | **204,317** (0.24% del modelo) |
| Imágenes entrenamiento | **516** (augmentadas de 31 originales) |
| Tamaño pesos | **~800 KB** |
| Tiempo entrenamiento | **23.5 minutos** (CPU) |

### Líneas de trabajo futuro

- 🔧 Actualizar PyTorch a nightly con soporte sm_120 para aprovechar la RTX 5070 Ti
- 📈 Agregar más imágenes de entrenamiento por clase para mayor robustez
- 🔔 Implementar sistema de alertas cuando se detecte un patrón de alta confianza
- 📊 Agregar más símbolos/índices además de ^GSPC
- 🧪 Implementar backtesting con datos históricos

---

> **Archivo generado automáticamente como documentación técnica del proyecto.**  
> **Sistema de Detección de Patrones Bursátiles en Tiempo Real v1.0**
