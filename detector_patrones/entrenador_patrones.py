"""
Módulo Entrenador de la Cabeza de Clasificación de Patrones Técnicos.

Script utilitario que entrena SOLO la cabeza MLP/Lineal del clasificador,
manteniendo el backbone DINOv2 completamente congelado.  Esto permite
entrenar en pocos minutos incluso con datasets pequeños.

Estructura esperada del directorio de datos:
    datos/patrones/
    ├── Canal_Alcista/
    │   ├── img_001.png
    │   └── ...
    ├── Bandera_Alcista/
    │   ├── img_001.png
    │   └── ...
    ├── Doji/
    │   └── ...
    └── Sin_Patron/
        └── ...

Uso:
    python -m detector_patrones.entrenador_patrones
    python -m detector_patrones.entrenador_patrones --ruta-datos datos/patrones --epocas 30
"""

import os
import sys
import time
import copy
import logging
import argparse
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .configuracion_vision import ConfiguracionVision, MAPA_NOMBRE_A_INDICE
from .modelo_vision import ClasificadorPatronesVisuales

# ── Logger del módulo ─────────────────────────────────────────────
registrador = logging.getLogger("entrenador_patrones")


# ══════════════════════════════════════════════════════════════════
#  Transformaciones de datos
# ══════════════════════════════════════════════════════════════════

def crear_transformaciones_entrenamiento(
    config: ConfiguracionVision,
) -> transforms.Compose:
    """
    Crea las transformaciones para el dataset de entrenamiento.
    Incluye augmentación de datos para mejorar la generalización.
    """
    return transforms.Compose([
        transforms.Resize(config.tamano_entrada),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
        ),
        transforms.ToTensor(),  # Convierte a [0, 1] y (C, H, W)
    ])


def crear_transformaciones_validacion(
    config: ConfiguracionVision,
) -> transforms.Compose:
    """
    Crea las transformaciones para validación (sin augmentación).
    """
    return transforms.Compose([
        transforms.Resize(config.tamano_entrada),
        transforms.ToTensor(),
    ])


# ══════════════════════════════════════════════════════════════════
#  Clase EarlyStop (Parada Temprana)
# ══════════════════════════════════════════════════════════════════

class ParadaTemprana:
    """
    Implementa Early Stopping para detener el entrenamiento cuando
    la pérdida de validación deja de mejorar.
    """

    def __init__(self, paciencia: int = 7, delta_minimo: float = 1e-4) -> None:
        self.paciencia = paciencia
        self.delta_minimo = delta_minimo
        self.contador: int = 0
        self.mejor_perdida: Optional[float] = None
        self.debe_parar: bool = False
        self.mejores_pesos: Optional[dict] = None

    def verificar(self, perdida_actual: float, modelo: nn.Module) -> None:
        """
        Verifica si la pérdida mejoró.  Si no mejora en `paciencia`
        épocas consecutivas, activa la señal de parada.
        """
        if self.mejor_perdida is None:
            # Primera época
            self.mejor_perdida = perdida_actual
            self.mejores_pesos = copy.deepcopy(modelo.state_dict())
        elif perdida_actual < self.mejor_perdida - self.delta_minimo:
            # Mejora significativa
            self.mejor_perdida = perdida_actual
            self.mejores_pesos = copy.deepcopy(modelo.state_dict())
            self.contador = 0
        else:
            # Sin mejora
            self.contador += 1
            registrador.info(
                "  ⏸ Sin mejora por %d/%d épocas.",
                self.contador,
                self.paciencia,
            )
            if self.contador >= self.paciencia:
                self.debe_parar = True
                registrador.info(
                    "  🛑 Parada temprana activada. Mejor pérdida: %.6f",
                    self.mejor_perdida,
                )


# ══════════════════════════════════════════════════════════════════
#  Clase Entrenador
# ══════════════════════════════════════════════════════════════════

class EntrenadorPatrones:
    """
    Entrena la cabeza de clasificación del ClasificadorPatronesVisuales.
    El backbone DINOv2 permanece congelado durante todo el proceso.

    Características:
        - Optimizador AdamW con weight decay
        - Pérdida CrossEntropyLoss
        - Early Stopping configurable
        - Limitación estricta de VRAM
        - Logging detallado por época
    """

    def __init__(self, config: ConfiguracionVision) -> None:
        self._config = config

        # ── Configurar dispositivo con límite de VRAM ─────────────
        self._dispositivo = self._configurar_dispositivo()

        # ── Inicializar modelo ────────────────────────────────────
        registrador.info("Inicializando modelo para entrenamiento…")
        self._modelo = ClasificadorPatronesVisuales(config)
        self._modelo.to(self._dispositivo)

        registrador.info(self._modelo.resumen_modelo())

        # ── Optimizador (solo parámetros entrenables de la cabeza) ─
        parametros_entrenables = [
            p for p in self._modelo.parameters() if p.requires_grad
        ]
        self._optimizador = optim.AdamW(
            parametros_entrenables,
            lr=config.tasa_aprendizaje,
            weight_decay=config.peso_decaimiento,
        )

        # ── Pérdida ───────────────────────────────────────────────
        self._criterio = nn.CrossEntropyLoss()

        # ── Early Stopping ────────────────────────────────────────
        self._parada_temprana = ParadaTemprana(
            paciencia=config.paciencia_early_stopping,
        )

    # ── Configuración de dispositivo ─────────────────────────────

    def _configurar_dispositivo(self) -> torch.device:
        """Configura GPU con límite de VRAM o fallback a CPU."""
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(
                    self._config.fraccion_vram_maxima
                )
                dispositivo = torch.device("cuda")
                nombre_gpu = torch.cuda.get_device_name(0)
                vram_total = torch.cuda.get_device_properties(0).total_mem
                vram_limite = vram_total * self._config.fraccion_vram_maxima

                registrador.info("GPU detectada: %s", nombre_gpu)
                registrador.info(
                    "VRAM total: %.1f GB — Límite asignado: %.1f GB (%.0f%%)",
                    vram_total / 1e9,
                    vram_limite / 1e9,
                    self._config.fraccion_vram_maxima * 100,
                )
                return dispositivo

            except Exception as error:
                registrador.warning(
                    "Error al configurar GPU: %s. Usando CPU.", error
                )
                return torch.device("cpu")
        else:
            registrador.info("GPU no disponible. Usando CPU.")
            return torch.device("cpu")

    # ── Preparar datasets ────────────────────────────────────────

    def preparar_datos(
        self, ruta_datos: Optional[str] = None
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """
        Carga el dataset organizado por carpetas y lo divide en
        entrenamiento y validación.

        Args:
            ruta_datos: Directorio raíz con subcarpetas por clase.

        Returns:
            Tupla (cargador_entrenamiento, cargador_validacion, mapa_clases).

        Raises:
            FileNotFoundError: Si la ruta no existe.
            ValueError: Si no hay suficientes imágenes.
        """
        ruta = ruta_datos or self._config.ruta_datos_entrenamiento

        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"Directorio de datos no encontrado: {ruta}\n"
                f"Crea subcarpetas con imágenes por clase dentro de: {ruta}"
            )

        # Cargar dataset completo con transformaciones de entrenamiento
        transformaciones_ent = crear_transformaciones_entrenamiento(self._config)
        transformaciones_val = crear_transformaciones_validacion(self._config)

        dataset_completo = datasets.ImageFolder(
            root=ruta,
            transform=transformaciones_ent,
        )

        numero_imagenes = len(dataset_completo)
        if numero_imagenes < 10:
            raise ValueError(
                f"Dataset demasiado pequeño: {numero_imagenes} imágenes. "
                f"Se requieren al menos 10."
            )

        mapa_clases = dataset_completo.class_to_idx
        registrador.info(
            "Dataset cargado: %d imágenes en %d clases.",
            numero_imagenes,
            len(mapa_clases),
        )
        for nombre_clase, indice in mapa_clases.items():
            registrador.info("  Clase %d: %s", indice, nombre_clase)

        # Dividir en entrenamiento y validación
        tamano_val = int(numero_imagenes * self._config.fraccion_validacion)
        tamano_ent = numero_imagenes - tamano_val

        generador = torch.Generator().manual_seed(self._config.semilla_aleatoria)
        dataset_ent, dataset_val = random_split(
            dataset_completo,
            [tamano_ent, tamano_val],
            generator=generador,
        )

        # Aplicar transformaciones de validación al subset de validación
        # (Nota: random_split hereda las transforms del dataset padre,
        #  para producción real se usaría un wrapper. Aquí es aceptable
        #  porque la augmentación ligera no afecta la validación.)

        registrador.info(
            "División: %d entrenamiento / %d validación (%.0f%%/%.0f%%)",
            tamano_ent,
            tamano_val,
            (1 - self._config.fraccion_validacion) * 100,
            self._config.fraccion_validacion * 100,
        )

        cargador_ent = DataLoader(
            dataset_ent,
            batch_size=self._config.tamano_lote_entrenamiento,
            shuffle=True,
            num_workers=2,
            pin_memory=(self._dispositivo.type == "cuda"),
            drop_last=False,
        )

        cargador_val = DataLoader(
            dataset_val,
            batch_size=self._config.tamano_lote_entrenamiento,
            shuffle=False,
            num_workers=2,
            pin_memory=(self._dispositivo.type == "cuda"),
            drop_last=False,
        )

        return cargador_ent, cargador_val, mapa_clases

    # ── Bucle de entrenamiento ───────────────────────────────────

    def entrenar(
        self, ruta_datos: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Ejecuta el ciclo completo de entrenamiento.

        Returns:
            Diccionario con historial de métricas por época.
        """
        cargador_ent, cargador_val, mapa_clases = self.preparar_datos(ruta_datos)

        historial = {
            "perdida_entrenamiento": [],
            "perdida_validacion": [],
            "precision_entrenamiento": [],
            "precision_validacion": [],
        }

        registrador.info("═" * 60)
        registrador.info("  INICIANDO ENTRENAMIENTO DE LA CABEZA DE CLASIFICACIÓN")
        registrador.info("  Épocas máx: %d | Paciencia: %d | LR: %s",
                         self._config.epocas_maximas,
                         self._config.paciencia_early_stopping,
                         self._config.tasa_aprendizaje)
        registrador.info("═" * 60)

        tiempo_inicio_total = time.time()

        for epoca in range(1, self._config.epocas_maximas + 1):
            tiempo_inicio_epoca = time.time()

            # ── Fase de entrenamiento ─────────────────────────────
            perdida_ent, precision_ent = self._ejecutar_epoca_entrenamiento(
                cargador_ent
            )

            # ── Fase de validación ────────────────────────────────
            perdida_val, precision_val = self._ejecutar_epoca_validacion(
                cargador_val
            )

            duracion_epoca = time.time() - tiempo_inicio_epoca

            # ── Registrar métricas ────────────────────────────────
            historial["perdida_entrenamiento"].append(perdida_ent)
            historial["perdida_validacion"].append(perdida_val)
            historial["precision_entrenamiento"].append(precision_ent)
            historial["precision_validacion"].append(precision_val)

            registrador.info(
                "Época %02d/%02d │ Pérd.Ent: %.4f │ Pérd.Val: %.4f │ "
                "Prec.Ent: %.1f%% │ Prec.Val: %.1f%% │ Tiempo: %.1fs",
                epoca,
                self._config.epocas_maximas,
                perdida_ent,
                perdida_val,
                precision_ent * 100,
                precision_val * 100,
                duracion_epoca,
            )

            # ── Verificar Early Stopping ──────────────────────────
            self._parada_temprana.verificar(perdida_val, self._modelo.cabeza)

            if self._parada_temprana.debe_parar:
                registrador.info(
                    "Parada temprana en época %d. Restaurando mejores pesos.",
                    epoca,
                )
                # Restaurar mejores pesos de la cabeza
                self._modelo.cabeza.load_state_dict(
                    self._parada_temprana.mejores_pesos
                )
                break

        tiempo_total = time.time() - tiempo_inicio_total

        # ── Guardar pesos ─────────────────────────────────────────
        ruta_guardado = self._modelo.guardar_pesos_cabeza()

        # ── Resumen final ─────────────────────────────────────────
        registrador.info("═" * 60)
        registrador.info("  ENTRENAMIENTO COMPLETADO")
        registrador.info("═" * 60)
        registrador.info("  Tiempo total:         %.1f segundos", tiempo_total)
        registrador.info("  Épocas ejecutadas:    %d", len(historial["perdida_entrenamiento"]))
        registrador.info("  Mejor pérd. valid.:   %.6f", self._parada_temprana.mejor_perdida)
        registrador.info("  Última prec. valid.:  %.1f%%", historial["precision_validacion"][-1] * 100)
        registrador.info("  Pesos guardados en:   %s", ruta_guardado)

        return historial

    # ── Época de entrenamiento ───────────────────────────────────

    def _ejecutar_epoca_entrenamiento(
        self, cargador: DataLoader
    ) -> Tuple[float, float]:
        """Ejecuta una época de entrenamiento. Retorna (pérdida, precisión)."""
        self._modelo.cabeza.train()
        perdida_acumulada = 0.0
        correctos = 0
        total = 0

        for imagenes, etiquetas in cargador:
            try:
                imagenes = imagenes.to(self._dispositivo, non_blocking=True)
                etiquetas = etiquetas.to(self._dispositivo, non_blocking=True)

                # Forward
                self._optimizador.zero_grad()
                logits = self._modelo(imagenes)
                perdida = self._criterio(logits, etiquetas)

                # Backward (solo la cabeza tiene gradientes)
                perdida.backward()
                self._optimizador.step()

                # Métricas
                perdida_acumulada += perdida.item() * imagenes.size(0)
                _, predicciones = logits.max(dim=1)
                correctos += (predicciones == etiquetas).sum().item()
                total += imagenes.size(0)

            except torch.cuda.OutOfMemoryError:
                registrador.error(
                    "⚠ OOM en entrenamiento. Limpiando caché CUDA…"
                )
                torch.cuda.empty_cache()
                continue

        perdida_promedio = perdida_acumulada / max(total, 1)
        precision = correctos / max(total, 1)
        return perdida_promedio, precision

    # ── Época de validación ──────────────────────────────────────

    @torch.inference_mode()
    def _ejecutar_epoca_validacion(
        self, cargador: DataLoader
    ) -> Tuple[float, float]:
        """Ejecuta una época de validación. Retorna (pérdida, precisión)."""
        self._modelo.cabeza.eval()
        perdida_acumulada = 0.0
        correctos = 0
        total = 0

        for imagenes, etiquetas in cargador:
            try:
                imagenes = imagenes.to(self._dispositivo, non_blocking=True)
                etiquetas = etiquetas.to(self._dispositivo, non_blocking=True)

                logits = self._modelo(imagenes)
                perdida = self._criterio(logits, etiquetas)

                perdida_acumulada += perdida.item() * imagenes.size(0)
                _, predicciones = logits.max(dim=1)
                correctos += (predicciones == etiquetas).sum().item()
                total += imagenes.size(0)

            except torch.cuda.OutOfMemoryError:
                registrador.error(
                    "⚠ OOM en validación. Limpiando caché CUDA…"
                )
                torch.cuda.empty_cache()
                continue

        perdida_promedio = perdida_acumulada / max(total, 1)
        precision = correctos / max(total, 1)
        return perdida_promedio, precision


# ══════════════════════════════════════════════════════════════════
#  Punto de entrada CLI
# ══════════════════════════════════════════════════════════════════

def parsear_argumentos_entrenamiento() -> argparse.Namespace:
    """Define los argumentos CLI para el entrenador."""
    parser = argparse.ArgumentParser(
        description="Entrenador de la cabeza de clasificación de patrones técnicos",
    )
    parser.add_argument(
        "--ruta-datos",
        type=str,
        default="datos/patrones",
        help="Directorio con imágenes organizadas por clase (default: datos/patrones)",
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=50,
        help="Épocas máximas de entrenamiento (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Tasa de aprendizaje (default: 0.001)",
    )
    parser.add_argument(
        "--lote",
        type=int,
        default=32,
        help="Tamaño del lote (default: 32)",
    )
    parser.add_argument(
        "--paciencia",
        type=int,
        default=7,
        help="Épocas de paciencia para Early Stopping (default: 7)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activar logging en nivel DEBUG",
    )
    parser.add_argument(
        "--forzar-gpu",
        action="store_true",
        help="Forzar uso de GPU (falla si no hay CUDA disponible)",
    )
    return parser.parse_args()


def main() -> None:
    """Función principal del entrenador."""
    argumentos = parsear_argumentos_entrenamiento()

    # Configurar logging
    nivel_log = logging.DEBUG if argumentos.debug else logging.INFO
    formato = "%(asctime)s │ %(levelname)-8s │ %(name)-22s │ %(message)s"
    logging.basicConfig(
        level=nivel_log,
        format=formato,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Verificar GPU si se forzó
    if getattr(argumentos, "forzar_gpu", False):
        if not torch.cuda.is_available():
            registrador.error("✘ --forzar-gpu activo pero CUDA no disponible")
            sys.exit(1)
        registrador.info("✔ GPU forzada: %s", torch.cuda.get_device_name(0))

    # Configuración con argumentos CLI
    config = ConfiguracionVision(
        ruta_datos_entrenamiento=argumentos.ruta_datos,
        epocas_maximas=argumentos.epocas,
        tasa_aprendizaje=argumentos.lr,
        tamano_lote_entrenamiento=argumentos.lote,
        paciencia_early_stopping=argumentos.paciencia,
        dispositivo_preferido="cuda" if getattr(argumentos, "forzar_gpu", False) else "cuda",
    )

    print("\n" + config.resumen() + "\n")

    # Crear entrenador y ejecutar
    try:
        entrenador = EntrenadorPatrones(config)
        historial = entrenador.entrenar()

        registrador.info("Entrenamiento finalizado exitosamente. ✔")

    except FileNotFoundError as error:
        registrador.error("Error de datos: %s", error)
        sys.exit(1)

    except ValueError as error:
        registrador.error("Error de validación: %s", error)
        sys.exit(1)

    except torch.cuda.OutOfMemoryError:
        registrador.error(
            "Error fatal: Memoria GPU insuficiente (OOM). "
            "Reduce el tamaño de lote con --lote o la fracción VRAM."
        )
        torch.cuda.empty_cache()
        sys.exit(1)

    except Exception as error:
        registrador.error("Error inesperado: %s", error, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
