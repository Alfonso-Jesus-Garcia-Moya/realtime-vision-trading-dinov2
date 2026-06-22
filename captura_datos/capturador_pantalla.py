"""
Módulo Capturador de Pantalla.
Se encarga de abrir el navegador, navegar a Yahoo Finance,
localizar el elemento del gráfico y devolver la captura
como bytes PNG **en memoria** (nunca se guarda a disco).
"""

import io
import time
import logging
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ServicioCromo
from selenium.webdriver.chrome.options import Options as OpcionesCromo
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as CE
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
)
from webdriver_manager.chrome import ChromeDriverManager

from .configuracion import Configuracion

# ── Logger del módulo ─────────────────────────────────────────────
registrador = logging.getLogger("capturador_pantalla")


class CapturadorPantalla:
    """
    Abre un navegador Chrome (headless o visible), navega al gráfico
    de Yahoo Finance y provee un método para obtener la captura del
    elemento gráfico como bytes PNG en memoria.
    """

    def __init__(self, config: Configuracion) -> None:
        self._config = config
        self._navegador: Optional[webdriver.Chrome] = None
        self._elemento_grafico = None
        self._capturas_realizadas: int = 0

    # ── Ciclo de vida ─────────────────────────────────────────────

    def iniciar(self) -> None:
        """Levanta el navegador y navega al gráfico."""
        registrador.info("Iniciando navegador Chrome…")
        opciones = self._construir_opciones()
        servicio = ServicioCromo(ChromeDriverManager().install())
        self._navegador = webdriver.Chrome(service=servicio, options=opciones)
        self._navegador.set_window_size(
            self._config.ancho_ventana,
            self._config.alto_ventana,
        )
        self._navegar_al_grafico()
        self._localizar_elemento_grafico()
        registrador.info("Capturador listo ✔")

    def detener(self) -> None:
        """Cierra el navegador limpiamente."""
        if self._navegador:
            registrador.info("Cerrando navegador…")
            self._navegador.quit()
            self._navegador = None
            self._elemento_grafico = None
            registrador.info(
                "Navegador cerrado. Capturas realizadas: %d",
                self._capturas_realizadas,
            )

    # ── Captura ───────────────────────────────────────────────────

    def capturar_grafico_bytes(self) -> Optional[bytes]:
        """
        Toma una captura del elemento gráfico y la devuelve como
        bytes PNG **en memoria**.  Retorna None si falla.
        """
        if self._elemento_grafico is None:
            registrador.warning("No hay elemento gráfico localizado.")
            return None

        try:
            datos_png: bytes = self._elemento_grafico.screenshot_as_png
            self._capturas_realizadas += 1
            registrador.debug(
                "Captura #%d obtenida (%d bytes)",
                self._capturas_realizadas,
                len(datos_png),
            )
            return datos_png

        except WebDriverException as error:
            registrador.error("Error al capturar: %s", error)
            # Intenta re-localizar el elemento (puede haberse refrescado)
            self._localizar_elemento_grafico()
            return None

    @property
    def capturas_realizadas(self) -> int:
        return self._capturas_realizadas

    # ── Métodos internos ──────────────────────────────────────────

    def _construir_opciones(self) -> OpcionesCromo:
        """Genera las opciones de Chrome según la configuración."""
        opciones = OpcionesCromo()

        if self._config.navegador_sin_cabeza:
            opciones.add_argument("--headless=new")

        opciones.add_argument("--no-sandbox")
        opciones.add_argument("--disable-dev-shm-usage")
        opciones.add_argument("--disable-gpu")
        opciones.add_argument("--disable-extensions")
        opciones.add_argument("--disable-infobars")
        opciones.add_argument(
            f"--window-size={self._config.ancho_ventana},"
            f"{self._config.alto_ventana}"
        )
        # Simular un user-agent real para evitar bloqueos
        opciones.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
        return opciones

    def _navegar_al_grafico(self) -> None:
        """Carga la URL y espera a que la página esté lista."""
        registrador.info("Navegando a %s", self._config.url_grafico)
        self._navegador.get(self._config.url_grafico)

        # Esperar a que el body exista (carga básica)
        WebDriverWait(
            self._navegador,
            self._config.tiempo_espera_carga_seg,
        ).until(CE.presence_of_element_located((By.TAG_NAME, "body")))

        # Cerrar posibles diálogos de cookies / consentimiento
        self._cerrar_dialogos()

        # Seleccionar vista YTD por defecto
        self._seleccionar_ytd()

        # Tiempo extra para que el JS renderice el chart
        registrador.info("Esperando renderizado del gráfico…")
        time.sleep(5)

    # ── Mapeo de rangos a IDs de botón de Yahoo Finance ───────────
    RANGOS_VALIDOS = {
        "1d": "tab-1d", "5d": "tab-5d", "1m": "tab-1m",
        "3m": "tab-3m", "6m": "tab-6m", "ytd": "tab-YTD",
        "1y": "tab-1y", "5y": "tab-5y", "max": "tab-Max",
    }

    def _seleccionar_ytd(self) -> None:
        """Hace click en el botón YTD del chart de Yahoo Finance."""
        self.seleccionar_rango("ytd")

    def seleccionar_rango(self, rango: str) -> bool:
        """
        Hace click en un botón de rango temporal del chart.
        rango: '1d','5d','1m','3m','6m','ytd','1y','5y','max'
        Retorna True si se seleccionó correctamente.
        """
        rango_lower = rango.lower().strip()
        id_boton = self.RANGOS_VALIDOS.get(rango_lower)
        if not id_boton:
            registrador.warning("Rango no válido: '%s'. Válidos: %s", rango, list(self.RANGOS_VALIDOS.keys()))
            return False

        selectores = [
            f"button#{id_boton}",
            f"button[id='{id_boton}']",
            f"button[title='{rango.upper()}']",
        ]
        for selector in selectores:
            try:
                boton = WebDriverWait(self._navegador, 5).until(
                    CE.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                boton.click()
                registrador.info("✔ Rango '%s' seleccionado con: %s", rango_lower, selector)
                time.sleep(3)  # Esperar a que el chart se actualice
                return True
            except (TimeoutException, NoSuchElementException, WebDriverException):
                continue
        registrador.warning("No se pudo seleccionar rango '%s'.", rango)
        return False

    def seleccionar_intervalo(self, intervalo: str) -> bool:
        """
        Abre el dropdown de intervalo de Yahoo Finance y selecciona
        el intervalo deseado (ej: '1 min','5 min','15 min','1 hour','1 day').
        """
        try:
            # 1. Buscar y hacer click en el botón del dropdown de intervalo
            selectores_btn = [
                "button[aria-label*='min']",
                "button[aria-label*='hour']",
                "button[aria-label*='day']",
                "button[aria-label*='week']",
                "[data-testid='chart-interval-selector'] button",
            ]
            boton_intervalo = None
            for sel in selectores_btn:
                try:
                    boton_intervalo = WebDriverWait(self._navegador, 3).until(
                        CE.element_to_be_clickable((By.CSS_SELECTOR, sel))
                    )
                    break
                except (TimeoutException, NoSuchElementException):
                    continue

            if boton_intervalo is None:
                registrador.warning("No se encontró el botón de intervalo.")
                return False

            boton_intervalo.click()
            time.sleep(1)

            # 2. Buscar la opción del intervalo en el menú abierto
            # Las opciones del listbox son items con role="option" o similares
            opciones_selectores = [
                f"div[role='option'] span",
                f"li[role='option']",
                f"div[role='listbox'] button",
                f"div[popover] button",
                f"div[popover] div[role='option']",
            ]

            intervalo_lower = intervalo.lower().strip()
            for sel_opcion in opciones_selectores:
                try:
                    opciones = self._navegador.find_elements(By.CSS_SELECTOR, sel_opcion)
                    for opcion in opciones:
                        texto = opcion.text.strip().lower()
                        if texto == intervalo_lower or intervalo_lower in texto:
                            opcion.click()
                            registrador.info("✔ Intervalo '%s' seleccionado.", intervalo)
                            time.sleep(3)
                            return True
                except (NoSuchElementException, WebDriverException):
                    continue

            # Si no se encontró, cerrar el dropdown haciendo click fuera
            try:
                body = self._navegador.find_element(By.TAG_NAME, "body")
                body.click()
            except Exception:
                pass

            registrador.warning("No se encontró la opción de intervalo '%s'.", intervalo)
            return False

        except Exception as error:
            registrador.error("Error al cambiar intervalo: %s", error)
            return False

    def _cerrar_dialogos(self) -> None:
        """Intenta cerrar banners de cookies u otros overlays."""
        selectores_boton_aceptar = [
            "button[name='agree']",
            "button.accept-all",
            "[data-testid='consent-accept']",
            ".consent-overlay button.primary",
            "button#scroll-down-btn",
            ".dialog button",
        ]
        for selector in selectores_boton_aceptar:
            try:
                boton = self._navegador.find_element(By.CSS_SELECTOR, selector)
                boton.click()
                registrador.info("Diálogo cerrado con selector: %s", selector)
                time.sleep(1)
                break
            except (NoSuchElementException, WebDriverException):
                continue

    def _localizar_elemento_grafico(self) -> None:
        """Busca el elemento canvas del gráfico usando los selectores configurados."""
        for selector in self._config.selectores_grafico:
            try:
                registrador.debug("Probando selector: %s", selector)
                elemento = WebDriverWait(
                    self._navegador,
                    self._config.tiempo_espera_grafico_seg,
                ).until(
                    CE.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                # Verificar que tenga dimensiones visibles
                tamano = elemento.size
                if tamano["width"] > 50 and tamano["height"] > 50:
                    self._elemento_grafico = elemento
                    registrador.info(
                        "Elemento gráfico encontrado con '%s' "
                        "(%dx%d px)",
                        selector,
                        tamano["width"],
                        tamano["height"],
                    )
                    return
            except (TimeoutException, NoSuchElementException):
                registrador.debug("Selector '%s' no encontrado.", selector)
                continue

        # Si ningún selector funcionó, captura la página completa como fallback
        registrador.warning(
            "No se encontró el canvas del gráfico. "
            "Se usará captura de página completa como respaldo."
        )
        self._elemento_grafico = self._navegador.find_element(By.TAG_NAME, "body")

    # ── Context manager ───────────────────────────────────────────

    def __enter__(self):
        self.iniciar()
        return self

    def __exit__(self, tipo_exc, valor_exc, traza):
        self.detener()
