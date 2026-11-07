"""Contagem de ovos por palheta e indicadores entomológicos (IPO e IDO)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classificador import area_referencia
from .processamento import Parametros, ler_imagem, segmentar

# Uma única detecção falsa transforma uma palheta negativa em positiva e altera o IPO.
# Por isso, contagens baixas (1 ou 2 ovos) são sinalizadas para conferência visual.
LIMITE_CONFERENCIA = 2


@dataclass
class ResultadoImagem:
    arquivo: str
    componentes: list
    rotulos: np.ndarray
    limiar: float
    area_referencia: float
    tempo_s: float

    def _da_classe(self, classe):
        return [c for c in self.componentes if c.classe == classe]

    @property
    def ovos_isolados(self) -> int:
        return len(self._da_classe("ovo"))

    @property
    def aglomerados(self) -> int:
        return len(self._da_classe("aglomerado"))

    @property
    def ovos_em_aglomerados(self) -> int:
        return sum(c.ovos_estimados for c in self._da_classe("aglomerado"))

    @property
    def detritos(self) -> int:
        return len(self._da_classe("detrito"))

    @property
    def total_ovos(self) -> int:
        return self.ovos_isolados + self.ovos_em_aglomerados

    @property
    def positiva(self) -> bool:
        return self.total_ovos > 0

    @property
    def conferir(self) -> bool:
        return 0 < self.total_ovos <= LIMITE_CONFERENCIA

    @property
    def situacao(self) -> str:
        if not self.positiva:
            return "negativa"
        return "positiva (conferir)" if self.conferir else "positiva"


def contar(img: np.ndarray, pacote: dict, arquivo: str = "", params: Parametros | None = None) -> ResultadoImagem:
    inicio = time.perf_counter()
    params = params or Parametros(**pacote["parametros"])
    seg = segmentar(img, params)
    ref = float(pacote["area_referencia"])
    if seg.componentes:
        X = np.vstack([c.caracteristicas for c in seg.componentes])
        classes = pacote["modelo"].predict(X)
        ref = area_referencia([c.area for c, k in zip(seg.componentes, classes) if k == "ovo"], ref)
        fator = float(pacote.get("fator_aglomerado", 1.0))
        for c, k in zip(seg.componentes, classes):
            c.classe = str(k)
            if k == "ovo":
                c.ovos_estimados = 1
            elif k == "aglomerado":
                c.ovos_estimados = max(2, int(round(fator * c.area / ref)))
            else:
                c.ovos_estimados = 0
    return ResultadoImagem(arquivo=arquivo, componentes=seg.componentes, rotulos=seg.rotulos, limiar=seg.limiar,
                           area_referencia=ref, tempo_s=time.perf_counter() - inicio)


def contar_arquivo(caminho, pacote: dict):
    img = ler_imagem(caminho)
    return contar(img, pacote, arquivo=Path(caminho).name), img


@dataclass
class Indicadores:
    examinadas: int
    positivas: int
    total_ovos: int

    @property
    def ipo(self) -> float:
        """Índice de Positividade de Ovitrampas (%) = positivas / examinadas x 100."""
        return 100.0 * self.positivas / self.examinadas if self.examinadas else 0.0

    @property
    def ido(self) -> float:
        """Índice de Densidade de Ovos = total de ovos / palhetas positivas."""
        return self.total_ovos / self.positivas if self.positivas else 0.0


def calcular_indicadores(totais) -> Indicadores:
    totais = [int(t) for t in totais]
    return Indicadores(examinadas=len(totais), positivas=sum(1 for t in totais if t > 0), total_ovos=sum(totais))
