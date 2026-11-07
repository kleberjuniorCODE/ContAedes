"""Saídas do ContAedes: imagem anotada e planilhas CSV (abrem direto no Excel)."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .processamento import salvar_imagem

CORES = {"ovo": (40, 190, 40), "aglomerado": (0, 160, 255), "detrito": (50, 50, 220)}  # BGR


def fmt(valor: float, casas: int = 2) -> str:
    """Número no padrão brasileiro (vírgula decimal)."""
    return f"{valor:.{casas}f}".replace(".", ",")


def anotar(img: np.ndarray, resultado, total_gabarito: int | None = None) -> np.ndarray:
    """Verde = ovo isolado, laranja = aglomerado (com o nº estimado), vermelho = detrito descartado."""
    saida = img.copy()
    for c in resultado.componentes:
        if not c.classe:
            continue
        cv2.drawContours(saida, [c.contorno], -1, CORES[c.classe], 1 if c.classe == "detrito" else 2, cv2.LINE_AA)
        if c.classe == "aglomerado":
            x, y, w, _ = c.caixa
            cv2.putText(saida, str(c.ovos_estimados), (x + w + 2, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 90, 200), 1, cv2.LINE_AA)
    largura = saida.shape[1]
    escala = max(0.35, min(0.6, largura / 2600))
    barra = np.full((int(70 * escala) + 8, largura, 3), 35, np.uint8)
    texto = (f"Ovos estimados: {resultado.total_ovos}  (isolados: {resultado.ovos_isolados} | "
             f"em {resultado.aglomerados} aglomerados: {resultado.ovos_em_aglomerados})  |  "
             f"detritos descartados: {resultado.detritos}")
    if total_gabarito is not None:
        texto += f"  |  gabarito: {total_gabarito}"
    cv2.putText(barra, texto, (10, barra.shape[0] - int(22 * escala) - 4), cv2.FONT_HERSHEY_SIMPLEX, escala,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([barra, saida])


def salvar_componentes_csv(resultado, caminho) -> Path:
    caminho = Path(caminho)
    with open(caminho, "w", newline="", encoding="utf-8-sig") as arq:
        w = csv.writer(arq, delimiter=";")
        w.writerow(["id", "classe", "ovos_estimados", "x", "y", "area_px", "alongamento", "espessura_px",
                    "intensidade_media"])
        for i, c in enumerate(resultado.componentes, 1):
            v = c.caracteristicas
            w.writerow([i, c.classe, c.ovos_estimados, fmt(c.centro[0], 1), fmt(c.centro[1], 1), int(v[0]),
                        fmt(v[5]), fmt(v[8]), fmt(v[9], 1)])
    return caminho


def salvar_resultado(resultado, img, pasta, total_gabarito: int | None = None) -> dict:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    base = Path(resultado.arquivo or "imagem").stem
    caminho_img = pasta / f"{base}_anotada.png"
    salvar_imagem(caminho_img, anotar(img, resultado, total_gabarito))
    return {"imagem": caminho_img, "csv": salvar_componentes_csv(resultado, pasta / f"{base}_componentes.csv")}


def salvar_lote_csv(linhas: list, indicadores, caminho) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    colunas = ["arquivo", "ovos", "isolados", "aglomerados", "ovos_em_aglomerados", "detritos", "situacao",
               "gabarito", "tempo_s", "observacao"]
    with open(caminho, "w", newline="", encoding="utf-8-sig") as arq:
        w = csv.writer(arq, delimiter=";")
        w.writerow(colunas)
        for linha in linhas:
            w.writerow([linha.get(c, "") for c in colunas])
        w.writerow([])
        w.writerow(["Palhetas examinadas", indicadores.examinadas])
        w.writerow(["Palhetas positivas", indicadores.positivas])
        w.writerow(["Total de ovos", indicadores.total_ovos])
        w.writerow(["IPO (%)", fmt(indicadores.ipo, 1)])
        w.writerow(["IDO (ovos por palheta positiva)", fmt(indicadores.ido, 1)])
    return caminho
