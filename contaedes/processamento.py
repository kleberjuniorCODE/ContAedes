"""Etapas de processamento digital de imagens (PDI) do ContAedes.

Fluxo: leitura -> tons de cinza -> realce pela transformada black-hat ->
suavização gaussiana -> limiarização de Otsu (com piso mínimo) -> abertura
morfológica -> rotulação de componentes conexos -> extração de
características de forma e de intensidade de cada componente.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

NOMES_CARACTERISTICAS = [
    "area",
    "perimetro",
    "circularidade",
    "eixo_maior",
    "eixo_menor",
    "alongamento",
    "solidez",
    "extensao",
    "espessura",
    "intensidade_media",
    "intensidade_desvio",
    "realce_medio",
    "saturacao_media",
]

EXTENSOES_ACEITAS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ErroImagem(Exception):
    """Arquivo inexistente, corrompido ou em formato não suportado."""


@dataclass
class Parametros:
    kernel_realce: int = 45      # lado (px) do elemento estruturante do black-hat
    suavizacao: int = 3          # janela do filtro gaussiano
    limiar_minimo: float = 35.0  # piso do limiar de Otsu (evita segmentar textura em palheta vazia)
    area_minima: int = 12        # componentes menores que isso são tratados como ruído


@dataclass
class Componente:
    rotulo: int
    caixa: tuple
    centro: tuple
    contorno: np.ndarray
    caracteristicas: np.ndarray
    classe: str = ""
    ovos_estimados: int = 0

    @property
    def area(self) -> float:
        return float(self.caracteristicas[0])


@dataclass
class Segmentacao:
    mascara: np.ndarray
    rotulos: np.ndarray
    realce: np.ndarray
    limiar: float
    componentes: list = field(default_factory=list)


def ler_imagem(caminho) -> np.ndarray:
    """Lê a imagem em BGR. Usa imdecode para aceitar caminhos com acento no Windows."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ErroImagem(f"Arquivo não encontrado: {caminho}")
    if caminho.suffix.lower() not in EXTENSOES_ACEITAS:
        raise ErroImagem(f"Formato não suportado ({caminho.suffix or 'sem extensão'}). Use PNG, JPG, BMP ou TIFF.")
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None
    if img is None:
        raise ErroImagem(f"Não foi possível abrir a imagem (arquivo corrompido?): {caminho.name}")
    return img


def salvar_imagem(caminho, img) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(caminho.suffix or ".png", img)
    if not ok:
        raise ErroImagem(f"Não foi possível salvar {caminho}")
    buffer.tofile(str(caminho))


def realcar(img_bgr: np.ndarray, params: Parametros):
    """Black-hat = fechamento - imagem: destaca objetos escuros menores que o elemento estruturante."""
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lado = params.kernel_realce | 1
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
    realce = cv2.morphologyEx(cinza, cv2.MORPH_BLACKHAT, elemento)
    return cinza, realce


def _caracteristicas(sel, x, y, w, h, area, cinza, realce, saturacao):
    mascara_roi = sel.astype(np.uint8) * 255
    contornos, _ = cv2.findContours(mascara_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contorno = max(contornos, key=len)
    perimetro = cv2.arcLength(contorno, True)

    # elipse equivalente pelos momentos centrais de segunda ordem
    m = cv2.moments(mascara_roi, binaryImage=True)
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    comum = math.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    eixo_maior = 4 * math.sqrt(max((mu20 + mu02) / 2 + comum, 1e-6))
    eixo_menor = 4 * math.sqrt(max((mu20 + mu02) / 2 - comum, 1e-6))

    area_casco = cv2.contourArea(cv2.convexHull(contorno))
    solidez = cv2.contourArea(contorno) / area_casco if area_casco > 0 else 1.0
    circularidade = min(4 * math.pi * area / perimetro ** 2, 1.5) if perimetro > 0 else 1.0

    tons = cinza[y:y + h, x:x + w][sel]
    vetor = np.array(
        [
            area,
            perimetro,
            circularidade,
            eixo_maior,
            eixo_menor,
            eixo_maior / eixo_menor,
            solidez,
            area / float(w * h),
            area / eixo_maior,
            tons.mean(),
            tons.std(),
            realce[y:y + h, x:x + w][sel].mean(),
            saturacao[y:y + h, x:x + w][sel].mean(),
        ],
        dtype=np.float64,
    )
    return contorno + np.array([[[x, y]]], dtype=np.int32), vetor


def segmentar(img_bgr: np.ndarray, params: Parametros | None = None) -> Segmentacao:
    params = params or Parametros()
    cinza, realce = realcar(img_bgr, params)
    lado = params.suavizacao | 1
    suave = cv2.GaussianBlur(realce, (lado, lado), 0)
    otsu, _ = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    limiar = max(float(otsu), params.limiar_minimo)
    _, mascara = cv2.threshold(suave, limiar, 255, cv2.THRESH_BINARY)
    abertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, abertura)

    n, rotulos, stats, centroides = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    saturacao = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1]
    componentes = []
    for r in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[r])
        sel = rotulos[y:y + h, x:x + w] == r
        if area < params.area_minima:
            mascara[y:y + h, x:x + w][sel] = 0
            continue
        contorno, vetor = _caracteristicas(sel, x, y, w, h, area, cinza, realce, saturacao)
        componentes.append(
            Componente(
                rotulo=r,
                caixa=(x, y, w, h),
                centro=(float(centroides[r][0]), float(centroides[r][1])),
                contorno=contorno,
                caracteristicas=vetor,
            )
        )
    return Segmentacao(mascara=mascara, rotulos=rotulos, realce=realce, limiar=limiar, componentes=componentes)
