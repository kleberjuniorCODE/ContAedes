"""Gerador de palhetas sintéticas com gabarito.

O grupo não teve acesso a palhetas reais de ovitrampa durante a prova de
conceito (risco R01 e solicitação de mudança SM-01 do documento). Para não
travar o projeto, as imagens de treino e de teste são geradas por computador:
um fundo que imita a face rugosa do Eucatex, ovos escuros e alongados
(isolados ou em grupos) e sujeiras de vários formatos. Cada imagem tem um
gabarito em JSON com a posição de cada ovo, o que permite medir o erro da
contagem automática.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .processamento import salvar_imagem

LARGURA = 1600
ALTURA = 480
COR_FUNDO = (118, 152, 186)  # BGR, marrom-claro do Eucatex
COR_OVO = (28, 30, 34)       # BGR, ovo seco é quase preto
_DESLOC = 4                  # desenho com precisão de 1/16 de pixel
_ESC = 1 << _DESLOC
COR_OVO_CLARO = (52, 62, 78)  # ovo pouco melanizado ou já eclodido
TIPOS_DETRITO = ("grao", "mancha", "fibra", "fragmento", "bolota", "semente", "grao_claro")
_PROB_DETRITO = (0.25, 0.13, 0.17, 0.13, 0.10, 0.12, 0.10)


@dataclass
class Ovo:
    x: float
    y: float
    comprimento: float
    largura: float
    angulo: float
    aglomerado: int = -1  # -1 indica ovo isolado


@dataclass
class Detrito:
    tipo: str
    x: float
    y: float


@dataclass
class Gabarito:
    arquivo: str
    semente: int
    largura: int
    altura: int
    ovos: list = field(default_factory=list)
    detritos: list = field(default_factory=list)

    @property
    def total_ovos(self) -> int:
        return len(self.ovos)

    def para_dict(self) -> dict:
        dados = asdict(self)
        dados["total_ovos"] = self.total_ovos
        return dados

    @classmethod
    def de_dict(cls, dados: dict) -> "Gabarito":
        return cls(
            arquivo=dados["arquivo"],
            semente=dados["semente"],
            largura=dados["largura"],
            altura=dados["altura"],
            ovos=[Ovo(**o) for o in dados["ovos"]],
            detritos=[Detrito(**d) for d in dados["detritos"]],
        )


def _p(valor: float) -> int:
    return int(round(valor * _ESC))


def _elipse(mascara, cx, cy, semi_a, semi_b, angulo):
    cv2.ellipse(mascara, (_p(cx), _p(cy)), (_p(semi_a), _p(semi_b)), angulo, 0, 360, 255, -1, cv2.LINE_AA, _DESLOC)


def _circulo(mascara, cx, cy, raio):
    cv2.circle(mascara, (_p(cx), _p(cy)), _p(raio), 255, -1, cv2.LINE_AA, _DESLOC)


def _desenhar(img, cx, cy, raio, desenho, cor, opacidade=1.0):
    """Desenha uma forma numa janela ao redor de (cx, cy) e a compõe sobre a imagem."""
    altura, largura = img.shape[:2]
    x0, y0 = max(int(cx - raio), 0), max(int(cy - raio), 0)
    x1, y1 = min(int(cx + raio) + 2, largura), min(int(cy + raio) + 2, altura)
    if x1 <= x0 or y1 <= y0:
        return
    mascara = np.zeros((y1 - y0, x1 - x0), np.uint8)
    desenho(mascara, cx - x0, cy - y0)
    alfa = (mascara.astype(np.float32) / 255.0 * opacidade)[..., None]
    regiao = img[y0:y1, x0:x1]
    regiao[:] = regiao * (1.0 - alfa) + np.asarray(cor, np.float32) * alfa


def _fundo(rng, altura, largura):
    base = np.asarray(COR_FUNDO, np.float32) + rng.normal(0, 8, 3).astype(np.float32)
    img = np.empty((altura, largura, 3), np.float32)
    img[:] = base
    # variação suave da cor da chapa
    baixa = rng.normal(0, 1, (altura // 32 + 2, largura // 32 + 2)).astype(np.float32)
    img += cv2.resize(baixa, (largura, altura), interpolation=cv2.INTER_CUBIC)[..., None] * 6
    # fibras da face rugosa, quase todas na horizontal
    fibras = np.zeros((altura, largura), np.float32)
    for _ in range(altura * largura // 700):
        x0, y0 = int(rng.integers(0, largura)), int(rng.integers(0, altura))
        ang = rng.normal(0, 0.3)
        comp = int(rng.integers(8, 50))
        x1, y1 = int(x0 + comp * math.cos(ang)), int(y0 + comp * math.sin(ang))
        cv2.line(fibras, (x0, y0), (x1, y1), float(rng.choice([-1.0, 1.0]) * rng.uniform(3, 10)), 1, cv2.LINE_8)
    img += cv2.GaussianBlur(fibras, (0, 0), 0.7)[..., None]
    # iluminação irregular: gradiente + leve vinheta + brilho geral
    gx = np.linspace(-1, 1, largura, dtype=np.float32)[None, :]
    gy = np.linspace(-1, 1, altura, dtype=np.float32)[:, None]
    a, b = rng.uniform(-0.08, 0.08, 2)
    iluminacao = (1.0 + a * gx + b * gy - 0.05 * (gx ** 2 + gy ** 2)) * rng.uniform(0.88, 1.08)
    return img * iluminacao[..., None]


def _novo_ovo(rng, x, y, angulo=None, aglomerado=-1) -> Ovo:
    return Ovo(
        x=float(x),
        y=float(y),
        comprimento=float(rng.uniform(17, 23)),
        largura=float(rng.uniform(5.5, 7.5)),
        angulo=float(rng.uniform(0, 180) if angulo is None else angulo),
        aglomerado=aglomerado,
    )


def _posicao(rng, altura, largura, faixa_y):
    """70% dos ovos ficam numa faixa horizontal (perto da linha d'água), o resto se espalha."""
    margem = 16
    y = rng.normal(faixa_y, altura * 0.12) if rng.random() < 0.7 else rng.uniform(margem, altura - margem)
    return float(rng.uniform(margem, largura - margem)), float(np.clip(y, margem, altura - margem))


def _longe(x, y, ovos, distancia):
    d2 = distancia * distancia
    return all((o.x - x) ** 2 + (o.y - y) ** 2 >= d2 for o in ovos)


def _aglomerado(rng, cx, cy, k, ident):
    ovos = []
    if rng.random() < 0.7:  # ovos lado a lado, como a fêmea costuma depositar
        base = float(rng.uniform(0, 180))
        rad = math.radians(base)
        perp = (-math.sin(rad), math.cos(rad))
        eixo = (math.cos(rad), math.sin(rad))
        passo = rng.uniform(4.6, 6.2)
        for i in range(k):
            d = (i - (k - 1) / 2) * passo
            s = rng.normal(0, 2.5)
            ovos.append(_novo_ovo(rng, cx + perp[0] * d + eixo[0] * s, cy + perp[1] * d + eixo[1] * s,
                                  base + rng.normal(0, 6), ident))
    else:  # amontoado desordenado
        for _ in range(k):
            ovos.append(_novo_ovo(rng, cx + rng.normal(0, 5), cy + rng.normal(0, 5), None, ident))
    return ovos


def _desenhar_ovo(img, rng, ovo: Ovo):
    base = COR_OVO_CLARO if rng.random() < 0.15 else COR_OVO
    cor = np.asarray(base, np.float32) + rng.normal(0, 6, 3)
    _desenhar(img, ovo.x, ovo.y, ovo.comprimento / 2 + 3,
              lambda m, x, y: _elipse(m, x, y, ovo.comprimento / 2, ovo.largura / 2, ovo.angulo), cor)


def _desenhar_detrito(img, rng, tipo, x, y):
    if tipo == "grao":  # partícula escura pequena
        r = rng.uniform(1.5, 3.5)
        _desenhar(img, x, y, r + 2, lambda m, cx, cy: _circulo(m, cx, cy, r), np.full(3, rng.uniform(35, 75)))
    elif tipo == "mancha":  # mancha de matéria orgânica, amarronzada e irregular
        cor = np.array([55, 75, 100], np.float32) + rng.normal(0, 8, 3)
        circulos = [(rng.normal(0, 5), rng.normal(0, 5), rng.uniform(4, 10)) for _ in range(int(rng.integers(3, 7)))]

        def desenho(m, cx, cy):
            for dx, dy, r in circulos:
                _circulo(m, cx + dx, cy + dy, r)
            m[:] = cv2.GaussianBlur(m, (0, 0), 1.2)

        _desenhar(img, x, y, 30, desenho, cor, rng.uniform(0.6, 0.9))
    elif tipo == "fibra":  # fio ou pelo: longo, fino e curvo
        comp, n = rng.uniform(30, 90), int(rng.integers(3, 6))
        esp = int(rng.integers(1, 3))
        ang, px, py, pontos = rng.uniform(0, 2 * math.pi), 0.0, 0.0, []
        for _ in range(n):
            pontos.append((px, py))
            ang += rng.normal(0, 0.5)
            px += comp / (n - 1) * math.cos(ang)
            py += comp / (n - 1) * math.sin(ang)

        def desenho(m, cx, cy):
            p = np.array([[_p(cx + a), _p(cy + b)] for a, b in pontos], np.int32)
            cv2.polylines(m, [p], False, 255, esp, cv2.LINE_AA, _DESLOC)

        _desenhar(img, x, y, comp + 4, desenho, np.full(3, rng.uniform(30, 60)))
    elif tipo == "fragmento":  # pedaço de inseto: alongado, porém bem mais fino que um ovo
        comp, larg, ang = rng.uniform(22, 40), rng.uniform(2.0, 3.5), rng.uniform(0, 180)
        _desenhar(img, x, y, comp / 2 + 3, lambda m, cx, cy: _elipse(m, cx, cy, comp / 2, larg / 2, ang),
                  np.full(3, rng.uniform(30, 55)))
    elif tipo == "bolota":  # fragmento escuro arredondado, com área parecida com a de um ovo
        r = rng.uniform(4, 6.5)
        _desenhar(img, x, y, r + 2, lambda m, cx, cy: _circulo(m, cx, cy, r), np.full(3, rng.uniform(35, 60)))
    elif tipo == "semente":  # grão vegetal: formato parecido com o de um ovo, mas amarronzado
        comp, larg, ang = rng.uniform(12, 20), rng.uniform(5, 8), rng.uniform(0, 180)
        cor = np.array([60, 88, 112], np.float32) + rng.normal(0, 8, 3)
        _desenhar(img, x, y, comp / 2 + 3, lambda m, cx, cy: _elipse(m, cx, cy, comp / 2, larg / 2, ang), cor,
                  rng.uniform(0.85, 1.0))
    else:  # grão claro de areia: não deve ser segmentado
        r = rng.uniform(1.5, 4)
        _desenhar(img, x, y, r + 2, lambda m, cx, cy: _circulo(m, cx, cy, r), np.full(3, rng.uniform(200, 235)))


def gerar_palheta(semente: int, negativa: bool | None = None, largura: int = LARGURA, altura: int = ALTURA):
    """Gera (imagem BGR uint8, Gabarito). negativa=None sorteia (20% de chance de palheta sem ovos)."""
    rng = np.random.default_rng(semente)
    img = _fundo(rng, altura, largura)
    if negativa is None:
        negativa = bool(rng.random() < 0.2)

    ovos: list[Ovo] = []
    if not negativa:
        faixa = altura * rng.uniform(0.35, 0.65)
        n_isolados = int(rng.integers(3, 110))
        n_grupos = int(rng.integers(0, n_isolados // 8 + 2))
        for g in range(n_grupos):
            for _tentativa in range(50):
                cx, cy = _posicao(rng, altura, largura, faixa)
                if _longe(cx, cy, ovos, 30):
                    break
            ovos.extend(_aglomerado(rng, cx, cy, int(rng.integers(2, 7)), g))
        for _ in range(n_isolados):
            for _tentativa in range(50):
                x, y = _posicao(rng, altura, largura, faixa)
                if _longe(x, y, ovos, 26):
                    break
            ovos.append(_novo_ovo(rng, x, y))

    detritos = [
        Detrito(str(rng.choice(TIPOS_DETRITO, p=_PROB_DETRITO)), float(rng.uniform(5, largura - 5)),
                float(rng.uniform(5, altura - 5)))
        for _ in range(int(rng.integers(6, 26)))
    ]
    for d in detritos:  # sujeira primeiro, ovos por cima
        _desenhar_detrito(img, rng, d.tipo, d.x, d.y)
    for o in ovos:
        _desenhar_ovo(img, rng, o)

    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.0))  # desfoque da óptica
    img += rng.normal(0, rng.uniform(1.5, 3.5), img.shape).astype(np.float32)  # ruído do sensor
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, Gabarito(arquivo="", semente=int(semente), largura=largura, altura=altura, ovos=ovos, detritos=detritos)


def gerar_base(pasta, quantidade: int, semente_inicial: int = 0, a_cada_negativa: int = 5) -> list[Path]:
    """Grava `quantidade` palhetas (PNG + gabarito JSON). Uma em cada `a_cada_negativa` não tem ovos."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for i in range(quantidade):
        semente = semente_inicial + i
        img, gab = gerar_palheta(semente, negativa=(a_cada_negativa > 0 and i % a_cada_negativa == 0))
        nome = f"palheta_{semente:05d}.png"
        gab.arquivo = nome
        salvar_imagem(pasta / nome, img)
        (pasta / nome).with_suffix(".json").write_text(
            json.dumps(gab.para_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        caminhos.append(pasta / nome)
    return caminhos


def ler_gabarito(caminho_imagem) -> Gabarito | None:
    caminho_json = Path(caminho_imagem).with_suffix(".json")
    if not caminho_json.is_file():
        return None
    try:
        return Gabarito.de_dict(json.loads(caminho_json.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError):
        return None
