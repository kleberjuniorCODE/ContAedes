"""Componente de aprendizado de máquina do ContAedes.

Cada componente conexo encontrado na segmentação é descrito por 13
características (área, alongamento, espessura, intensidade...) e classificado
por uma Random Forest em 'ovo', 'aglomerado' ou 'detrito'. O treino é
supervisionado: o rótulo de cada componente vem do gabarito das palhetas
sintéticas (quantos centros de ovos caem dentro dele).
"""
from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .processamento import NOMES_CARACTERISTICAS, Parametros, segmentar
from .sintetico import gerar_palheta

CLASSES = ("ovo", "aglomerado", "detrito")
PASTA_PROJETO = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = PASTA_PROJETO / "modelos" / "classificador_rf.joblib"
MIN_ISOLADOS_REFERENCIA = 5
_I = {nome: i for i, nome in enumerate(NOMES_CARACTERISTICAS)}


def classe_de(n_ovos: int) -> str:
    return "detrito" if n_ovos == 0 else ("ovo" if n_ovos == 1 else "aglomerado")


def ovos_por_componente(componentes, rotulos, gabarito) -> np.ndarray:
    """Quantos centros de ovos do gabarito caem dentro de cada componente."""
    indice = {c.rotulo: i for i, c in enumerate(componentes)}
    contagem = np.zeros(len(componentes), dtype=int)
    altura, largura = rotulos.shape
    for ovo in gabarito.ovos:
        x, y = int(round(ovo.x)), int(round(ovo.y))
        for dy, dx in ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            r = int(rotulos[min(max(y + dy, 0), altura - 1), min(max(x + dx, 0), largura - 1)])
            if r in indice:
                contagem[indice[r]] += 1
                break
    return contagem


def area_referencia(areas_isolados, padrao: float) -> float:
    """Mediana da área dos ovos isolados da própria imagem (adapta-se à resolução)."""
    if len(areas_isolados) >= MIN_ISOLADOS_REFERENCIA:
        return float(np.median(areas_isolados))
    return float(padrao)


def amostras_sinteticas(quantidade: int, semente_inicial: int):
    for i in range(quantidade):
        yield gerar_palheta(semente_inicial + i, negativa=(i % 5 == 0))


def montar_dados(amostras, params: Parametros | None = None) -> dict:
    """Segmenta cada imagem e rotula os componentes pelo gabarito."""
    params = params or Parametros()
    X, n, ref = [], [], []
    for img, gab in amostras:
        seg = segmentar(img, params)
        if not seg.componentes:
            continue
        cont = ovos_por_componente(seg.componentes, seg.rotulos, gab)
        areas = [c.area for c, k in zip(seg.componentes, cont) if k == 1]
        ref_img = area_referencia(areas, np.median(areas) if areas else np.nan)
        for c, k in zip(seg.componentes, cont):
            X.append(c.caracteristicas)
            n.append(int(k))
            ref.append(ref_img)
    n = np.array(n)
    return {"X": np.array(X), "y": np.array([classe_de(k) for k in n]), "n": n, "area_ref": np.array(ref)}


def novo_modelo() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1)


def treinar(quantidade: int = 60, semente_inicial: int = 1000, params: Parametros | None = None,
            caminho=CAMINHO_MODELO, amostras=None, verbose: bool = True) -> dict:
    params = params or Parametros()
    dados = montar_dados(amostras if amostras is not None else amostras_sinteticas(quantidade, semente_inicial), params)
    X, y, n = dados["X"], dados["y"], dados["n"]

    # 1) validação: separa 25% dos componentes (estratificado) e faz validação cruzada em 5 partes
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    previsto = novo_modelo().fit(X_tr, y_tr).predict(X_te)
    relatorio = classification_report(y_te, previsto, labels=list(CLASSES), output_dict=True, zero_division=0)
    cv = cross_val_score(novo_modelo(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1_macro")

    # 2) modelo final treinado com todos os componentes
    modelo = novo_modelo().fit(X, y)

    # 3) calibração do número de ovos por aglomerado: n ~ fator * área / área de referência
    isolados = X[y == "ovo", _I["area"]]
    area_padrao = float(np.median(isolados)) if isolados.size else 100.0
    agl = (y == "aglomerado") & np.isfinite(dados["area_ref"])
    fator = float(np.median(n[agl] / (X[agl, _I["area"]] / dados["area_ref"][agl]))) if agl.any() else 1.0

    pacote = {
        "modelo": modelo,
        "caracteristicas": list(NOMES_CARACTERISTICAS),
        "classes": list(CLASSES),
        "area_referencia": area_padrao,
        "fator_aglomerado": fator,
        "parametros": asdict(params),
        "versao_sklearn": sklearn.__version__,
        "metricas_treino": {
            "imagens": quantidade if amostras is None else None,
            "componentes": int(len(y)),
            "por_classe": {c: int((y == c).sum()) for c in CLASSES},
            "f1_macro_holdout": float(relatorio["macro avg"]["f1-score"]),
            "f1_macro_cv_media": float(cv.mean()),
            "f1_macro_cv_desvio": float(cv.std()),
            "relatorio_holdout": relatorio,
            "importancias": dict(zip(NOMES_CARACTERISTICAS, np.round(modelo.feature_importances_, 4).tolist())),
        },
    }
    if caminho:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pacote, caminho)
    if verbose:
        m = pacote["metricas_treino"]
        print(f"[ContAedes] classificador treinado com {m['componentes']} componentes {m['por_classe']}")
        print(f"[ContAedes] F1 macro (teste 25%): {m['f1_macro_holdout']:.3f} | "
              f"validação cruzada 5 partes: {m['f1_macro_cv_media']:.3f} ± {m['f1_macro_cv_desvio']:.3f}")
    return pacote


def carregar_modelo(caminho=CAMINHO_MODELO, treinar_se_preciso: bool = True, verbose: bool = True) -> dict:
    """Carrega o modelo salvo. Se faltar ou for de outra versão do scikit-learn, treina de novo."""
    caminho = Path(caminho)
    motivo = "modelo ainda não treinado"
    if caminho.is_file():
        try:
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always")
                pacote = joblib.load(caminho)
            if any("InconsistentVersion" in a.category.__name__ for a in avisos):
                motivo = "modelo salvo com outra versão do scikit-learn"
            elif not isinstance(pacote, dict) or "modelo" not in pacote:
                motivo = "arquivo de modelo inválido"
            else:
                return pacote
        except Exception as erro:  # arquivo corrompido, versão incompatível etc.
            motivo = f"falha ao carregar o modelo ({erro.__class__.__name__})"
    if not treinar_se_preciso:
        raise FileNotFoundError(motivo)
    if verbose:
        print(f"[ContAedes] {motivo}: treinando um novo classificador (leva alguns segundos)...")
    return treinar(caminho=caminho, verbose=verbose)


def classificar_por_regras(X: np.ndarray) -> np.ndarray:
    """Referência SEM aprendizado: limiares fixos escolhidos à mão, usada só para comparação."""
    saida = []
    for v in np.atleast_2d(X):
        area, along, esp, tom = v[_I["area"]], v[_I["alongamento"]], v[_I["espessura"]], v[_I["intensidade_media"]]
        if tom > 85 or esp < 4.0 or area < 50:
            saida.append("detrito")
        elif area <= 190 and 1.7 <= along <= 5.0:
            saida.append("ovo")
        elif area > 190 and esp >= 6.0:
            saida.append("aglomerado")
        else:
            saida.append("detrito")
    return np.array(saida)
