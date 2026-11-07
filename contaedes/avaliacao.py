"""Avaliação do protótipo numa base de validação independente (sementes diferentes das do treino).

Mede o que o plano da qualidade definiu como critério de aceitação:
erro de contagem por palheta, F1 do classificador, IPO/IDO e tempo de
processamento. Também separa o erro por causa, para o gráfico de Pareto.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .classificador import CLASSES, classe_de, classificar_por_regras, ovos_por_componente
from .contagem import calcular_indicadores, contar
from .sintetico import gerar_palheta


def avaliar_negativas(pacote: dict, quantidade: int = 100, semente_inicial: int = 20000) -> dict:
    """Taxa de falso positivo: palhetas SEM ovos que o sistema contaria como positivas."""
    falsos = [contar(gerar_palheta(semente_inicial + i, negativa=True)[0], pacote).total_ovos for i in range(quantidade)]
    falsos = [f for f in falsos if f > 0]
    return {
        "negativas_testadas": quantidade,
        "falsos_positivos": len(falsos),
        "taxa_falso_positivo_pct": 100.0 * len(falsos) / quantidade,
        "falsos_positivos_sinalizados_para_conferencia": sum(1 for f in falsos if f <= 2),
        "maior_contagem_falsa": max(falsos, default=0),
    }


def avaliar(pacote: dict, quantidade: int = 40, semente_inicial: int = 5000, pasta_saida=None) -> dict:
    linhas, verdade, previsto, regras = [], [], [], []
    causas = {"ovo isolado descartado como detrito": 0, "detrito contado como ovo": 0,
              "erro no nº estimado de um aglomerado": 0, "aglomerado tratado como ovo isolado": 0,
              "ovo isolado tratado como aglomerado": 0, "aglomerado descartado como detrito": 0,
              "ovo não segmentado": 0, "detrito tratado como aglomerado": 0}
    for i in range(quantidade):
        semente = semente_inicial + i
        img, gab = gerar_palheta(semente, negativa=(i % 5 == 0))
        res = contar(img, pacote, arquivo=f"palheta_{semente:05d}")
        n_real = ovos_por_componente(res.componentes, res.rotulos, gab)
        causas["ovo não segmentado"] += gab.total_ovos - int(n_real.sum())
        X = np.vstack([c.caracteristicas for c in res.componentes]) if res.componentes else np.empty((0, 13))
        for c, k, r in zip(res.componentes, n_real, classificar_por_regras(X) if len(X) else []):
            classe_real = classe_de(int(k))
            verdade.append(classe_real)
            previsto.append(c.classe)
            regras.append(r)
            if classe_real == "ovo" and c.classe == "detrito":
                causas["ovo isolado descartado como detrito"] += 1
            elif classe_real == "detrito" and c.classe == "ovo":
                causas["detrito contado como ovo"] += 1
            elif classe_real == "detrito" and c.classe == "aglomerado":
                causas["detrito tratado como aglomerado"] += c.ovos_estimados
            elif classe_real == "aglomerado" and c.classe == "aglomerado":
                causas["erro no nº estimado de um aglomerado"] += abs(c.ovos_estimados - int(k))
            elif classe_real == "aglomerado" and c.classe == "ovo":
                causas["aglomerado tratado como ovo isolado"] += int(k) - 1
            elif classe_real == "ovo" and c.classe == "aglomerado":
                causas["ovo isolado tratado como aglomerado"] += c.ovos_estimados - 1
            elif classe_real == "aglomerado" and c.classe == "detrito":
                causas["aglomerado descartado como detrito"] += int(k)
        erro = res.total_ovos - gab.total_ovos
        linhas.append({
            "palheta": res.arquivo,
            "ovos_reais": gab.total_ovos,
            "ovos_estimados": res.total_ovos,
            "erro": erro,
            "erro_percentual": round(100 * abs(erro) / gab.total_ovos, 2) if gab.total_ovos else None,
            "aglomerados_reais": len({o.aglomerado for o in gab.ovos if o.aglomerado >= 0}),
            "detritos_gabarito": len(gab.detritos),
            "tempo_s": round(res.tempo_s, 4),
        })

    reais = np.array([l["ovos_reais"] for l in linhas])
    estim = np.array([l["ovos_estimados"] for l in linhas])
    erros = np.abs(estim - reais)
    pos = reais > 0
    ind_real, ind_est = calcular_indicadores(reais), calcular_indicadores(estim)
    tempos = np.array([l["tempo_s"] for l in linhas])
    rel = classification_report(verdade, previsto, labels=list(CLASSES), output_dict=True, zero_division=0)
    resumo = {
        "palhetas": int(quantidade),
        "palhetas_positivas": int(pos.sum()),
        "palhetas_negativas": int((~pos).sum()),
        "ovos_reais_total": int(reais.sum()),
        "ovos_estimados_total": int(estim.sum()),
        "eam_ovos": float(erros.mean()),
        "epam_percentual": float(100 * np.mean(erros[pos] / reais[pos])) if pos.any() else 0.0,
        "erro_maximo_ovos": int(erros.max()),
        "erro_percentual_maximo": float(100 * np.max(erros[pos] / reais[pos])) if pos.any() else 0.0,
        "palhetas_com_erro_ate_10pct": float(100 * np.mean(erros[pos] / reais[pos] <= 0.10)) if pos.any() else 0.0,
        "negativas_corretas_pct": float(100 * np.mean(estim[~pos] == 0)) if (~pos).any() else 100.0,
        "ipo_real": ind_real.ipo, "ipo_estimado": ind_est.ipo,
        "ido_real": ind_real.ido, "ido_estimado": ind_est.ido,
        "erro_relativo_ido_pct": float(100 * abs(ind_est.ido - ind_real.ido) / ind_real.ido) if ind_real.ido else 0.0,
        "componentes_avaliados": len(verdade),
        "f1_macro_rf": float(rel["macro avg"]["f1-score"]),
        "f1_macro_regras": float(f1_score(verdade, regras, labels=list(CLASSES), average="macro", zero_division=0)),
        "relatorio_rf": rel,
        "matriz_confusao": confusion_matrix(verdade, previsto, labels=list(CLASSES)).tolist(),
        "matriz_confusao_regras": confusion_matrix(verdade, regras, labels=list(CLASSES)).tolist(),
        "rotulos_matriz": list(CLASSES),
        "tempo_medio_s": float(tempos.mean()),
        "tempo_maximo_s": float(tempos.max()),
        "causas_de_erro_ovos": causas,
        "semente_inicial": semente_inicial,
        "teste_negativas": avaliar_negativas(pacote),
    }
    if pasta_saida:
        pasta = Path(pasta_saida)
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / "avaliacao_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(pasta / "avaliacao_palhetas.csv", "w", newline="", encoding="utf-8-sig") as arq:
            w = csv.DictWriter(arq, fieldnames=list(linhas[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(linhas)
    resumo["linhas"] = linhas
    return resumo
