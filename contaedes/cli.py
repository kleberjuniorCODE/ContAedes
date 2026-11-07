"""Interface de linha de comando: python -m contaedes <comando>."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .processamento import EXTENSOES_ACEITAS, ErroImagem
from .relatorio import fmt


def _ovos(n: int) -> str:
    return f"{n} ovo" if n == 1 else f"{n} ovos"


def _listar_imagens(pasta: Path) -> list[Path]:
    return sorted(p for p in pasta.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTENSOES_ACEITAS and not p.stem.endswith("_anotada"))


def processar_lote(pasta, pacote, pasta_saida=None, mostrar=print):
    """Conta todas as imagens da pasta. Arquivo com problema é registrado e o lote continua."""
    from .contagem import calcular_indicadores, contar_arquivo
    from .relatorio import salvar_lote_csv, salvar_resultado
    from .sintetico import ler_gabarito

    pasta = Path(pasta)
    saida = Path(pasta_saida) if pasta_saida else pasta / "resultados_contaedes"
    linhas, totais = [], []
    for caminho in _listar_imagens(pasta):
        try:
            res, img = contar_arquivo(caminho, pacote)
        except ErroImagem as erro:
            linhas.append({"arquivo": caminho.name, "situacao": "erro", "observacao": str(erro)})
            mostrar(f"  [aviso] {erro}")
            continue
        gab = ler_gabarito(caminho)
        salvar_resultado(res, img, saida, gab.total_ovos if gab else None)
        totais.append(res.total_ovos)
        linhas.append({"arquivo": caminho.name, "ovos": res.total_ovos, "isolados": res.ovos_isolados,
                       "aglomerados": res.aglomerados, "ovos_em_aglomerados": res.ovos_em_aglomerados,
                       "detritos": res.detritos, "situacao": res.situacao,
                       "gabarito": gab.total_ovos if gab else "", "tempo_s": fmt(res.tempo_s, 3),
                       "observacao": "contagem baixa: conferir visualmente" if res.conferir else ""})
        extra = f" (gabarito: {gab.total_ovos})" if gab else ""
        aviso = "  <- conferir visualmente" if res.conferir else ""
        mostrar(f"  {caminho.name}: {_ovos(res.total_ovos)}{extra}{aviso}")
    indicadores = calcular_indicadores(totais)
    csv_lote = salvar_lote_csv(linhas, indicadores, saida / "resumo_lote.csv")
    return linhas, indicadores, csv_lote


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="contaedes",
        description="ContAedes - contagem automática de ovos de Aedes aegypti em palhetas de ovitrampas "
                    "(prova de conceito acadêmica).")
    sub = parser.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("contar", help="conta os ovos de uma imagem")
    p.add_argument("imagem")
    p.add_argument("--saida", help="pasta dos resultados (padrão: resultados_contaedes ao lado da imagem)")
    p = sub.add_parser("lote", help="processa todas as imagens de uma pasta e calcula IPO e IDO")
    p.add_argument("pasta")
    p.add_argument("--saida")
    p = sub.add_parser("gerar-base", help="gera palhetas sintéticas com gabarito")
    p.add_argument("--saida", default="dados/sinteticas")
    p.add_argument("--quantidade", type=int, default=20)
    p.add_argument("--semente", type=int, default=0)
    p = sub.add_parser("treinar", help="treina o classificador Random Forest")
    p.add_argument("--quantidade", type=int, default=60)
    p.add_argument("--semente", type=int, default=1000)
    p = sub.add_parser("avaliar", help="mede o desempenho numa base de validação independente")
    p.add_argument("--quantidade", type=int, default=40)
    p.add_argument("--semente", type=int, default=5000)
    p.add_argument("--saida", default="resultados")
    sub.add_parser("interface", help="abre a interface gráfica")
    args = parser.parse_args(argv)

    if args.comando == "interface":
        from .gui import executar
        return executar()

    if args.comando == "gerar-base":
        from .sintetico import gerar_base
        caminhos = gerar_base(args.saida, args.quantidade, args.semente)
        print(f"{len(caminhos)} palhetas sintéticas (PNG + gabarito JSON) gravadas em {Path(args.saida).resolve()}")
        return 0

    from .classificador import CAMINHO_MODELO, carregar_modelo, treinar

    if args.comando == "treinar":
        treinar(quantidade=args.quantidade, semente_inicial=args.semente, caminho=CAMINHO_MODELO)
        print(f"Modelo salvo em {CAMINHO_MODELO}")
        return 0

    pacote = carregar_modelo()

    if args.comando == "contar":
        from .contagem import contar_arquivo
        from .relatorio import salvar_resultado
        from .sintetico import ler_gabarito
        try:
            res, img = contar_arquivo(args.imagem, pacote)
        except ErroImagem as erro:
            print(f"Erro: {erro}", file=sys.stderr)
            return 2
        gab = ler_gabarito(args.imagem)
        saida = Path(args.saida) if args.saida else Path(args.imagem).resolve().parent / "resultados_contaedes"
        arquivos = salvar_resultado(res, img, saida, gab.total_ovos if gab else None)
        print(f"Ovos estimados: {res.total_ovos} (isolados: {res.ovos_isolados}; "
              f"{res.aglomerados} aglomerados com {res.ovos_em_aglomerados} ovos)")
        print(f"Detritos descartados: {res.detritos} | tempo: {fmt(res.tempo_s, 3)} s | situação: {res.situacao}")
        if res.conferir:
            print("Atenção: contagem baixa. Confira a palheta visualmente antes de registrá-la como positiva.")
        if gab:
            print(f"Gabarito: {gab.total_ovos} ovos (erro: {res.total_ovos - gab.total_ovos:+d})")
        print(f"Imagem anotada: {arquivos['imagem']}\nPlanilha: {arquivos['csv']}")
        return 0

    if args.comando == "lote":
        if not Path(args.pasta).is_dir():
            print(f"Erro: pasta não encontrada: {args.pasta}", file=sys.stderr)
            return 2
        _, ind, csv_lote = processar_lote(args.pasta, pacote, args.saida)
        print(f"\nPalhetas examinadas: {ind.examinadas} | positivas: {ind.positivas} | ovos: {ind.total_ovos}")
        print(f"IPO = {fmt(ind.ipo, 1)}% | IDO = {fmt(ind.ido, 1)} ovos por palheta positiva")
        print(f"Planilha do lote: {csv_lote}")
        return 0

    if args.comando == "avaliar":
        from .avaliacao import avaliar
        r = avaliar(pacote, args.quantidade, args.semente, args.saida)
        print(f"Base de validação: {r['palhetas']} palhetas ({r['palhetas_negativas']} sem ovos), "
              f"{r['ovos_reais_total']} ovos no gabarito")
        print(f"Erro absoluto médio: {fmt(r['eam_ovos'])} ovo(s) por palheta | "
              f"erro percentual médio: {fmt(r['epam_percentual'])}% | erro máximo: {r['erro_maximo_ovos']} ovos")
        print(f"Palhetas positivas com erro <= 10%: {fmt(r['palhetas_com_erro_ate_10pct'], 1)}% | "
              f"negativas reconhecidas: {fmt(r['negativas_corretas_pct'], 1)}%")
        print(f"IPO real x estimado: {fmt(r['ipo_real'], 1)}% x {fmt(r['ipo_estimado'], 1)}% | "
              f"IDO: {fmt(r['ido_real'], 1)} x {fmt(r['ido_estimado'], 1)}")
        neg = r["teste_negativas"]
        print(f"Teste extra com {neg['negativas_testadas']} palhetas sem ovos: {neg['falsos_positivos']} falso(s) "
              f"positivo(s) ({fmt(neg['taxa_falso_positivo_pct'], 1)}%), "
              f"{neg['falsos_positivos_sinalizados_para_conferencia']} deles sinalizado(s) para conferência")
        print(f"F1 macro - Random Forest: {fmt(r['f1_macro_rf'], 3)} | regra fixa: {fmt(r['f1_macro_regras'], 3)}")
        print(f"Tempo médio por imagem: {fmt(r['tempo_medio_s'], 3)} s (máximo {fmt(r['tempo_maximo_s'], 3)} s)")
        print(f"Detalhes em {Path(args.saida).resolve()}")
        return 0
    return 1
