"""Testes automatizados do ContAedes (python -m unittest discover -s testes -v).

Cada teste tem o código do caso de teste (CT) usado no plano da qualidade do documento.
"""
import io
import json
import math
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from contaedes.classificador import carregar_modelo, classe_de, ovos_por_componente
from contaedes.cli import main, processar_lote
from contaedes.contagem import calcular_indicadores, contar, contar_arquivo
from contaedes.processamento import ErroImagem, ler_imagem, salvar_imagem, segmentar
from contaedes.relatorio import salvar_resultado
from contaedes.sintetico import ALTURA, LARGURA, _aglomerado, _desenhar_ovo, _fundo, gerar_base, gerar_palheta


def _fundo_limpo(semente=0, altura=200, largura=400):
    rng = np.random.default_rng(semente)
    return rng, _fundo(rng, altura, largura)


class TestContAedes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pacote = carregar_modelo(verbose=False)

    def test_ct01_gerador_respeita_gabarito(self):
        img, gab = gerar_palheta(1, negativa=False)
        self.assertEqual(img.shape, (ALTURA, LARGURA, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertGreater(gab.total_ovos, 0)
        _, gab_neg = gerar_palheta(2, negativa=True)
        self.assertEqual(gab_neg.total_ovos, 0)

    def test_ct02_segmentacao_encontra_todos_os_ovos(self):
        for semente in (11, 12, 13):
            img, gab = gerar_palheta(semente, negativa=False)
            seg = segmentar(img)
            encontrados = int(ovos_por_componente(seg.componentes, seg.rotulos, gab).sum())
            self.assertGreaterEqual(encontrados, 0.98 * gab.total_ovos, f"semente {semente}")

    def test_ct03_palheta_limpa_nao_tem_ovos(self):
        _, fundo = _fundo_limpo(3)
        img = np.clip(fundo, 0, 255).astype(np.uint8)
        self.assertEqual(contar(img, self.pacote).total_ovos, 0)

    def test_ct04_erro_de_contagem_ate_10_por_cento(self):
        for semente in range(7000, 7006):
            img, gab = gerar_palheta(semente, negativa=False)
            res = contar(img, self.pacote)
            erro = abs(res.total_ovos - gab.total_ovos) / gab.total_ovos
            self.assertLessEqual(erro, 0.10, f"semente {semente}: {res.total_ovos} x {gab.total_ovos}")

    def test_ct05_aglomerado_de_quatro_ovos(self):
        rng, img = _fundo_limpo(5)
        for ovo in _aglomerado(rng, 200, 100, 4, 0):
            _desenhar_ovo(img, rng, ovo)
        for k, (x, y) in enumerate([(60, 60), (330, 150), (90, 160), (320, 50), (200, 30), (40, 110)]):
            from contaedes.sintetico import _novo_ovo
            _desenhar_ovo(img, rng, _novo_ovo(rng, x, y, angulo=30 * k))
        img = np.clip(img, 0, 255).astype(np.uint8)
        res = contar(img, self.pacote)
        self.assertEqual(res.aglomerados, 1)
        self.assertIn(res.ovos_em_aglomerados, (3, 4, 5))
        self.assertEqual(res.ovos_isolados, 6)

    def test_ct06_indicadores_ipo_ido(self):
        ind = calcular_indicadores([0, 10, 20, 0])
        self.assertTrue(math.isclose(ind.ipo, 50.0))
        self.assertTrue(math.isclose(ind.ido, 15.0))
        vazio = calcular_indicadores([])
        self.assertEqual((vazio.ipo, vazio.ido), (0.0, 0.0))

    def test_ct07_gera_imagem_anotada_e_planilha(self):
        with tempfile.TemporaryDirectory() as tmp:
            img, _ = gerar_palheta(21, negativa=False)
            caminho = Path(tmp) / "palheta teste ç.png"  # nome com espaço e acento
            salvar_imagem(caminho, img)
            res, img_lida = contar_arquivo(caminho, self.pacote)
            arquivos = salvar_resultado(res, img_lida, Path(tmp) / "saida")
            self.assertTrue(arquivos["imagem"].is_file())
            linhas = arquivos["csv"].read_text(encoding="utf-8-sig").strip().splitlines()
            self.assertEqual(len(linhas), len(res.componentes) + 1)
            self.assertEqual(ler_imagem(arquivos["imagem"]).shape[1], LARGURA)

    def test_ct08_arquivo_invalido_nao_interrompe_o_lote(self):
        with tempfile.TemporaryDirectory() as tmp:
            pasta = Path(tmp)
            gerar_base(pasta, 3, semente_inicial=300)
            (pasta / "corrompida.png").write_bytes(b"isto nao e uma imagem")
            (pasta / "anotacoes.txt").write_text("arquivo que deve ser ignorado")
            with self.assertRaises(ErroImagem):
                ler_imagem(pasta / "corrompida.png")
            with self.assertRaises(ErroImagem):
                ler_imagem(pasta / "anotacoes.txt")
            linhas, ind, csv_lote = processar_lote(pasta, self.pacote, mostrar=lambda _t: None)
            self.assertEqual(sum(1 for l in linhas if l["situacao"] == "erro"), 1)
            self.assertEqual(ind.examinadas, 3)
            self.assertTrue(csv_lote.is_file())

    def test_ct09_tempo_medio_abaixo_de_3_segundos(self):
        tempos = []
        for semente in range(400, 405):
            img, _ = gerar_palheta(semente)
            inicio = time.perf_counter()
            contar(img, self.pacote)
            tempos.append(time.perf_counter() - inicio)
        self.assertLess(sum(tempos) / len(tempos), 3.0)

    def test_ct10_classificador_f1_minimo(self):
        from sklearn.metrics import f1_score
        verdade, previsto = [], []
        for semente in range(800, 808):
            img, gab = gerar_palheta(semente, negativa=(semente == 800))
            res = contar(img, self.pacote)
            n_real = ovos_por_componente(res.componentes, res.rotulos, gab)
            verdade += [classe_de(int(k)) for k in n_real]
            previsto += [c.classe for c in res.componentes]
        self.assertGreaterEqual(f1_score(verdade, previsto, average="macro"), 0.90)

    def test_ct11_linha_de_comando(self):
        with tempfile.TemporaryDirectory() as tmp:
            gerar_base(tmp, 2, semente_inicial=600)
            saida = io.StringIO()
            with redirect_stdout(saida):
                codigo = main(["contar", str(Path(tmp) / "palheta_00601.png"), "--saida", str(Path(tmp) / "r")])
                codigo_lote = main(["lote", tmp])
            self.assertEqual((codigo, codigo_lote), (0, 0))
            self.assertIn("Ovos estimados", saida.getvalue())
            self.assertIn("IPO", saida.getvalue())
            self.assertEqual(main(["contar", str(Path(tmp) / "nao_existe.png")]), 2)

    def test_ct13_contagem_baixa_sinalizada_para_conferencia(self):
        from contaedes.sintetico import _novo_ovo
        rng, img = _fundo_limpo(13)
        _desenhar_ovo(img, rng, _novo_ovo(rng, 200, 100, angulo=20))
        um_ovo = contar(np.clip(img, 0, 255).astype(np.uint8), self.pacote)
        self.assertEqual((um_ovo.total_ovos, um_ovo.situacao), (1, "positiva (conferir)"))
        for k, x in enumerate((60, 120, 280, 340)):
            _desenhar_ovo(img, rng, _novo_ovo(rng, x, 60 + 25 * k, angulo=40 * k))
        cinco = contar(np.clip(img, 0, 255).astype(np.uint8), self.pacote)
        self.assertEqual((cinco.total_ovos, cinco.situacao), (5, "positiva"))

    def test_ct12_interface_grafica(self):
        import tkinter as tk
        try:
            from contaedes.gui import Aplicativo
            app = Aplicativo(self.pacote)
        except tk.TclError as erro:  # máquina sem ambiente gráfico
            self.skipTest(f"sem ambiente gráfico: {erro}")
        try:
            app.withdraw()
            res, gab = app.gerar_exemplo(semente=123)
            texto = app.texto.get("1.0", "end")
            self.assertIn("Ovos estimados", texto)
            self.assertIn(f"Gabarito: {gab.total_ovos}", texto)
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
