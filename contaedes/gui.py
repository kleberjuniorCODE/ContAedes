"""Interface gráfica simples (Tkinter) para demonstração do ContAedes."""
from __future__ import annotations

import base64
import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from .classificador import carregar_modelo
from .contagem import contar, contar_arquivo
from .processamento import ErroImagem
from .relatorio import anotar, fmt, salvar_resultado
from .sintetico import gerar_palheta, ler_gabarito

LARGURA_MAX, ALTURA_MAX = 1180, 430


class Aplicativo(tk.Tk):
    def __init__(self, pacote: dict | None = None):
        super().__init__()
        self.title("ContAedes - contagem de ovos de Aedes aegypti (prova de conceito)")
        self.geometry("1240x720")
        self.minsize(900, 600)
        self.pacote = pacote
        self._foto = None
        self._ultimo = None  # (resultado, imagem, total do gabarito)

        barra = ttk.Frame(self, padding=8)
        barra.pack(fill="x")
        ttk.Button(barra, text="Abrir imagem...", command=self.abrir_imagem).pack(side="left", padx=4)
        ttk.Button(barra, text="Processar pasta (lote)...", command=self.processar_pasta).pack(side="left", padx=4)
        ttk.Button(barra, text="Gerar palheta de exemplo", command=self.gerar_exemplo).pack(side="left", padx=4)
        ttk.Button(barra, text="Salvar resultado...", command=self.salvar).pack(side="left", padx=4)

        self.quadro_imagem = ttk.Label(self, anchor="center", relief="groove")
        self.quadro_imagem.pack(fill="both", expand=True, padx=8, pady=4)

        legenda = ttk.Frame(self, padding=(8, 0))
        legenda.pack(fill="x")
        for cor, texto in (("#28be28", "ovo isolado"), ("#ffa000", "aglomerado (nº estimado ao lado)"),
                           ("#dc3232", "detrito descartado")):
            tk.Label(legenda, text="  ", bg=cor).pack(side="left", padx=(8, 4))
            ttk.Label(legenda, text=texto).pack(side="left", padx=(0, 12))

        self.texto = tk.Text(self, height=7, font=("Segoe UI", 11), wrap="word", relief="flat", bg="#f4f4f4")
        self.texto.pack(fill="x", padx=8, pady=6)
        self.status = ttk.Label(self, text="Pronto.", anchor="w", padding=(8, 2))
        self.status.pack(fill="x", side="bottom")
        self._escrever("Abra uma imagem de palheta, processe uma pasta inteira ou gere uma palheta de exemplo.")
        if self.pacote is None:
            self.after(100, self._carregar_modelo)

    # ------------------------------------------------------------------ utilidades
    def _carregar_modelo(self):
        self._status("Carregando o classificador...")
        self.pacote = carregar_modelo(verbose=False)
        self._status("Classificador pronto.")

    def _status(self, texto):
        self.status.configure(text=texto)
        self.update_idletasks()

    def _escrever(self, texto):
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.insert("1.0", texto)
        self.texto.configure(state="disabled")

    def mostrar_imagem(self, img_bgr):
        h, w = img_bgr.shape[:2]
        escala = min(LARGURA_MAX / w, ALTURA_MAX / h, 1.0)
        if escala < 1.0:
            img_bgr = cv2.resize(img_bgr, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", img_bgr)
        self._foto = tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))
        self.quadro_imagem.configure(image=self._foto)

    def exibir_resultado(self, res, img, total_gabarito=None):
        self._ultimo = (res, img, total_gabarito)
        self.mostrar_imagem(anotar(img, res, total_gabarito))
        linhas = [
            f"Arquivo: {res.arquivo or '(palheta gerada)'}",
            f"Ovos estimados: {res.total_ovos}   |   isolados: {res.ovos_isolados}   |   "
            f"aglomerados: {res.aglomerados} (≈ {res.ovos_em_aglomerados} ovos)",
            f"Detritos descartados: {res.detritos}   |   limiar de Otsu usado: {fmt(res.limiar, 0)}   |   "
            f"tempo: {fmt(res.tempo_s, 3)} s",
            "Situação da palheta: " + ("NEGATIVA" if not res.positiva else
                                       "POSITIVA - contagem baixa, confira visualmente antes de registrar"
                                       if res.conferir else "POSITIVA"),
        ]
        if total_gabarito is not None:
            linhas.append(f"Gabarito: {total_gabarito} ovos  ->  erro de {res.total_ovos - total_gabarito:+d}")
        self._escrever("\n".join(linhas))

    # ------------------------------------------------------------------ ações
    def processar_arquivo(self, caminho):
        try:
            res, img = contar_arquivo(caminho, self.pacote)
        except ErroImagem as erro:
            messagebox.showerror("ContAedes", str(erro))
            return None
        gab = ler_gabarito(caminho)
        self.exibir_resultado(res, img, gab.total_ovos if gab else None)
        self._status(f"Processado: {caminho}")
        return res

    def abrir_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Escolha a imagem da palheta",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos os arquivos", "*.*")])
        if caminho:
            self.processar_arquivo(caminho)

    def gerar_exemplo(self, semente=None):
        semente = random.randint(10_000, 99_999) if semente is None else semente
        img, gab = gerar_palheta(semente)
        res = contar(img, self.pacote, arquivo=f"exemplo_{semente}.png")
        self.exibir_resultado(res, img, gab.total_ovos)
        self._status(f"Palheta sintética gerada (semente {semente}).")
        return res, gab

    def processar_pasta(self, pasta=None):
        from .cli import processar_lote
        pasta = pasta or filedialog.askdirectory(title="Escolha a pasta com as imagens das palhetas")
        if not pasta:
            return None
        self._status("Processando lote...")
        linhas, ind, csv_lote = processar_lote(pasta, self.pacote, mostrar=lambda _t: None)
        erros = [l for l in linhas if l.get("situacao") == "erro"]
        ok = [l for l in linhas if l.get("situacao") != "erro"]
        texto = [f"Lote: {pasta}",
                 f"Palhetas examinadas: {ind.examinadas}   |   positivas: {ind.positivas}   |   ovos: {ind.total_ovos}",
                 f"IPO = {fmt(ind.ipo, 1)}%   |   IDO = {fmt(ind.ido, 1)} ovos por palheta positiva",
                 "Por palheta: " + "; ".join(f"{l['arquivo']}: {l['ovos']}"
                                             + (" (conferir)" if "conferir" in l.get("situacao", "") else "")
                                             for l in ok[:12])
                 + (" ..." if len(ok) > 12 else "")]
        if erros:
            texto.append(f"{len(erros)} arquivo(s) ignorado(s) por erro de leitura.")
        texto.append(f"Planilha e imagens anotadas em: {Path(csv_lote).parent}")
        self._escrever("\n".join(texto))
        if ok:
            ultimo = Path(pasta) / ok[-1]["arquivo"]
            self.processar_arquivo(ultimo)
            self._escrever("\n".join(texto))
        self._status("Lote concluído.")
        return ind

    def salvar(self):
        if not self._ultimo:
            messagebox.showinfo("ContAedes", "Processe uma imagem antes de salvar.")
            return
        pasta = filedialog.askdirectory(title="Onde salvar a imagem anotada e a planilha?")
        if pasta:
            res, img, total = self._ultimo
            arquivos = salvar_resultado(res, img, pasta, total)
            self._status(f"Salvo: {arquivos['imagem'].name} e {arquivos['csv'].name}")


def executar() -> int:
    Aplicativo().mainloop()
    return 0
