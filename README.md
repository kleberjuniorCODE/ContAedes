# ContAedes — protótipo (prova de conceito)

Contagem automática de ovos de *Aedes aegypti* em imagens de palhetas de ovitrampas,
com cálculo do IPO (Índice de Positividade de Ovitrampas) e do IDO (Índice de Densidade de Ovos).

Trabalho da disciplina de Gestão de Projetos — Ciência da Computação, UNIP São José do Rio Preto.
Autores: Kleber de Almeida Araujo e Gilson Aparecido Sousa da Silva.

## O que o protótipo faz

1. Lê a foto ou o escaneamento de uma palheta (PNG, JPG, BMP ou TIFF).
2. **Processamento de imagens:** tons de cinza → realce *black-hat* → filtro gaussiano →
   limiar de Otsu (com piso mínimo) → abertura morfológica → componentes conexos.
3. **Aprendizado de máquina:** cada objeto encontrado é descrito por 13 características
   (área, alongamento, espessura, tom médio...) e classificado por uma *Random Forest* em
   **ovo**, **aglomerado** ou **detrito**.
4. Conta os ovos (aglomerados são estimados pela área), gera a imagem anotada e uma planilha CSV.
5. Em lote (uma pasta = uma semana de coleta), calcula IPO e IDO.
6. Palhetas com 1 ou 2 ovos são marcadas para **conferência visual**, porque uma única
   detecção falsa transformaria uma palheta negativa em positiva.

## Instalação (Windows 10/11)

Requisito: Python 3.10 ou mais novo (https://www.python.org/downloads/ — marque *Add python.exe to PATH*).

Opção rápida: dê dois cliques em **`instalar.bat`**. Ele cria um ambiente virtual (`.venv`) e instala
as bibliotecas do `requirements.txt` (NumPy, OpenCV e scikit-learn).

Opção manual, no terminal, dentro desta pasta:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Como usar

**Interface gráfica:** dois cliques em **`executar_interface.bat`**.
- *Abrir imagem...* — conta uma palheta.
- *Processar pasta (lote)...* — conta todas as imagens de uma pasta e mostra IPO e IDO.
- *Gerar palheta de exemplo* — cria uma palheta sintética nova e mostra a contagem ao lado do gabarito.
- *Salvar resultado...* — grava a imagem anotada e a planilha.

**Linha de comando** (com o ambiente ativado: `.venv\Scripts\activate`):

| Comando | O que faz |
|---|---|
| `python -m contaedes contar dados\exemplos\palheta_09002.png` | conta uma imagem |
| `python -m contaedes lote dados\exemplos` | processa a pasta e calcula IPO/IDO |
| `python -m contaedes gerar-base --saida dados\sinteticas --quantidade 20` | gera palhetas sintéticas com gabarito |
| `python -m contaedes treinar` | treina de novo o classificador |
| `python -m contaedes avaliar` | mede o desempenho numa base de validação independente |
| `python -m unittest discover -s testes -v` | roda os 13 testes automatizados |

Os resultados vão para a pasta `resultados_contaedes`, ao lado das imagens.

## Estrutura

```
contaedes/        código-fonte
  processamento.py   PDI: realce, limiarização, morfologia, características
  classificador.py   Random Forest: treino, validação cruzada, carga do modelo
  contagem.py        contagem por palheta, IPO e IDO
  sintetico.py       gerador de palhetas sintéticas com gabarito
  avaliacao.py       métricas dos critérios de aceitação
  relatorio.py       imagem anotada e planilhas CSV
  cli.py / gui.py    linha de comando e interface gráfica
modelos/          classificador treinado
dados/exemplos/   6 palhetas sintéticas de demonstração (com gabarito .json)
resultados/       saída da última avaliação (avaliacao_resumo.json e .csv)
testes/           testes automatizados (unittest)
```

## Limitações conhecidas

- O classificador foi treinado e validado com **imagens sintéticas**, porque não houve acesso a
  palhetas reais durante o projeto (solicitação de mudança SM-01). O desempenho em fotos reais
  ainda precisa ser medido, com palhetas contadas por técnicos como referência.
- O número de ovos de um aglomerado é estimado pela área; é a maior fonte de erro medida.
- Se o modelo salvo for incompatível com a versão do scikit-learn instalada, o programa treina
  um novo automaticamente (cerca de 15 segundos).
