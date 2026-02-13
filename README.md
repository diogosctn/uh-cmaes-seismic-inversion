# UH-CMA-ES para Inversão Sísmica

Este projeto implementa o algoritmo **UH-CMA-ES** (*Uncertainty Handling - Covariance Matrix Adaptation Evolution Strategy*) adaptado para resolver problemas inversos de sísmica 1D.

## ⚠️ Requisito Importante: SeReM

Este código **não funciona isoladamente**. Ele é uma extensão do repositório de física de rochas e modelagem de ondas do pacote **SeReM** (*Seismic Reservoir Modeling*).

Para que o algoritmo funcione corretamente, ele precisa ter acesso às funções `RickerWavelet`, `WaveletMatrix`, `DifferentialMatrix`, entre outras.

### 🛠️ Instalação e Execução

1. Certifique-se de que você possui o código fonte do **SeReM**.
2. Clone este repositório.
3. **Copie todos os arquivos deste projeto (`.m` e `.json`) para dentro da pasta do SeReM** (ou certifique-se de que a pasta `../SeReM/` esteja acessível no *path*).

A estrutura de pastas deve ficar semelhante a esta:

```text
SeReM/
├── config.json            # Arquivo de configuração dos parâmetros
├── UHCMAESV3.m            # Script principal de otimização
└── PlotResults.m          # Script para gerar gráficos pós-execução

```

## 🚀 Como Usar

1. Abra o MATLAB ou Octave.
2. Configure os parâmetros da simulação no arquivo `config.json` (se necessário).
3. Execute o script principal:
```matlab
UHCMAESV3

```


4. Os resultados serão salvos automaticamente na pasta `Results/`, organizados por data e hora.

## 📊 Visualização

Para visualizar os gráficos de uma execução passada:

1. Rode o script `PlotResults`.
2. Uma janela abrirá pedindo para selecionar o arquivo `.mat`.
3. Navegue até `Results/<TIMESTAMP>/` e selecione o arquivo `run_data.mat`.