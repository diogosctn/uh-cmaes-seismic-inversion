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
├── config.json            # Arquivo de configuração dos parâmetros base
├── UHCMAES.m              # Script principal de otimização
├── BatchRun_UHCMAES.m     # Script para automação de múltiplos cenários
└── PlotResults.m          # Script para gerar gráficos pós-execução

```

## 🚀 Como Usar (Execução Única)

1. Abra o MATLAB ou Octave.
2. Configure os parâmetros da simulação no arquivo `config.json` (se necessário).
3. Execute o script principal:
```matlab
UHCMAES

```


4. Os resultados serão salvos automaticamente na pasta `Results/`, organizados por data e hora.

## 🔄 Execução em Lote (Batch Run)

Se você deseja rodar o algoritmo diversas vezes consecutivas testando diferentes configurações de parâmetros (ex: variando o Sigma ou o nível de ruído), utilize o script **`BatchRun_UHCMAES.m`**.

**⚠️ ATENÇÃO ESTRUTURAL:** Antes de utilizar o script de Batch, você **DEVE** abrir o arquivo `UHCMAES.m` e comentar a linha que contém `clear all;` no início do código, caso contrário o loop do Batch será apagado da memória.

```matlab
% clear all; close all; clc; % <-- Deixe assim no UHCMAES.m

```

**Como configurar os testes:**

1. Abra o arquivo `BatchRun_UHCMAES.m`.
2. Vá até a seção **`2. DEFINIÇÃO DOS CENÁRIOS`**.
3. Adicione ou modifique os blocos de experimentos definindo os parâmetros que deseja alterar em relação ao `config.json` base. Exemplo:
```matlab
exp_count = exp_count + 1;
experiments(exp_count).name = 'Meu_Novo_Teste';
experiments(exp_count).params.cmaes.sigma_initial = 2.5;

```


4. Execute o script `BatchRun_UHCMAES`.
5. O script fará um backup da sua configuração original, rodará todos os cenários gerando pastas de resultados independentes e, ao final, restaurará seu `config.json` original.

## 📊 Visualização

Para visualizar os gráficos de uma execução passada:

1. Rode o script `PlotResults`.
2. Uma janela abrirá pedindo para selecionar o arquivo `.mat`.
3. Navegue até `Results/<TIMESTAMP>/` e selecione o arquivo `run_data.mat`.