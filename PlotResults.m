% =========================================================================
% Script: PlotResults.m
% Objetivo: Carregar binário (.mat) de uma execução e gerar diagnósticos
% =========================================================================

clear all; close all; clc;

% 1. SELEÇÃO DO ARQUIVO DE RESULTADOS
fprintf('Selecione o arquivo .mat da execução desejada...\n');
[file, path] = uigetfile('*.mat', 'Selecione o arquivo de resultado (run_*.mat)');

if isequal(file, 0)
    disp('Seleção cancelada.');
    return;
end

fullpath = fullfile(path, file);
fprintf('Carregando resultados de: %s\n', file);
load(fullpath); % Carrega: history, x_new, cfg

% 2. CARREGAMENTO DOS DADOS ORIGINAIS (GROUND TRUTH)
% Precisamos carregar o data3.mat para ter o Vp, Vs, Rho REAIS e o Tempo
% Usamos o caminho salvo na struct 'cfg' para garantir consistência
try
    data_file = [cfg.files.data_folder cfg.files.input_filename];
    fprintf('Carregando dados originais de: %s\n', data_file);
    load(data_file); % Carrega: Vp, Vs, Rho, TimeSeis, Snear, ...
catch
    % Fallback caso a estrutura de pastas tenha mudado
    warning('Não foi possível carregar via cfg. Tentando carregar "data3.mat" localmente.');
    load('data3.mat');
end

% 3. RECONSTRUÇÃO DAS DIMENSÕES E ALINHAMENTO
% Recalcula dimensões baseadas no dado carregado
nm = size(Snear, 1) + 1;
limit_plot = nm - 1;

% Saneamento de dimensões (Regra N vs N+1 discutida anteriormente)
% Garante que os vetores REAIS tenham o mesmo tamanho dos INVERTIDOS
if length(Vp) > limit_plot
    VpReal = Vp(1:limit_plot);
    VsReal = Vs(1:limit_plot);
    RhoReal = Rho(1:limit_plot);
    time_plot = TimeSeis(1:limit_plot);
else
    VpReal = Vp; VsReal = Vs; RhoReal = Rho; time_plot = TimeSeis;
end

% Determina o número real de gerações (caso tenha parado antes)
stop_generations = length(history.fitness);

% =========================================================
% PLOTAGEM (CÓDIGO SOLICITADO)
% =========================================================

% Recupera as soluções do vetor único x_new
% Estrutura: [Vp (1..nm); Vs (1..nm); Rho (1..nm)]
VpSol = x_new(1 : limit_plot);
VsSol = x_new(nm + 1 : nm + limit_plot);
RhoSol = x_new(2*nm + 1 : 2*nm + limit_plot);

% Recupera a média das soluções do vetor history.solutions
if isfield(history, 'solutions') && ~isempty(history.solutions)
    mean_history_solutions = mean(history.solutions, 2);
    VpMean = mean_history_solutions(1 : limit_plot);
    VsMean = mean_history_solutions(nm + 1 : nm + limit_plot);
    RhoMean = mean_history_solutions(2*nm + 1 : 2*nm + limit_plot);
else
    % Fallback se não houver histórico
    VpMean = VpSol; VsMean = VsSol; RhoMean = RhoSol;
end

% --- DIAGNÓSTICO DO ALGORITMO ---
figure('Name', ['Diagnóstico: ' file], 'Color', 'w', 'Position', [50, 50, 1000, 600]);

subplot(2,2,1);
plot(history.fitness, 'LineWidth', 2);
title('Convergência (Misfit + Regularização)');
grid on; set(gca, 'YScale', 'log');
ylabel('Log Cost'); xlabel('Geração');

subplot(2,2,2);
plot(history.t_eval, 'r', 'LineWidth', 2);
title('Incerteza: Amostras (t_{eval})');
grid on; ylabel('Samples'); xlabel('Geração');

subplot(2,2,3);
plot(history.diversity, 'm', 'LineWidth', 2);
title('Diversidade (Euclidiana)');
grid on; ylabel('Distância Média'); xlabel('Geração');

subplot(2,2,4);
% Verifica se reg_weight foi salvo no histórico (versões mais novas)
if isfield(history, 'reg_weight')
    plot(history.reg_weight, 'k--', 'LineWidth', 2);
    title('Peso da Regularização');
    grid on; ylabel('Alpha Reg'); xlabel('Geração');
end


% --- RESULTADOS DA INVERSÃO (Vp, Vs, Rho) ---
figure('Name', ['Evolução da Inversão: ' file], 'Color', 'w', 'Position', [100, 100, 1200, 700]);

% Definição das Cores para Evolução
num_snapshots = size(history.solutions, 2);
if num_snapshots == 0
    warning('Nenhum snapshot intermediário salvo. Plotando apenas o final.');
    num_snapshots = 1;
    history.solutions = x_new;
end

colors = winter(num_snapshots + 2);

% --- PLOT 1: Velocidade P (Vp) ---
subplot(1,3,1); hold on;
for i = 1:num_snapshots
    sol_i = history.solutions(:, i);
    vp_i = sol_i(1 : limit_plot);
    % Apenas RGB (compatível com Octave/Matlab)
    plot(vp_i, time_plot, 'Color', colors(i,:), 'LineWidth', 1);
end
h_mean = plot(VpMean, time_plot, 'y--', 'LineWidth', 2);
h_final = plot(VpSol, time_plot, 'r', 'LineWidth', 3);
h_real = plot(VpReal, time_plot, 'k', 'LineWidth', 2);

set(gca, 'YDir', 'reverse'); grid on; xlabel('Vp (km/s)'); ylabel('Tempo (s)');
title('Evolução Vp');
legend([h_real, h_mean, h_final], {'Real (Poço)', 'Média Soluções', 'Melhor Solução'}, 'Location', 'SouthWest');

% --- PLOT 2: Velocidade S (Vs) ---
subplot(1,3,2); hold on;
for i = 1:num_snapshots
    sol_i = history.solutions(:, i);
    vs_i = sol_i(nm + 1 : nm + limit_plot);
    plot(vs_i, time_plot, 'Color', colors(i,:), 'LineWidth', 1);
end
plot(VsMean, time_plot, 'y--', 'LineWidth', 2);
plot(VsSol, time_plot, 'r', 'LineWidth', 3);
plot(VsReal, time_plot, 'k', 'LineWidth', 2);

set(gca, 'YDir', 'reverse'); grid on; xlabel('Vs (km/s)');
title('Evolução Vs'); set(gca, 'YTickLabel', []);

% --- PLOT 3: Densidade (Rho) ---
subplot(1,3,3); hold on;
for i = 1:num_snapshots
    sol_i = history.solutions(:, i);
    rho_i = sol_i(2*nm + 1 : 2*nm + limit_plot);
    plot(rho_i, time_plot, 'Color', colors(i,:), 'LineWidth', 1);
end
plot(RhoMean, time_plot, 'y--', 'LineWidth', 2);
plot(RhoSol, time_plot, 'r', 'LineWidth', 3);
plot(RhoReal, time_plot, 'k', 'LineWidth', 2);

set(gca, 'YDir', 'reverse'); grid on; xlabel('Densidade (g/cc)');
title('Evolução \rho'); set(gca, 'YTickLabel', []);

% Barra de cores
colormap(winter);
c = colorbar('Position', [0.92 0.11 0.02 0.81]);
try
    ylabel(c, 'Evolução das Gerações (Antigo -> Recente)');
catch
end
caxis([0 stop_generations]);
