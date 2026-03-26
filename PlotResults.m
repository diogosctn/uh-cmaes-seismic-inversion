% =========================================================================
% Script: PlotResults.m
% Objetivo: Carregar binário (.mat) de uma execução e gerar diagnósticos
% =========================================================================

clear; clc; close all;

pkg load signal

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
try
    data_file = [cfg.files.data_folder cfg.files.input_filename];
    fprintf('Carregando dados originais de: %s\n', data_file);
    load(data_file);
catch
    warning('Não foi possível carregar via cfg. Tentando carregar "data3.mat" localmente.');
    load('data3.mat');
end

% 3. RECONSTRUÇÃO DAS DIMENSÕES E ALINHAMENTO
fprintf('Saneando dimensões dos vetores...\n');
nm = size(Snear, 1) + 1;
limit_plot = nm - 1;

if length(Vp) > limit_plot
    VpReal = Vp(1:limit_plot);
    VsReal = Vs(1:limit_plot);
    RhoReal = Rho(1:limit_plot);
    time_plot = TimeSeis(1:limit_plot);
else
    VpReal = Vp; VsReal = Vs; RhoReal = Rho; time_plot = TimeSeis;
end

stop_generations = length(history.fitness);

% Recupera as soluções
VpSol = x_new(1 : limit_plot);
VsSol = x_new(nm + 1 : nm + limit_plot);
RhoSol = x_new(2*nm + 1 : 2*nm + limit_plot);

if isfield(history, 'ensamble') && ~isempty(history.ensamble)
    mean_history_ensamble = mean(history.ensamble, 2);
    VpMean = mean_history_ensamble(1 : limit_plot);
    VsMean = mean_history_ensamble(nm + 1 : nm + limit_plot);
    RhoMean = mean_history_ensamble(2*nm + 1 : 2*nm + limit_plot);
else
    VpMean = VpSol; VsMean = VsSol; RhoMean = RhoSol;
end

% =========================================================
% EXTRAÇÃO DO PRIOR (FILTRAGEM COM BASE NO UHCMAES)
% =========================================================
fprintf('Extraindo modelo a priori (filtrado)...\n');
try
    nfilt = cfg.physics.prior_filter_order;
    cutofffr = cfg.physics.prior_cutoff_freq;
    [b, a] = butter(nfilt, cutofffr);

    VpPrior = filtfilt(b, a, Vp);
    VsPrior = filtfilt(b, a, Vs);
    RhoPrior = filtfilt(b, a, Rho);

    % Alinhamento das dimensões do prior
    if length(VpPrior) > limit_plot
        VpPrior = VpPrior(1:limit_plot);
        VsPrior = VsPrior(1:limit_plot);
        RhoPrior = RhoPrior(1:limit_plot);
    end
catch ME
    warning('Aviso: Não foi possível aplicar o filtro com os dados de cfg. Usando default...');
    [b, a] = butter(2, 0.05); % Filtro padrão se der erro
    VpPrior = filtfilt(b, a, VpReal);
    VsPrior = filtfilt(b, a, VsReal);
    RhoPrior = filtfilt(b, a, RhoReal);
end

% =========================================================
% PLOTAGEM 1: DIAGNÓSTICO
% =========================================================
fprintf('Gerando Figura 1: Diagnóstico do Algoritmo...\n');
figure('Name', ['Diagnóstico: ' file], 'Color', 'w', 'Position', [50, 50, 1000, 600]);

subplot(2,2,1); plot(history.fitness, 'LineWidth', 2);
title('Convergência'); grid on; set(gca, 'YScale', 'log');

subplot(2,2,2); plot(history.t_eval, 'r', 'LineWidth', 2);
title('Amostras (t_{eval})'); grid on;

subplot(2,2,3); plot(history.diversity, 'm', 'LineWidth', 2);
title('Diversidade'); grid on;

if isfield(history, 'reg_weight')
    subplot(2,2,4); plot(history.reg_weight, 'k--', 'LineWidth', 2);
    title('Peso da Regularização'); grid on;
end
drawnow;

% =========================================================
% PLOTAGEM 2: EVOLUÇÃO
% =========================================================
fprintf('Gerando Figura 2: Evolução da Inversão...\n');
figure('Name', ['Evolução da Inversão: ' file], 'Color', 'w', 'Position', [100, 100, 1200, 700]);

num_snapshots = size(history.solutions, 2);
if num_snapshots == 0
    num_snapshots = 1;
    history.solutions = x_new;
end
colors = winter(num_snapshots + 2);

% Vp
subplot(1,3,1); hold on;
for i = 1:num_snapshots
    sol_i = history.solutions(:, i);
    plot(sol_i(1 : limit_plot), time_plot, 'Color', colors(i,:), 'LineWidth', 1);
end
plot(VpSol, time_plot, 'r', 'LineWidth', 3);
plot(VpReal, time_plot, 'k', 'LineWidth', 2);
set(gca, 'YDir', 'reverse'); grid on; xlabel('Vp'); title('Vp');

% Vs
subplot(1,3,2); hold on;
for i = 1:num_snapshots
    sol_i = history.solutions(:, i);
    plot(sol_i(nm + 1 : nm + limit_plot), time_plot, 'Color', colors(i,:), 'LineWidth', 1);
end
plot(VsSol, time_plot, 'r', 'LineWidth', 3);
plot(VsReal, time_plot, 'k', 'LineWidth', 2);
set(gca, 'YDir', 'reverse'); grid on; xlabel('Vs'); title('Vs');

% Rho
subplot(1,3,3); hold on;
for i = 1:num_snapshots
    sol_i = history.solutions(:, i);
    plot(sol_i(2*nm + 1 : 2*nm + limit_plot), time_plot, 'Color', colors(i,:), 'LineWidth', 1);
end
plot(RhoSol, time_plot, 'r', 'LineWidth', 3);
plot(RhoReal, time_plot, 'k', 'LineWidth', 2);
set(gca, 'YDir', 'reverse'); grid on; xlabel('Rho'); title('\rho');

try
    colormap(winter);
    caxis([0 max(1, stop_generations)]);
catch ME
    fprintf('Aviso ao aplicar colormap na Figura 2: %s\n', ME.message);
end
drawnow;

% =========================================================
% PLOTAGEM 3: ENSAMBLE + PRIOR
% =========================================================
fprintf('Gerando Figura 3: Ensamble + Prior...\n');

if ~isfield(history, 'ensamble') || isempty(history.ensamble)
    fprintf('AVISO: O campo history.ensamble está vazio. Figura 3 cancelada.\n');
else
    try
        figure('Name', ['Ensamble e Prior: ' file], 'Color', 'w', 'Position', [150, 150, 1200, 700]);

        num_individuals = size(history.ensamble, 2);
        block_size = size(history.ensamble, 1) / 3;

        % --- Vp Ensamble ---
        subplot(1,3,1); hold on;
        for i = 1:num_individuals
            sol_i = history.ensamble(:, i);
            plot(sol_i(1 : limit_plot), time_plot, 'Color', [0.6 0.6 1.0], 'LineWidth', 0.5);
        end
        h_mean  = plot(VpMean, time_plot, 'y', 'LineWidth', 2);
        h_prior = plot(VpPrior, time_plot, 'g', 'LineWidth', 2);
        h_best  = plot(VpSol, time_plot, 'r', 'LineWidth', 2);
        h_real  = plot(VpReal, time_plot, 'k', 'LineWidth', 2);
        set(gca, 'YDir', 'reverse'); grid on;
        xlabel('Vp (km/s)'); ylabel('Tempo (s)'); title('Ensamble Vp');
        legend([h_real, h_best, h_prior, h_mean], {'Real', 'Melhor Sol.', 'Prior (Filtrado)', 'Média do Ensamble'}, 'Location', 'SouthWest');

        % --- Vs Ensamble ---
        subplot(1,3,2); hold on;
        for i = 1:num_individuals
            sol_i = history.ensamble(:, i);
            plot(sol_i(block_size + 1 : block_size + limit_plot), time_plot, 'Color', [0.6 0.6 1.0], 'LineWidth', 0.5);
        end
        h_mean_s = plot(VsMean, time_plot, 'y', 'LineWidth', 2);
        h_prior_s = plot(VsPrior, time_plot, 'g', 'LineWidth', 2);
        h_best_s  = plot(VsSol, time_plot, 'r', 'LineWidth', 2);
        h_real_s  = plot(VsReal, time_plot, 'k', 'LineWidth', 2);
        set(gca, 'YDir', 'reverse'); grid on;
        xlabel('Vs (km/s)'); title('Ensamble Vs');
        legend([h_real_s, h_best_s, h_prior_s, h_mean_s], {'Real', 'Melhor Sol.', 'Prior (Filtrado)', 'Média do Ensamble'}, 'Location', 'SouthWest');

        % --- Rho Ensamble ---
        subplot(1,3,3); hold on;
        for i = 1:num_individuals
            sol_i = history.ensamble(:, i);
            plot(sol_i(2*block_size + 1 : 2*block_size + limit_plot), time_plot, 'Color', [0.6 0.6 1.0], 'LineWidth', 0.5);
        end
        h_mean_r  = plot(RhoMean, time_plot, 'y', 'LineWidth', 2);
        h_prior_r = plot(RhoPrior, time_plot, 'g', 'LineWidth', 2);
        h_best_r  = plot(RhoSol, time_plot, 'r', 'LineWidth', 2);
        h_real_r  = plot(RhoReal, time_plot, 'k', 'LineWidth', 2);
        set(gca, 'YDir', 'reverse'); grid on;
        xlabel('Densidade (g/cc)'); title('Ensamble \rho');
        legend([h_real_r, h_best_r, h_prior_r, h_mean_r], {'Real', 'Melhor Sol.', 'Prior (Filtrado)', 'Média do Ensamble'}, 'Location', 'SouthWest');

        drawnow;
        fprintf('Figura 3 gerada com sucesso!\n');
    catch ME
        fprintf('\n>>> ERRO FATAL AO GERAR ENSAMBLE <<<\n');
        fprintf('Mensagem de Erro: %s\n', ME.message);
        fprintf('Linha do erro: %d\n', ME.stack(1).line);
    end
end

fprintf('Script finalizado.\n');
