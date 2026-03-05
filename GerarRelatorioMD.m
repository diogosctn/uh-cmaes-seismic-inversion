% =========================================================================
% Script: GerarRelatorioMD.m
% Objetivo: Ler pastas de resultados, gerar os gráficos silenciosamente e
% criar um relatório Markdown final com TODOS os parâmetros dinamicamente.
% =========================================================================

clear; clc; close all;

% =========================================================================
% FUNÇÕES AUXILIARES LOCAIS
% =========================================================================

function print_struct_to_md(fid, s, prefix)
    % Lê todos os campos da estrutura recursivamente e os escreve no MD
    fields = fieldnames(s);
    for i = 1:numel(fields)
        f = fields{i};
        val = s.(f);

        % Monta o nome do campo (ex: cmaes.sigma_initial)
        if isempty(prefix)
            full_name = f;
        else
            full_name = [prefix, '.', f];
        end

        if isstruct(val)
            % Se for uma sub-pasta (ex: cfg.cmaes), entra recursivamente
            print_struct_to_md(fid, val, full_name);
        else
            % Se for um valor final, converte para texto para imprimir
            if isnumeric(val)
                if numel(val) > 1
                    val_str = mat2str(val); % Trata arrays como [15 30 45]
                else
                    val_str = num2str(val); % Trata números isolados
                end
            elseif ischar(val) || isstring(val)
                val_str = char(val); % Trata textos
            elseif islogical(val)
                if val, val_str = 'true'; else, val_str = 'false'; end % Trata booleanos
            else
                val_str = 'Desconhecido';
            end

            % Escreve a linha na tabela Markdown
            fprintf(fid, '| `%s` | %s |\n', full_name, val_str);
        end
    end
end

results_dir = 'Results';
output_md = 'Relatorio_Resultados.md';

% Verifica se a pasta Results existe
if ~exist(results_dir, 'dir')
    error('A pasta "%s" não foi encontrada. Rode um experimento primeiro.', results_dir);
end

% Lista todas as subpastas dentro de Results
folders = dir(results_dir);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name}, {'.', '..'}));

if isempty(folders)
    error('Nenhuma subpasta de experimento encontrada em "%s".', results_dir);
end

% Abre o arquivo Markdown para escrita
fid_md = fopen(output_md, 'w', 'n', 'UTF-8');
fprintf(fid_md, '# Relatório de Experimentos: UH-CMA-ES\n\n');
fprintf(fid_md, 'Este documento foi gerado automaticamente pelo MATLAB e consolida os resultados de todas as execuções salvas.\n\n');
fprintf(fid_md, '---\n\n');

fprintf('Iniciando geração do relatório. %d experimentos encontrados.\n', length(folders));

% Loop por cada pasta de resultado
for i = 1:length(folders)
    run_name = folders(i).name;
    run_path = fullfile(results_dir, run_name);
    mat_file = fullfile(run_path, 'run_data.mat');

    if ~exist(mat_file, 'file')
        fprintf(' Ignorando "%s" (run_data.mat não encontrado).\n', run_name);
        continue;
    end

    fprintf('Processando experimento: %s...\n', run_name);
    fprintf(fid_md, '## Execução: `%s`\n\n', run_name);

    % --- 1. CARREGAMENTO DOS DADOS ---
    data_exp = load(mat_file);
    cfg = data_exp.cfg;
    history = data_exp.history;
    x_new = data_exp.x_new;

    % --- 2. TABELA DINÂMICA COM TODOS OS PARÂMETROS ---
    fprintf(fid_md, '### Configurações Utilizadas\n\n');
    fprintf(fid_md, '| Parâmetro (Caminho JSON) | Valor |\n');
    fprintf(fid_md, '|---|---|\n');

    % Chama a função auxiliar para imprimir todos os campos da struct cfg
    print_struct_to_md(fid_md, cfg, '');
    fprintf(fid_md, '\n');

    % --- 3. PREPARAÇÃO DOS DADOS PARA PLOTAGEM ---
    try
        data_file = [cfg.files.data_folder cfg.files.input_filename];
        true_data = load(data_file);
    catch
        true_data = load('data3.mat'); % Fallback
    end

    nm = size(true_data.Snear, 1) + 1;
    limit_plot = nm - 1;
    time_plot = true_data.TimeSeis(1:limit_plot);

    if length(true_data.Vp) > limit_plot
        VpReal = true_data.Vp(1:limit_plot);
        VsReal = true_data.Vs(1:limit_plot);
        RhoReal = true_data.Rho(1:limit_plot);
    else
        VpReal = true_data.Vp; VsReal = true_data.Vs; RhoReal = true_data.Rho;
    end

    VpSol = x_new(1 : limit_plot);
    VsSol = x_new(nm + 1 : nm + limit_plot);
    RhoSol = x_new(2*nm + 1 : 2*nm + limit_plot);

    if isfield(history, 'solutions') && ~isempty(history.solutions)
        mean_history_solutions = mean(history.solutions, 2);
        VpMean = mean_history_solutions(1 : limit_plot);
        VsMean = mean_history_solutions(nm + 1 : nm + limit_plot);
        RhoMean = mean_history_solutions(2*nm + 1 : 2*nm + limit_plot);
        num_snapshots = size(history.solutions, 2);
    else
        VpMean = VpSol; VsMean = VsSol; RhoMean = RhoSol;
        num_snapshots = 1;
        history.solutions = x_new;
    end

    stop_generations = length(history.fitness);

    % --- 4. GERAÇÃO DO GRÁFICO 1 (DIAGNÓSTICO) SILENCIOSAMENTE ---
    fig1 = figure('Visible', 'off', 'Color', 'w', 'Position', [50, 50, 1000, 600]);

    subplot(2,2,1); plot(history.fitness, 'LineWidth', 2); title('Convergência'); grid on; set(gca, 'YScale', 'log');
    subplot(2,2,2); plot(history.t_eval, 'r', 'LineWidth', 2); title('Amostras (t_{eval})'); grid on;
    subplot(2,2,3); plot(history.diversity, 'm', 'LineWidth', 2); title('Diversidade'); grid on;
    subplot(2,2,4);
    if isfield(history, 'reg_weight'), plot(history.reg_weight, 'k--', 'LineWidth', 2); title('Regularização'); grid on; end

    % Salvar Imagem 1
    img1_name = 'diagnostico.png';
    img1_path = fullfile(run_path, img1_name);
    try exportgraphics(fig1, img1_path, 'Resolution', 300); catch; print(fig1, img1_path, '-dpng', '-r300'); end
    close(fig1);

    % --- 5. GERAÇÃO DO GRÁFICO 2 (EVOLUÇÃO) SILENCIOSAMENTE ---
    fig2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1200, 700]);
    colors = winter(num_snapshots + 2);

    subplot(1,3,1); hold on;
    for k = 1:num_snapshots, plot(history.solutions(1:limit_plot, k), time_plot, 'Color', colors(k,:), 'LineWidth', 1); end
    plot(VpMean, time_plot, 'y--', 'LineWidth', 2); plot(VpSol, time_plot, 'r', 'LineWidth', 3); plot(VpReal, time_plot, 'k', 'LineWidth', 2);
    set(gca, 'YDir', 'reverse'); grid on; title('Vp');

    subplot(1,3,2); hold on;
    for k = 1:num_snapshots, plot(history.solutions(nm+1:nm+limit_plot, k), time_plot, 'Color', colors(k,:), 'LineWidth', 1); end
    plot(VsMean, time_plot, 'y--', 'LineWidth', 2); plot(VsSol, time_plot, 'r', 'LineWidth', 3); plot(VsReal, time_plot, 'k', 'LineWidth', 2);
    set(gca, 'YDir', 'reverse'); grid on; title('Vs'); set(gca, 'YTickLabel', []);

    subplot(1,3,3); hold on;
    for k = 1:num_snapshots, plot(history.solutions(2*nm+1:2*nm+limit_plot, k), time_plot, 'Color', colors(k,:), 'LineWidth', 1); end
    plot(RhoMean, time_plot, 'y--', 'LineWidth', 2); plot(RhoSol, time_plot, 'r', 'LineWidth', 3); plot(RhoReal, time_plot, 'k', 'LineWidth', 2);
    set(gca, 'YDir', 'reverse'); grid on; title('\rho'); set(gca, 'YTickLabel', []);

    % Salvar Imagem 2
    img2_name = 'evolucao_perfis.png';
    img2_path = fullfile(run_path, img2_name);
    try exportgraphics(fig2, img2_path, 'Resolution', 300); catch; print(fig2, img2_path, '-dpng', '-r300'); end
    close(fig2);

    % --- 6. ANEXAR IMAGENS NO MARKDOWN ---
    img1_rel = strrep(sprintf('%s/%s/%s', results_dir, run_name, img1_name), '\', '/');
    img2_rel = strrep(sprintf('%s/%s/%s', results_dir, run_name, img2_name), '\', '/');

    fprintf(fid_md, '### Gráficos de Resultado\n\n');
    fprintf(fid_md, '**Diagnóstico de Convergência**\n\n');
    fprintf(fid_md, '![Diagnóstico](%s)\n\n', img1_rel);
    fprintf(fid_md, '**Evolução dos Perfis Elásticos**\n\n');
    fprintf(fid_md, '![Evolução](%s)\n\n', img2_rel);
    fprintf(fid_md, '---\n\n');
end

fclose(fid_md);
fprintf('\n Concluído! O relatório Markdown foi gerado com sucesso: %s\n', output_md);

