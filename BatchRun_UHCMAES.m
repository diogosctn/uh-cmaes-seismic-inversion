% =========================================================================
% Script: BatchRun_UHCMAES.m (VERSÃO OCTAVE)
% Objetivo: Executar o algoritmo UHCMAES multiplas vezes via CSV
% Nota: REMOVA ou COMENTE o 'clear all' dentro do UHCMAES.m antes de rodar!
% =========================================================================

clear all; close all; clc;

diary('BatchRun_UHCMAES.log');
diary on;

% 1. Configurações Iniciais
script_name = 'UHCMAES'; % Nome do script principal (sem .m)
config_file = 'config.json';
backup_file = 'config_backup.json';
csv_file    = 'Experiments/experiments3.csv'; % Arquivo com a matriz de experimentos

% Verifica se os arquivos necessários existem
if ~exist(config_file, 'file')
    error('Arquivo %s não encontrado.', config_file);
    diary off
end
if ~exist(csv_file, 'file')
    error('Arquivo %s não encontrado. Crie o CSV com os cenários.', csv_file);
    diary off
end

% Faz backup do config original para restaurar no final
copyfile(config_file, backup_file);
fprintf('Backup da configuração original salvo em %s\n', backup_file);

% Carrega a configuração base (Octave 7+ suporta nativamente)
base_cfg = jsondecode(fileread(config_file));

% =========================================================================
% FUNÇÕES AUXILIARES LOCAIS (Adaptadas para Octave)
% =========================================================================

function b = ismissing_custom(val)
    % Verifica se um valor é efetivamente vazio/nulo no Octave
    b = false;
    if ischar(val)
        if isempty(strtrim(val))
            b = true;
        end
    elseif isnumeric(val)
        if isnan(val)
            b = true;
        end
    elseif isempty(val)
        b = true;
    end
end

function cfg = set_nested_field(cfg, path_str, val)
    % Navega na struct a partir de uma string com pontos (ex: 'cmaes.sigma_initial')
    parts = strsplit(path_str, '.');
    switch length(parts)
        case 1
            cfg.(parts{1}) = val;
        case 2
            cfg.(parts{1}).(parts{2}) = val;
        case 3
            cfg.(parts{1}).(parts{2}).(parts{3}) = val;
        case 4
            cfg.(parts{1}).(parts{2}).(parts{3}).(parts{4}) = val;
        otherwise
            error('Profundidade de parâmetro %s excede o limite (4 níveis).', path_str);
            diary off
    end
end

% =========================================================================
% LEITURA CUSTOMIZADA DO CSV PARA OCTAVE
% =========================================================================
fid = fopen(csv_file, 'r');
lines = {};
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && ~isempty(strtrim(line))
        lines{end+1} = line;
    end
end
fclose(fid);

% Converte as linhas em uma matriz de células (cell array)
num_rows = length(lines);
num_cols = length(strsplit(lines{1}, ',', 'CollapseDelimiters', false));
data = cell(num_rows, num_cols);

for i = 1:num_rows
    cols = strsplit(lines{i}, ',', 'CollapseDelimiters', false);
    for j = 1:num_cols
        if j <= length(cols)
            val = strtrim(cols{j});
            num_val = str2double(val);
            if ~isnan(num_val) && ~isempty(val) % É um número válido
                data{i,j} = num_val;
            else
                data{i,j} = val; % Mantém como string ou vazio
            end
        else
            data{i,j} = ''; % Preenche colunas faltantes com vazio
        end
    end
end

headers = data(1, :);
num_exps = size(data, 1) - 1;

fprintf('Iniciando bateria de %d experimentos via CSV no Octave...\n\n', num_exps);

% =========================================================================
% 2. LOOP DE EXECUÇÃO
% =========================================================================

for i = 1:num_exps
    exp_name = data{i+1, 1};

    fprintf('---------------------------------------------------\n');
    fprintf('Rodando Experimento %d/%d: %s\n', i, num_exps, exp_name);
    fprintf('---------------------------------------------------\n');

    % Copia a base e sobrescreve apenas os campos definidos no CSV
    current_cfg = base_cfg;

    % Itera sobre todas as colunas de parâmetros (a partir da coluna 2)
    for j = 2:length(headers)
        param_path = headers{j};
        param_val = data{i+1, j};

        % Ignora a célula se estiver vazia no CSV
        if ismissing_custom(param_val)
            continue;
        end

        % Atualiza o campo aninhado na struct
        current_cfg = set_nested_field(current_cfg, param_path, param_val);

        if isnumeric(param_val)
            fprintf('   -> %s = %g\n', param_path, param_val);
        else
            fprintf('   -> %s = %s\n', param_path, param_val);
        end
    end

    % Salva o config.json temporário
    json_str = jsonencode(current_cfg);

    fid = fopen(config_file, 'w');
    if fid == -1, error('Não foi possível escrever no config.json');diary off; end
    fprintf(fid, '%s', json_str);
    fclose(fid);

    % Executa o Algoritmo
    try
        pause(1); % Pausa breve para garantir I/O de disco
        eval(script_name);
        fprintf('\n>>> Experimento %s concluído com sucesso.\n', exp_name);
    catch ME
        fprintf('\n!!! Erro durante o experimento %s:\n%s\n', exp_name, ME.message);
    end

    % Limpa variáveis do script filho, mantendo as essenciais do loop
    clearvars -except csv_file data headers num_exps i base_cfg config_file backup_file script_name
    close all;
end

% =========================================================================
% 3. FINALIZAÇÃO
% =========================================================================

copyfile(backup_file, config_file);
delete(backup_file);

fprintf('\n===================================================\n');
fprintf('Bateria de testes finalizada.\n');
fprintf('Configuração original restaurada.\n');
diary off
