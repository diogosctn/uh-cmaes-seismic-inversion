% =========================================================================
% Script: BatchRun_UHCMAES.m
% Objetivo: Executar o algoritmo UHCMAES multiplas vezes variando parametros
% Nota: REMOVA ou COMENTE o 'clear all' dentro do UHCMAES.m antes de rodar!
% =========================================================================

clear all; close all; clc;

% 1. Configurações Iniciais
script_name = 'UHCMAES'; % Nome do script principal (sem .m)
config_file = 'config.json';
backup_file = 'config_backup.json';

% Verifica se o arquivo de config existe
if ~exist(config_file, 'file')
    error('Arquivo config.json não encontrado.');
end

% Faz backup do config original para restaurar no final
copyfile(config_file, backup_file);
fprintf('Backup da configuração original salvo em %s\n', backup_file);

% Carrega a configuração base
base_cfg = jsondecode(fileread(config_file));

% -------------------------------------------------------------------------
% Função Auxiliar Local para atualizar structs aninhadas
% -------------------------------------------------------------------------
function base = update_struct(base, changes)
    fields = fieldnames(changes);
    for k = 1:numel(fields)
        f = fields{k};
        if isstruct(changes.(f)) && isfield(base, f) && isstruct(base.(f))
            % Chamada recursiva para sub-estruturas (ex: params.uh.noise)
            base.(f) = update_struct(base.(f), changes.(f));
        else
            % Sobrescreve valor
            base.(f) = changes.(f);
            fprintf('   -> Parametro alterado: %s = %s\n', f, mat2str(changes.(f)));
        end
    end
end

% =========================================================================
% 2. DEFINIÇÃO DOS CENÁRIOS (EXPERIMENTOS)
% =========================================================================
% Aqui você define quais parâmetros quer variar.
% Crie um array de estruturas 'experiments'.

experiments = [];
exp_count = 0;

% --- Cenário 1: Sigma Baixo (Exploração Local) ---
exp_count = exp_count + 1;
experiments(exp_count).name = 'Low_Sigma';
experiments(exp_count).params.cmaes.sigma_initial = 0.5;
experiments(exp_count).params.uh.noise_level = 0.05;

% --- Cenário 2: Sigma Alto (Exploração Global) ---
exp_count = exp_count + 1;
experiments(exp_count).name = 'High_Sigma';
experiments(exp_count).params.cmaes.sigma_initial = 5.0;
experiments(exp_count).params.uh.noise_level = 0.05;

% --- Cenário 3: Ruído Alto (Teste de Robustez UH) ---
exp_count = exp_count + 1;
experiments(exp_count).name = 'High_Noise_UH';
experiments(exp_count).params.cmaes.sigma_initial = 2.0;
experiments(exp_count).params.uh.noise_level = 0.20; % 20% de ruído
experiments(exp_count).params.uh.t_max = 100;        % Permite mais reamostragens

% --- Cenário 4: População Maior (Alterando Configurações não listadas acima) ---
% Nota: O cálculo de lambda é feito no código MATLAB, mas se você passar
% parâmetros que afetam o lambda (como stop_generations ou outros do json),
% defina aqui. Como exemplo, vamos mudar a tolerância.
exp_count = exp_count + 1;
experiments(exp_count).name = 'High_Precision';
experiments(exp_count).params.cmaes.stop_tol_diversity = 1e-4;
experiments(exp_count).params.cmaes.stop_generations = 800;


% =========================================================================
% 3. LOOP DE EXECUÇÃO
% =========================================================================

total_exps = length(experiments);
fprintf('Iniciando bateria de %d experimentos...\n\n', total_exps);

for i = 1:total_exps
    current_exp = experiments(i);
    fprintf('---------------------------------------------------\n');
    fprintf('Rodando Experimento %d/%d: %s\n', i, total_exps, current_exp.name);
    fprintf('---------------------------------------------------\n');

    % 3.1. Prepara a Configuração Atual
    % Copia a base e sobrescreve apenas os campos definidos no experimento
    current_cfg = base_cfg;

    % Função recursiva simples para atualizar campos aninhados (structs)
    current_cfg = update_struct(current_cfg, current_exp.params);

    % 3.2. Salva o config.json temporário
    % Usa formatação bonita se disponível (Matlab R2016b+)
    try
        json_str = jsonencode(current_cfg, 'PrettyPrint', true);
    catch
        json_str = jsonencode(current_cfg);
    end

    fid = fopen(config_file, 'w');
    if fid == -1, error('Não foi possível escrever no config.json'); end
    fprintf(fid, '%s', json_str);
    fclose(fid);

    % 3.3. Executa o Algoritmo
    try
        % Pausa breve para garantir I/O de disco
        pause(1);

        % Executa o script
        eval(script_name);

        fprintf('\n>>> Experimento %s concluído com sucesso.\n', current_exp.name);

    catch ME
        fprintf('\n!!! Erro durante o experimento %s:\n%s\n', current_exp.name, ME.message);
        % Não para o loop, tenta o próximo
    end

    % Limpa variáveis pesadas geradas pelo script anterior para poupar memória,
    % MAS mantém as variáveis de controle deste loop (experiments, i, etc)
    clearvars -except experiments i total_exps base_cfg config_file backup_file script_name update_struct

    close all; % Fecha figuras geradas
end

% =========================================================================
% 4. FINALIZAÇÃO
% =========================================================================

% Restaura o config original
copyfile(backup_file, config_file);
delete(backup_file);

fprintf('\n===================================================\n');
fprintf('Bateria de testes finalizada.\n');
fprintf('Configuração original restaurada.\n');
