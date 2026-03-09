% 检查数据格式的调试脚本
clear; clc;

data_path = '../../data/afdm_n256_test.mat';
fprintf('检查数据文件: %s\n\n', data_path);

% 列出所有数据集
info = h5info(data_path);
fprintf('数据集列表:\n');
for i = 1:length(info.Datasets)
    ds = info.Datasets(i);
    fprintf('  %s: size = [%s]\n', ds.Name, num2str(ds.Dataspace.Size));
end

fprintf('\n读取数据...\n');

H_all = h5read(data_path, '/H_dataset');
x_all = h5read(data_path, '/x_dataset');
snr_all = h5read(data_path, '/snr_dataset');
noise_var_all = h5read(data_path, '/noise_var_dataset');

fprintf('\n读取后的维度:\n');
fprintf('  H_all: [%s]\n', num2str(size(H_all)));
fprintf('  x_all: [%s]\n', num2str(size(x_all)));
fprintf('  snr_all: [%s]\n', num2str(size(snr_all)));
fprintf('  noise_var_all: [%s]\n', num2str(size(noise_var_all)));

% 检查第一个样本
fprintf('\n第一个样本检查:\n');
fprintf('  H(1,1,1) = %s\n', num2str(H_all(1,1,1)));
fprintf('  x(1,1,1) = %s\n', num2str(x_all(1,1,1)));
fprintf('  snr(1) = %s\n', num2str(snr_all(1)));

% 检查参数文件
params_path = '../../data/oampnet_v4_params.mat';
if exist(params_path, 'file')
    params = load(params_path);
    fprintf('\nOAMPNet参数:\n');
    fprintf('  gamma: [%s]\n', num2str(size(params.gamma)));
    fprintf('  damping: [%s]\n', num2str(size(params.damping)));
    fprintf('  temperature: [%s]\n', num2str(size(params.temperature)));
    fprintf('  num_layers: %d\n', params.num_layers);
else
    fprintf('\n警告: 未找到参数文件\n');
end

fprintf('\n检查完成!\n');
