clc; clear; close all
x = xlsread('F:\数据处理\A框架结构\数据.xlsx');

%% 2. 选择方法
n_point = 4000; % 以1024 个点 划分样本
flag =0; % 坐标图窗控制参数，flag=0 时显示坐标信息，=1 不显示坐标信息
method = 'GADF'; % 字符串格式
% method可选：
% 时频类：
% method = 'mel'，       运行→ 梅尔频谱图Mel spectrogram
% method = 'stft'，        运行→ 短时傅里叶变换short-time Fourier transform
% method = 'ST'，         运行→ s变换S-transform
% method = 'WVD'，     运行→ 魏格纳分布Wigner-Ville Distribution
% method = 'DWVD'，   运行→ 离散魏格纳分布Discrete Wigner-Ville Distribution
% method = 'HHT'，      运行→ 希尔伯特变换Hilbert-Huang Transform
% method = 'cwt'，        运行→ 连续小波变换Continuous wavelet transform
% method = 'RWT'，      运行→ 实小波变换Real wavelet transform
% method = 'SST'，        运行→ 同步压缩变换Synchrosqueezing transform
% method = 'wsst'，       运行→ 小波同步压缩变换wavelet synchrosqueezed transform
% method = 'wsst2'，     运行→ 小波二阶同步压缩变换wavelet second order synchrosqueezed transform
% method = 'VSST2'，    运行→ 垂直二阶同步压缩变换vertical second-order synchrosqueezing 
% method = 'MSST'，     运行→ 多尺度同步压缩变换Multisynchrosqueezing Transform 
% method = 'wmsst'，    运行→ 小波多尺度同步压缩变换Wavelet Multisynchrosqueezed Transform 
% method = 'LMSST'，   运行→ 局部最大同步压缩变换Local maximum synchrosqueezing transform 
% method = 'TMSST'，   运行→ 时间重分配多同步压缩变换Time-reassigned Multisynchrosqueezing Transform 
% method = 'SET'，         运行→ 同步提取变换Synchroextracted  transform
% method = 'wset'，       运行→ 小波同步提取变换Wavelet Synchroextracted Transform
% method = 'TET'，         运行→ 暂态提取变换transient-extracting transform
% method = 'STET'，       运行→ 二阶暂态提取变换Second-order transient-extracting transform 
% 转换类：
% method = 'GASF'，    运行→ 格拉姆角和场Gramian angular summation field
% method = 'GADF'，   运行→ 格拉姆角差场Gramian angular difference field
% method = 'RP'，        运行→ 递归图recurrence plots
% method = 'RPM'，     运行→ 相对位置矩阵Relative Position Matrix


%% 3. 定义文件路径，用于保存图像
%% 这里是自动定义的文件路径，如不需要可将此部分注释后，手动添加路径，类似 file_path='D:\code\...'
%% file_path 一定要给文件路径
file_path =[method '_images']; % 
% 判断文件路径是否存在，如果文件路径不存在，则创建
if ~exist(file_path, 'dir')
    mkdir(file_path);
    fprintf('文件路径 %s 创建成功！\n', file_path); 
end
%%  4.生成图像
addpath(genpath('Algorithms')) % 将算法加入路径
get_images(x,n_point,method,file_path,flag);
rmpath(genpath('Algorithms')) % 使用完后，将算法移除路径