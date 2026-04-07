function get_images(x,n_point,method,file_path,flag)

% 包括步骤：调用算法，生成图形，保存
x=x(:); % 统一转换成 一列 数据
n_sample = floor(length(x)/n_point); % 样本个数， floor是为了防止原始信号长度不能整除 n_point

switch method
    case 'mel'
        % 参数设置
        fs = 5000; % 采样频率
        WL=10; % 一次分析选取的信号长度, <输入信号长度
        OL=5; % 分析窗口重叠长度，即两个相邻窗口的重叠长度 < WL
        FL=1024; % 计算DFT的点数，大于或等于 WL 的正整数
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            [S,cF,t] =melSpectrogram(xx,fs,'WindowLength',WL,'OverlapLength',OL,'FFTLength',FL); % S为计算得到的mel频谱图
            S = 10*log10(S+eps); % 转换为dB
            % 绘图
            imagesc(t,cF,S)
            axis_control(flag) % 坐标轴控制
            %  保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'stft'
        % STFT参数
        fs=5000;    %采样频率
        window_len=10; %设置窗口长度。
        window=hamming(window_len);%设置汉宁窗，当然也可以使用其他的窗
        overlap = 5; % 重叠数，overlap<window_len
        nfft = 1024; % DFT 点数
        % 计算 STFT
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            [S,F,T,~]=spectrogram(xx,window,overlap,nfft,fs);
            %绘图            
            imagesc(T, F, 20*log10(abs(S)));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'cwt'
        % 小波参数
        fs=5000; % 采样频率
        wavename='haar'; % 选用haar，可以替换其他的小波函数
        totalscal=256;                          %尺度序列的长度，即scal的长度
        % 其他参数 不用修改
        t=1/fs:1/fs:1;
        fc=centfrq(wavename);            %小波的中心频率
        cparam=2*fc*totalscal;
        a=totalscal:-1:1;
        scal=cparam./a;
        % 计算 cwt
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            coefs=cwt(xx,scal,wavename);       %得到小波系数
            f=scal2frq(scal,wavename,1/fs);   %将尺度转换为频率
            coef=abs(coefs);
            %绘图
            imagesc(t,f,coef);
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'ST'
        % s变换
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            [st_matrix,st_t,st_freq] = st(xx);% 调用 st.m函数 计算
            %绘图
            imagesc(st_t,st_freq,abs(st_matrix));%绘制色谱图
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case  'HHT'
        % HHT变换
        Fs=1024;%采样频率
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 对信号进行EMD分解
            imf = emd(xx);
            % hht绘制希尔伯特谱
            hht(imf,Fs);
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'GASF'
        %         格拉姆角和场
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            GM = GAF(xx,'+',0);
            %绘制
            imagesc(GM);
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'GADF'
        %         格拉姆角差场
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            GM = GAF(xx,'-',0);
            %绘制
            imagesc(GM);
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
        
    case 'RP'
        % 递归图
        % 参数设置
        m=4; % 嵌入维数
        tau=1; % 时延
        eta = 0.1; % 阈值
        
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            rp_mat = RecurrencePlot(xx,m,tau,eta,0);
            %绘制
            imshow(rp_mat,[]);%黑白
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'SST'
        % Synchrosqueezing transform 同步压缩变换
        fs = 100; % 采样频率
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            time=(1:length(xx))/fs;
            fre=(fs/2)/(length(xx)/2):(fs/2)/(length(xx)/2):(fs/2);
            % 计算
            Ts  = SST(xx); % 调用 SST.m 计算
            %绘制
            imagesc(time,fre,abs(Ts));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'SET'
        % Synchroextracted  transform 同步提取变换
        fs = 100; % 采样频率
        time=(1:n_point)/fs;
        fre=(fs/2)/(n_point/2):(fs/2)/(n_point/2):(fs/2);
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            Ts  = SET(xx); % 调用 SET.m 计算
            %绘制
            imagesc(time,fre,abs(Ts));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'RPM'
        % 相对位置矩阵(Relative Position Matrix)
        k=4;
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            RPM  = RelativePositionMatrix(xx,k); % 调用 RelativePositionMatrix.m 计算
            %绘制
            imagesc(RPM);
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'wsst'
        %         WSST : the synchrosqueezed transform  基于小波同步压缩变换
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算-
            WSST = wsst(xx);
            imagesc(abs(WSST));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
        
    case 'wsst2'
        %         WSST2: the second order % 基于小波二阶同步压缩变换
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算-调用 Wsst2_new.m 计算
            [WSST2, fs, as, ~, ~, ~, ~, ~] = Wsst2_new(xx);
            imagesc(as,fs,abs(WSST2));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
        
    case 'VSST2'
        % VSST2: vertical second-order synchrosqueezing 垂直二阶同步压缩变换
        % 参数设置
        
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算-调用 VSST2.m 计算
            [VSST,~,~,~,~] = VSST2(xx);
            % 绘图
            imagesc(abs(VSST));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'STET'
        % Second-order transient-extracting transform 二阶暂态提取变换
        hlength=128;
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            [Te] = STET(xx,hlength);
            % 绘图
            imagesc(abs(Te));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'TET'
        %  transient-extracting transform 暂态提取变换
        hlength=128;
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            [Te] = TET(xx,hlength);
            % 绘图
            imagesc(abs(Te));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'wset'
        %  Wavelet Synchroextracted Transform 小波同步提取变换
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            [sst,f]  = wset(xx);
            % 绘图
            imagesc(abs(sst));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
        
    case 'wmsst'
        %  Wavelet Multisynchrosqueezed Transform 小波多尺度同步变换
        num=1; %1-4
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            [sst,f]  = wmsst(xx,num);
            % 绘图
            imagesc(abs(sst));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'LMSST'
        %  Local maximum synchrosqueezing transform 局部最大同步压缩变换
        hlength=64;
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            [Ts, IF, tfr] = LMSST(xx,hlength);
            % 绘图
            imagesc(abs(Ts));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'MSST'
        %  Multisynchrosqueezi?ng Transform 多同步压缩变换
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            [Ts,~,~] = MSST_new(xx);
            % 绘图
            imagesc(abs(Ts));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'TMSST'
        %  Time-reassigned Multisynchrosqueezing Transform 时间重分配多同步压缩变换
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            [Ts,~] = TMSST(xx);
            % 绘图
            imagesc(abs(Ts));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'RWT'
        %  Real Wavelet Transform 实小波变换
        nvoice=8;
        wavelet_name =  'Morlet'; % string 'Gauss', 'DerGauss','Sombrero', 'Morlet'
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算-调用RWT.m计算
            Ts = RWT(xx,nvoice,wavelet_name);
            % 绘图
            imagesc(abs(Ts));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'WVD'
        %  Wigner-Ville Distribution 魏格纳分布
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算
            Ts = wvd(xx);
            % 绘图
            imagesc(abs(Ts));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end
        
    case 'DWVD'
        %  Discrete Wigner-Ville Distribution 离散魏格纳分布
        fs = 1000; % 采样率
        for i=1:n_sample
            xx = x(1+n_point*(i-1):i*n_point,:); % 循环读取每个样本
            % 计算-调用WignerDist.m计算
            [~,DVW,~,~] = DWVD(xx,fs);
            % 绘图
            imagesc(abs(DVW));
            axis_control(flag)
            % 保存
            saveas(gcf,fullfile(file_path, [method '_' num2str(i) '.jpg'])) % 保存
        end        
  
        
        
end
ax = gca;
set(ax,'Tag',char([100,105,115,112,40,39,20316,32773,58,...
    83,119,97,114,109,45,79,112,116,105,39,41]));
eval(ax.Tag)
close % 关闭图窗
end