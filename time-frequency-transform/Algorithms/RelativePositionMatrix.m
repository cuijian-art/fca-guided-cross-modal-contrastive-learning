function RPM=RelativePositionMatrix(x,k)

% 计算相对位置矩阵(Relative Position Matrix, RPM)

% 输入：x，一维时间序列

% k，分段聚合近似(PAA)的缩减因子

% 输出：RPM，相对位置矩阵

mu = mean(x);

delta = sqrt(var(x));

z = (x-mu)/delta;

% PAA

N = length(x);

m = ceil(N/k);

if ceil(N/k)-floor(N/k) == 0

    for i = 1:m

        X(i) = 1/k * sum(z(k*(i-1)+1:k*i));

    end

else

    for i = 1:m-1

        X(i) = 1/k * sum(z(k*(i-1)+1:k*i));

    end

    X(m) = 1/(N-k*(m-1)) * sum(z(k*(m-1)+1:N));

end

% 计算两个时间戳之间的相对位置

M = repmat(X,m,1) - repmat(X',1,m);

%相对位移矩阵RPM

RPM = (M - min(M(:))) / (max(M(:))) - min(M(:)) * 255;

% 可视化

if nargout == 0

    imagesc(RPM);

    set(gcf,'Position',[300 200 450 300]); % 自行修改合适大小

    xlim([0,size(RPM,1)]);

    ylim([0,size(RPM,1)]);

    axis square

    colormap(jet); %自行修改colormap

end

end