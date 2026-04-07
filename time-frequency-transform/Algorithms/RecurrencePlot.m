function rp_mat = RecurrencePlot(x,m,tau,eta,vision)
% RecurrencePlot
% 输入数据 x
% m: 嵌入维数
% tau：时延
% eta：阈值
% vision=1 可视化
N = length(x);
T=N-(m-1)*tau;
Y=zeros(T,m);
% 相空间
for j=1:T
    Y(j,:)=x(j:tau:j+(m-1)*tau);
end
% 
rp_mat = ones(size(Y,1),size(Y,1));  % 初始化
for i=1:size(Y,1)
    for j=i+1:size(Y,1)
        d(i,j) = norm(Y(i,:)-Y(j,:));

        if d(i,j) > eta
            rp_mat(i,j)=0;
            rp_mat(j,i)=0;
        end
    end
    
end
% 绘图
if vision==1
    imshow(rp_mat,[]);%黑白 
end
end