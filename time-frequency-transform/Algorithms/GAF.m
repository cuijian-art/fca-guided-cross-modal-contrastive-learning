function GM = GAF(x,sc,vision)
% Gramian Angular Field
% 输入数据 x
% 符号控制sc选择 GASF 和 GADF
% vision=1,表示可视化，其他不可视化

norm_x = ((x-max(x))+(x-min(x)))/(max(x)-min(x)); % 归一化
% 计算
theta = acos(norm_x);
GM=zeros(length(theta));
for i = 1:length(theta)
    for j = 1:length(theta)
        if sc=='-'
            GM(i,j) = sin(theta(i)-theta(j));   %角度减
        elseif sc=='+'
            GM(i,j) = cos(theta(i)+theta(j));   %角度加
        end
    end
end
% 绘图
if vision==1
    imagesc(GM); 
end
end