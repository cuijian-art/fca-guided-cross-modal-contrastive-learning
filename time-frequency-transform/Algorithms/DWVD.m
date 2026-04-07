function [K,DVW,t,f] = DWVD(x,fs)
   
% Discrete Wigner-Ville Distribution
% x = Input Signal
% fs = sampling rate
% K = Kernel values
% DVW = DWVT
%This program is free software; you can redistribute it and/or modify

if (rem(length(x),2) == 0)
%     error("Window Size must be in odd number")
    x=[x;0];
end
N = length(x)-1;
X=fft(x);
X=[X(1:N/2+1);zeros(N,1);X(N/2+2:N+1)];
x=2*ifft(X);
x1 = x;
x=[zeros(N,1);x;zeros(N,1)];
z = hilbert(x);
X=zeros(2*N+1);
for k=1:2*N+1  
    X(:,k)=z(k+(0:2*N)); 
end
 K = X.*conj(flipud(X));
 DVW=real(fft(K([N+1:2*N+1,1:N],:))); 
 t = (0:2*N)/fs;
 f = (0:N)'/(N+1) * fs;
f = linspace(0,f(end),2*N);
% figure
% subplot(2,2,[3,4]);
% imagesc(t,f/1e6,DVW);
% set(gca,'YDir','normal')
% xlabel('Time (s)');
% ylabel('Freqeuncy (MHz)');
% l = caxis;
% caxis([0 l(2)]);
% subplot(2,2,1);
% plot(t,real(x1));
% xlabel('Time (s)');
% ylabel('Amp');
% subplot(2,2,2);
% plot(t,max(DVW))
% xlabel('Time (S)');
% ylabel('P|x|');
end