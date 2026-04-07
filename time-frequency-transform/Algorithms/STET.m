function [Te2] = STET(x,hlength)
% Computes the second-order transient-extracting transform (STET) of the signal x.
% INPUT
%    x      :  Signal needed to be column vector.
%    hlength:  The length of window function.
% OUTPUT
%    Te2     :  The STET result
[xrow,xcol] = size(x);
if (xcol~=1),
    error('X must be column vector');
end;
N=xrow;
% hlength: the length of window function.
hlength=hlength+1-rem(hlength,2);
ht = linspace(-0.5,0.5,hlength);ht=ht';
% Gaussian window
h = exp(-0.5*ht.^2/0.02);
% derivative of window
dh = -ht .* h/0.02; % g'
[hrow,~]=size(h); Lh=(hrow-1)/2;
tfr1= zeros (round(N/2),N) ;% G(t,w);
tfr2= zeros (round(N/2),N) ;% Gg'(t,w);
tfr3= zeros (round(N/2),N) ;% Gsu(t,w);
t=1:N;
su=x.*t';% s*u;
for icol=1:N,
    ti= t(icol); tau=-min([round(N/2)-1,Lh,ti-1]):min([round(N/2)-1,Lh,xrow-ti]);
    indices= rem(N+tau,N)+1;
    rSig = x(ti+tau,1);
    suSig = su(ti+tau,1);
    tfr1(indices,icol)=rSig.*h(Lh+1+tau);% G(t,w)*exp(iwt);
    tfr2(indices,icol)=rSig.*dh(Lh+1+tau);% Gg'(t,w)*exp(iwt);
    tfr3(indices,icol)=suSig.*h(Lh+1+tau);% Gsu(t,w)*exp(iwt);
end;
tfr1=fft(tfr1);tfr1=tfr1(1:round(N/2),:);
tfr2=fft(tfr2);tfr2=tfr2(1:round(N/2),:);
tfr3=fft(tfr3);tfr3=tfr3(1:round(N/2),:);
for eta=1:round(N/2)
    tfr1(eta,:)=tfr1(eta,:).*exp(-1j * 2 * pi*(eta-1)*((t)/N));% G(t,w);
    tfr2(eta,:)=tfr2(eta,:).*exp(-1j * 2 * pi*(eta-1)*((t)/N));% Gg'(t,w);
    tfr3(eta,:)=tfr3(eta,:).*exp(-1j * 2 * pi*(eta-1)*((t)/N));% Gsu(t,w);
end
omega_t=real(tfr3./tfr1);
omega_w=real(-tfr2./tfr1);
omega_t_dt = zeros (round(N/2),N-1);
omega_w_dt = zeros (round(N/2),N-1);
omega_t_dw = zeros (round(N/2)-1,N);
omega_w_dw = zeros (round(N/2)-1,N);
for i=1:round(N/2)
    omega_t_dt(i,:)=diff(omega_t(i,:));
    omega_w_dt(i,:)=diff(omega_w(i,:));
end
omega_t_dt(:,end+1)=omega_t_dt(:,end);
omega_w_dt(:,end+1)=omega_w_dt(:,end);
for i=1:N
    omega_t_dw(:,i)=diff(omega_t(:,i));
    omega_w_dw(:,i)=diff(omega_w(:,i));
end
omega_t_dw(end+1,:)=omega_t_dw(end,:);
omega_w_dw(end+1,:)=omega_w_dw(end,:);
omega_t2=zeros(round(N/2),N);
tfr=zeros(round(N/2),N);
tfr1=tfr1/(xrow/2);
for i=1:round(N/2)%frequency variable
    for j=1:N     %time variable
        if abs(omega_w_dt(i,j).*omega_t_dw(i,j))>1e-10;
            omega_t2(i,j)=-omega_w_dw(i,j)/(omega_w_dt(i,j)*omega_t_dw(i,j))*(omega_t(i,j)-t(j)*omega_t_dt(i,j));
            tfr(i,j)=tfr1(i,j)/(sqrt(-omega_w_dt(i,j)*omega_t_dw(i,j)/omega_w_dw(i,j)-1i*omega_w_dw(i,j)));
        else
            omega_t2(i,j)=omega_t(i,j);
            tfr(i,j)=tfr1(i,j);
        end
    end
end
Te2=zeros(round(N/2),N);
omega_t2=round(omega_t2);
for i=1:round(N/2)
    for j=1:N
        if abs(omega_t2(i,j)-j)==0
            Te2(i,j)=tfr(i,j);
        end
    end
end
end