function [Ts, tfr2] = TMSST(x,hlength,num)
%   Time-reassigned Multisynchrosqueezing Transform
%   Input:
%	x       : Signal.
%	hlength : Window length.
%   num     : Iteration number
%   Output:
%   Ts     : TMSST result.
%	tfr2   : STFT result.
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
%
%  Written by YuGang.
[xrow,xcol] = size(x);
N=xrow;
if (xcol~=1),
 error('X must be column vector');
end;
if (nargin < 2)
    hlength=round(xrow/5);
    num=1;
else if (nargin < 3)
        num=1;
    end
end


t=1:N;tcol=N;
hlength=hlength+1-rem(hlength,2);
ht = linspace(-0.5,0.5,hlength);ht=ht';
% Gaussian window
h = exp(-pi/0.32^2*ht.^2);
th=h.*ht;
[hrow,~]=size(h); Lh=(hrow-1)/2;
tfr1= zeros (N,tcol);
tfr3= zeros (N,tcol);
for icol=1:tcol,
ti= t(icol); tau=-min([round(N/2)-1,Lh,ti-1]):min([round(N/2)-1,Lh,xrow-ti]);
indices= rem(N+tau,N)+1;
rSig = x(ti+tau,1);
tfr1(indices,icol)=rSig.*conj(h(Lh+1+tau));
tfr3(indices,icol)=rSig.*conj(th(Lh+1+tau));
end;
tfr1=fft(tfr1);
tfr3=fft(tfr3);
tfr1=tfr1(1:round(N/2),:);
tfr3=tfr3(1:round(N/2),:);
omega2= zeros(round(N/2),tcol);
omega22= zeros(round(N/2),tcol);
for a=1:round(N/2)
omega2(a,:) = t+(hlength-1)*real(tfr3(a,t)./tfr1(a,t));
end
[neta,nb]=size(tfr1);
if num>1
    for kk=1:num-1
        for b=1:nb
            for eta=1:neta
                k2 = round(omega2(eta,b));
                if k2>=1 && k2<=nb
                    omega22(eta,b)=omega2(eta,k2);
                end
            end
        end
        omega2=omega22;
    end
else
    omega22=omega2;
end
omega22=round(round(omega22*2)/2);
tfr2 = zeros(round(N/2),tcol);
    for eta=1:round(N/2)%frequency
        tfr2(eta,:)=tfr1(eta,:).*exp(-1j * 2 * pi*eta*(t/N));
    end
Ts = zeros(round(N/2),tcol);
% Reassignment step
for b=1:N%time
    for eta=1:round(N/2)%frequency
 %       if abs(tfr1(eta,b))>0.000001
            k2 = omega22(eta,b);
            if k2>=1 && k2<=N
                Ts(eta,k2) = Ts(eta,k2) + (tfr2(eta,b));
            end
  %      end
    end
end
Ts=Ts/(xrow/2);
