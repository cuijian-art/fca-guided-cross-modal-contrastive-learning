function [VSST,omega,tau,omega2,g] = VSST2(s,aa,Nfft,gamma)

 %% vertical second-order synchrosqueezing 垂直二阶同步压缩变换
 %
 % INPUTS:   
 %   s: real or complex signal
 %   aa: the parameter a in the amgauss function of the Gaussian window
 %   Nfft: number of frequency bins
 %   gamma: threshold on the STFT for reassignment
 % OUTPUTS:   
 %   VSST : vertical second-order synchrosqueezing 
 % REFERENCES
 % [1] Pham, D-H., Meignen, S. Second-order Synchrosqueezing Transform: The
 % Wavelet Case and Comparisons

% nargin/out
if nargin<4
        aa = 64; %
        gamma = 0.001;
        Nfft =1024;
end
 
 s = s(:);
           
 ft   = 1:Nfft;
 bt   = 1:length(s);
 nb   = length(bt);
 neta = length(ft);
        
 prec = 10^(-6) ;
 L = sqrt(Nfft/aa);
 l = floor(sqrt(-Nfft*log(prec)/(aa*pi)))+1;
 g = amgauss(2*l+1,l+1,L);       
  %figure(); plot(g)
 % Window definition
 n   = (0:2*l)'-l;
 t0  = n/Nfft;
 t0  = t0(:);
 a   = aa*Nfft*pi; 
 gp  = -2*a*t0.*g; 
 gpp = (-2*a+4*a^2*t0.^2).*g; % g''
  
 % Initialization
 STFT  = zeros(neta,nb);
 SST   = zeros(neta,nb);
 VSST  = zeros(neta,nb);
 
 omega  = zeros(neta,nb);
 tau    = zeros(neta,nb);
 omega2 = zeros(neta,nb);
 phipp  = zeros(neta,nb);
             
 %% Computes STFT and reassignment operators
        
 for b=1:nb
 	% STFT, window g  
 	time_inst = -min([l,bt(b)-1]):min([l,nb-bt(b)]); 
    tmp(1:length(time_inst)) = s(bt(b)+time_inst).*g(l+1+time_inst);
    A = fft(tmp(:),Nfft);
    STFT(:,b) = A.*exp(-2/Nfft*pi*1i*(0:Nfft-1)'*time_inst(1))/Nfft*g(l+1);%renormalized so that it fits with recmodes
 	vg  = A;
     
 	% STFT, window xg
    tmp(1:length(time_inst)) = s(bt(b)+time_inst).*(time_inst)'/Nfft.*g(l+1+time_inst);
    vxg = fft(tmp(:),Nfft);
  
    % operator Lx (dtau)
	tau(:,b)  = vxg./vg;
 	
    % STFT, window gp
    tmp(1:length(time_inst))= s(bt(b)+time_inst).*gp(l+1+time_inst);
    vgp = fft(tmp(:),Nfft);
 
            
    omega(:,b) = (ft-1)'- real(vgp/2/1i/pi./vg);
 	
    % STFT, window gpp
 	tmp(1:length(time_inst)) = s(bt(b)+time_inst).*gpp(l+1+time_inst);
    vgpp = fft(tmp(:),Nfft);
 
    %STFT, windox xgp
 	tmp(1:length(time_inst)) = s(bt(b)+time_inst).*(time_inst)'/Nfft.*gp(l+1+time_inst);
 	vxgp = fft(tmp(:),Nfft);
      
 	%computation of the two different omega 
        
    phipp(:,b) = 1/2/1i/pi*(vgpp.*vg-vgp.^2)./(vxg.*vgp-vxgp.*vg);
       
    %new omega2
    omega2(:,b) = omega(:,b) - real(phipp(:,b)).*real(tau(:,b))...
                              + imag(phipp(:,b)).*imag(tau(:,b)); 
    %omega2(:,b) = omega(:,b) - real(phipp(:,b)).*real(tau(:,b)); 

 end

 %% reassignment step
 
 for b=1:nb
   
     for eta=1:neta
        if (abs(STFT(eta,b))> 2*sqrt(2)*gamma*norm(g)/length(s))
            k = 1+round(omega(eta,b));
            if (k >= 1) && (k <= neta)
                % original reassignment
                SST(k,b) = SST(k,b) + STFT(eta,b);
            end
           
            k = 1+round(omega2(eta,b));
            if k>=1 && k<=neta
                % second-order Vertical reassignment: VSST
                VSST(k,b) = VSST(k,b) + STFT(eta,b);
            end 
        end
    end
 end
end


function y = amgauss(N,t0,T);
%AMGAUSS Generate gaussian amplitude modulation.
%	Y=AMGAUSS(N,T0,T) generates a gaussian amplitude modulation 
%	centered on a time T0, and with a spread proportional to T.
%	This modulation is scaled such that Y(T0)=1 
%	and Y(T0+T/2) and Y(T0-T/2) are approximately equal to 0.5 .
% 
%	N  : number of points.
%	T0 : time center		(default : N/2).
%	T  : time spreading		(default : 2*sqrt(N)). 
%	Y  : signal.
%
%	Examples:
%	 z=amgauss(160); plot(z);
%	 z=amgauss(160,90,40); plot(z);
%	 z=amgauss(160,180,50); plot(z);
%
%	See also AMEXPO1S, AMEXPO2S, AMRECT, AMTRIANG.

%	F. Auger, July 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2 of the License, or
%  (at your option) any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software
%  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

if (nargin == 0),
 error ( 'The number of parameters must be at least 1.' );
elseif (nargin == 1),
 t0=N/2; T=2*sqrt(N);
elseif (nargin ==2),
 T=2*sqrt(N);
end;

if (N<=0),
 error('N must be greater or equal to 1.');
else
 tmt0=(1:N)'-t0;
 y = exp(-(tmt0/T).^2 * pi);
end
end