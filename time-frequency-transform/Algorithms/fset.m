function [set,f,t] = fset(x,varargin)
%FSET Fourier synchroextracted transform
%   SET = FSET(X) returns the Fourier synchroextracted transform of X. Each
%   column of SET contains the synchroextracted spectrum of a windowed
%   segment of X. The number of columns of SET is equal to the number of
%   samples of the input X, which is padded with zeros on each side. Each
%   spectrum is one-sided for real X and two-sided and centered for complex
%   X. By default, a Kaiser window of length 256 and a beta of 10 is used.
%   If X has fewer than 256 samples, then the Kaiser window has the same
%   length as X.
% 
%   [SET,W,S] = FSET(X) returns a vector of normalized frequencies, W,
%   corresponding to the rows of SET and a vector of sample numbers, S,
%   corresponding to the columns of SET. Each element of S is the sample
%   number of the midpoint of a windowed segment of X.
%
%   [SET,F,T] = FSET(X,Fs) specifies the sampling frequency, Fs, in hertz,
%   as a positive scalar. The output contains sample times, T, expressed in
%   seconds and frequencies, F, expressed in hertz.
%
%   [SET,F,T] = FSET(X,Ts) specifies the sampling interval, Ts, as a 
%   <a href="matlab:help duration">duration</a>. Ts is the time between samples of X. T has the same units 
%   as the duration. The units of F are in cycles/unit time of the
%   duration.
%
%   [...] = FSET(X,...,WINDOW) when WINDOW is a vector, divides X into
%   segments of the same length as WINDOW and then multiplies each segment
%   by WINDOW.  If WINDOW is an integer, a Kaiser window with length WINDOW
%   and a beta of 10 is used. If WINDOW is not specified, the default
%   Kaiser window is used.
%
%   FSET(...) with no output arguments plots the synchroextracted transform
%   of the input vector x on the current figure.
%
%   FSET(...,FREQLOCATION) controls where MATLAB displays the frequency
%   axis on the plot. This string can be either 'xaxis' or 'yaxis'.
%   Setting FREQLOCATION to 'yaxis' displays frequency on the y-axis and
%   time on the x-axis.  The default is 'xaxis', which displays frequency
%   on the x-axis. FREQLOCATION is ignored when output arguments are
%   specified.
%
%   % Example 1: 
%   %   Compute and plot the Fourier synchroextracted transform (SET) of a
%   % signal that consists of two chirps. 
%   fs = 3000;
%   t = 0:1/fs:1-1/fs;   
%   x1 = 2*chirp(t,500,t(end),1000);
%   x2 = chirp(t,400,t(end),800); 
%   fset(x1+x2,fs,'yaxis')
%   title('Magnitude of Fourier Synchroextracted Transform of Two Chirps')
%
%   % Compute and plot the short-time Fourier transform.
%   [stft,f,t] = spectrogram(x1+x2,kaiser(256,10),255,256,fs);
%   figure
%   h = imagesc(t,f,abs(stft));
%   xlabel('Time (s)') 
%   ylabel('Frequency (Hz)')
%   title('Magnitude of Short-time Fourier Transform of Two Chirps')
%   h.Parent.YDir = 'normal';
%
%   % Example 2
%   %   Compute the Fourier synchroextracted transform of a speech signal.
%   load mtlb
%   fset(mtlb,Fs,hann(256),'yaxis')
%
%   See also ifsst, tfridge, spectrogram, duration.
% Copyright 2015-2018 The MathWorks, Inc.
narginchk(1,4);
nargoutchk(0,3);
if nargin > 1
    [varargin{:}] = convertStringsToChars(varargin{:});
end
% Parse inputs. Fs is populated as used throughout the bulk of the code
% even if Ts is specified. fNorm specifies normalized frequencies.
[Fs,Ts,win,fNorm,freqloc] = parseInputs(x,varargin{:});
validateInputs(x,Fs,Ts,win);
% Parameters based on window size - noverlap is fixed so the transform is
% invertible.
nwin = length(win);
nfft = nwin;
noverlap = nwin-1;
% Convert to column vectors
x = signal.internal.toColIfVect(x);
win = win(:);
% Compute the output time vector (one time per sample point of the input)
if fNorm
  tout = (1:length(x));
else
  tout = (0:length(x)-1)/Fs;
end
% cast to enforce precision rules (we already checked that the inputs are
% numeric.
% cast to enforce precision rules
if (isa(x,'single') || isa(win,'single'))
  x = single(x);
  win = single(win);
  Fs = single(Fs);
  tout = single(tout);
else
  Fs = double(Fs);
end
  
% Pad the signal vector x
if mod(nwin,2)
  xp = [zeros((nwin-1)/2,1) ; x ; zeros((nwin-1)/2,1)];
else
  xp = [zeros((nwin)/2,1) ; x ; zeros((nwin-2)/2,1)];
end
nxp = length(xp);
% Place xp into columns for the STFT
xin = getSTFTColumns(xp,nxp,nwin,noverlap,Fs);
% Compute the STFT
[sstout,fout] = computeDFT(bsxfun(@times,win,xin),nfft,Fs); 
stftc = computeDFT(bsxfun(@times,dtwin(win,Fs),xin),nfft,Fs);
clear xin;
% Compute the reassignment vector
fcorr = -imag(stftc./ sstout);
fcorr(~isfinite(fcorr)) = 0;
fcorr = bsxfun(@plus,fout,fcorr);
tcorr = bsxfun(@plus,tout,zeros(size(fcorr)));
clear stftc;
% Mulitply STFT by a linear phase shift to produce the modified STFT
m = floor(nwin/2);
inds = 0:nfft-1;
ez = exp(-1i*2*pi*m*inds/nfft)';
sstout = bsxfun(@times,sstout,ez); 
% Synchroextracting transform
if nargin >1
    if isduration(varargin{1})
        if isempty(varargin{1})
            error(message('signal:fsst:EmptyDuration'));
        else
            Ts = varargin{1};
            Fs = 1/seconds(Ts);
        end
    elseif ischar(varargin{1})
        freqloc = validatestring(varargin{1},{'xaxis','yaxis'});
        Fs=length(x);
        fcorr=fcorr*Fs/2/pi;
    end
else
    Fs=length(x);
    fcorr=fcorr*Fs/2/pi;
end
f=0:(Fs/2)/(nwin/2):round(Fs/2)-(Fs/2)/(nwin/2);
fcorr=fcorr(1:round(nwin/2),:);
[neta,nb]=size(fcorr);
set=zeros(size(fcorr));
for b=1:nb
    for eta=1:neta
        if abs(fcorr(eta,b)-f(eta))<0.5*(Fs/nwin)
        set(eta,b)=sstout(eta,b);
        end
    end
end
% Reduce to one-sided spectra if the input is real, otherwise return a
% two-sided (centered) spectra.
if isreal(x)
  fout = psdfreqvec('npts',nfft,'Fs',Fs,'Range','half');
  sstout = sstout(1:length(fout),:);
else
  % Centered spectra
  sstout = centerest(sstout);
  fout = centerfreq(fout);
end
% Scale fout and tout if the input is a duration object
if ~isempty(Ts)
  [~,units,timeScale] = getDurationandUnits(Ts);
  tout = tout*timeScale;
  fout = fout/timeScale;
else
    units = [];
end
if nargout == 0
    if ~isempty(units)
        switch units
            case 'sec'
                plotsst(seconds(tout),fout,set,fNorm,freqloc);
            case 'min'
                plotsst(minutes(tout),fout,set,fNorm,freqloc);
            case 'hr'
                plotsst(hours(tout),fout,set,fNorm,freqloc);
            case 'day'
                plotsst(days(tout),fout,set,fNorm,freqloc);
            case 'year'
                plotsst(years(tout),fout,set,fNorm,freqloc);
        end
    else
        plotsst(tout,fout,set,fNorm,freqloc);
    end
else
  sst = sstout;
  f = fout;
  t = tout(:)';
end
%--------------------------------------------------------------------------
function [Fs,Ts,win,fNorm,freqloc] = parseInputs(x,varargin)
% Set defaults
Fs = [];
Ts = [];
win = kaiser(min(256,length(x)),10); %#ok<*NASGU>
fNorm = false;
freqloc = '';
% Parse optional inputs
if nargin > 1 
  if isduration(varargin{1})
    if isempty(varargin{1})
      % Throw error is empty duration object is supplied
      error(message('signal:fsst:EmptyDuration'));
    else
      Ts = varargin{1};
      Fs = 1/seconds(Ts);
    end
  elseif ischar(varargin{1})
    freqloc = validatestring(varargin{1},{'xaxis','yaxis'});  
  else
    if ~isempty(varargin{1})
      Fs = varargin{1};
    end
  end
end
if isempty(Fs) && isempty(Ts)
  fNorm = true;
  Fs = 2*pi;
end
if nargin > 2 
  if ischar(varargin{2})
    freqloc = validatestring(varargin{2},{'xaxis','yaxis'});  
  elseif ~isempty(varargin{2})
    win = varargin{2};
    if isscalar(win)
      validateattributes(win,{'numeric'},{'positive'},'fsst','WINDOW');
      win = kaiser(double(win),10);
    end
  end
end
if nargin > 3
  freqloc = validatestring(varargin{3},{'xaxis','yaxis'}); 
end
%--------------------------------------------------------------------------
function validateInputs(x,Fs,Ts,win)
validateattributes(x,{'single','double'},...
  {'nonsparse','finite','nonnan','vector'},'fsst','X');
validateattributes(Fs,{'numeric'},...
    {'real','positive','finite','nonnan','scalar'},'fsst','Fs');
validateattributes(win,{'single','double'},...
  {'real','finite','nonnegative','nonnan','vector'},'fsst','WINDOW');  
  
if ~isempty(Ts)
  dt = getDurationandUnits(Ts);
  validateattributes(dt,{'numeric'},...
    {'real','positive','finite','nonnan','scalar'},'fsst','Ts');
end
% Check X has at least 2 samples
if length(x) < 2
  error(message('signal:fsst:MustBeMinLengthX'));
end
% Check WINDOW has at least 2 samples
if length(win) < 2
  error(message('signal:fsst:MustBeMinLengthWin'));
end
% Check window length is not more than the length of the input signal.
if length(win) > length(x)
  error(message('signal:fsst:WinLength'));
end
%--------------------------------------------------------------------------
function [dt,units,timeScale] = getDurationandUnits(Ts)
% This function returns the sampling interval and a format string
% The Units string is only for plotting.
tsformat = Ts.Format;
% Use first character of format string to determine correct
% duration object method.
tsformat = tsformat(1);
% Using the same time units as engunits. Units in engunits are
% not localized.
% time_units = {'secs','mins','hrs','days','years'};
switch tsformat
    case 's'
        dt = seconds(Ts);
        units = 'sec';
        timeScale = 1;
    case 'm'
        dt = minutes(Ts);
        units = 'min';
        timeScale = 1/seconds(minutes(1));
    case 'h'
        dt = hours(Ts);
        units = 'hr';
        timeScale = 1/seconds(hours(1));
    case 'd'
        dt = days(Ts);
        units = 'day';
        timeScale = 1/seconds(days(1));
    case 'y'
        dt = years(Ts);
        units = 'year';
        timeScale = 1/seconds(years(1));
end
%--------------------------------------------------------------------------
function plotsst(t,f,set,fNorm,freqloc)
  % Plot the FSET in the current figure
  if fNorm
      %Convert to time as expected by plotTFR
      t = t ./ (2*pi);
  end
  
  if isempty(freqloc)
    plotOpts.freqlocation = 'xaxis';
  else
    plotOpts.freqlocation = freqloc;
  end
  
  plotOpts.title = getString(message('signal:fsst:titleFSST'));
  plotOpts.cblbl = getString(message('signal:fsst:ColorbarLabel'));
  plotOpts.cursorclbl = [getString(message('signal:fsst:ColorbarLabel')) ': '];
  plotOpts.isFsnormalized = fNorm;
  signalwavelet.internal.convenienceplot.plotTFR(t,f,abs(set),plotOpts);
 
