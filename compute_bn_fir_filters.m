function [b_n_in_time, b_n_inverse_in_time] = ...
    compute_bn_fir_filters(sh_order, fs, b_n_len, c, rmic, isRigid)
%COMPUTE_BN_FIR_FILTERS Compute SH b_n radial filters and their inverse FIRs (linear phase).
%
% Inputs
%   sh_order   : spherical harmonic order (N)
%   fs         : sampling rate [Hz]
%   b_n_len    : FIR length (samples)
%   c          : speed of sound [m/s]
%   rmic       : array radius [m]
%   isRigid    : boolean (rigid sphere model if true)
%
% Outputs
%   b_n_in_time         : (ncoeff x b_n_len) FIR filters (time domain)
%   b_n_inverse_in_time : (ncoeff x b_n_len) inverse FIR filters (time domain)
%   b_n                 : (ncoeff x (b_n_len+1)) thresholded b_n (0..Nyquist)
%   b_n_originals       : (ncoeff x (b_n_len+1)) original b_n (0..Nyquist)
%   freqs_stft_filters  : (1 x (b_n_len+1)) frequency grid [Hz]
% 
% Author: Pedro Vera Candeas
% Date: December 2025

    % Filtering radial function
    thresholds_order = ones(1,sh_order+1)*0.01;
    thresholds_order(2) = 0.2;
    thresholds = zeros((sh_order+1)^2, 1);
    thresholds(1) = thresholds_order(1);
    ind = 2;
    for n = 1:sh_order
        numModes = 2*n;
        thresholds(ind:ind+numModes) = thresholds_order(n+1);
        ind = ind + numModes + 1;
    end
    
    ncoeff = (sh_order+1)^2;
    % ---- frequency grid ----
    freqs_stft_filters = fs/2*linspace(0,1,b_n_len+1);
    nf = numel(freqs_stft_filters);

    % ---- compute b_n originals (0..Nyquist) ----
    b_n = zeros(ncoeff,nf);
    b_n_originals = zeros(ncoeff,nf);
    derivative = zeros(ncoeff,1);
    for f=1:nf
        f_tmp = freqs_stft_filters(f);
        k_tmp = 2*pi*f_tmp/c;
        b_n_originals(:,f) = SHTools.sph_bn(SHTools.getACNOrderArray(sh_order), k_tmp*rmic, isRigid);
        b_n(:,f) = b_n_originals(:,f);
        if f==1
            b_n(:,f) = max(abs(b_n(:,f)), thresholds) .* exp(1i*angle(b_n(:,f)));
        else
            % Only is limited when the function is increasing in frequency
            derivative(:) = 0;
            derivative = (abs(b_n_originals(:,f)) - abs(b_n_originals(:,f-1)))>=0;
            b_n(derivative,f) = max(abs(b_n(derivative,f)), thresholds(derivative)) .* exp(1i*angle(b_n(derivative,f)));
        end
    end
    b_n(:,end) = abs(b_n(:,end)); % The last bin is pi rad and must be real
    b_n_temp = zeros(size(b_n));
    b_n_in_time = zeros(ncoeff,b_n_len);
    b_n_inverse_in_time = zeros(ncoeff,b_n_len);
    
    for shc=1:ncoeff % FIR filters with linear phase (delay b_n_len/2 samples)
        b_n_temp(shc,:) = abs(b_n(shc,:)).*exp(-1i*((0:2^(nextpow2(b_n_len+1)-1))*2*pi/2^nextpow2(b_n_len+1))*b_n_len/2);
        b_n_in_time_temp = real(ifft([b_n_temp(shc,:),conj(b_n_temp(shc,end-1:-1:2))],2^nextpow2(b_n_len+1)));
        b_n_in_time(shc,:) = b_n_in_time_temp(1:b_n_len);
        
        b_n_in_freq_temp = fft(b_n_in_time(shc,:),2^nextpow2(b_n_len+1));
        b_n_temp(shc,:) = b_n_in_freq_temp(1:b_n_len+1);
        b_n_temp(shc,:) = abs(1./b_n_temp(shc,:)).*exp(-1i*((0:2^(nextpow2(b_n_len+1)-1))*2*pi/2^nextpow2(b_n_len+1))*b_n_len/2);
        b_n_in_time_temp = real(ifft([b_n_temp(shc,:),conj(b_n_temp(shc,end-1:-1:2))],2^nextpow2(b_n_len+1)));
        b_n_inverse_in_time(shc,:) = b_n_in_time_temp(1:b_n_len);
    end

end