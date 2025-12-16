function simulated_hoa_rir_generator(rir_dir)
%SIMULATED_HOA_RIR_GENERATOR Simulate RIRs and encode them in HOA format
%   simulated_hoa_rir_generator() simulates RIRs using SMIR-generator, encodes
%   them in HOA format and saves them in ./rirs/hoa.
% Example:
%   simulated_hoa_rir_generator("./rirs/rir")
% Requires:
%   - SMIR-Generator: addpath('src/SMIR-Generator/');
%   - SHTools: addpath('src/audioprocessing/scripts/library/');
%
% Author: Julio J. Carabias-Orti
% Date: June, 2025

% Load parameters
P = readyaml('config.yaml');

% Parameters
c=P.general.c;
fs=P.general.fs;
sh_order = P.hoa_encoding.sh_order;
b_n_len = P.hoa_encoding.b_n_len;
% Eigenmike parameters
isRigid = strcmp(P.eigenmike.micType,'rigid');
rmic = P.eigenmike.mic_r;
mic_az = P.eigenmike.mic_az;
mic_el = P.eigenmike.mic_el;

% SH coefficients
ncoeff = (sh_order+1)^2;

% Compute FIR filters for radial function
[~, b_n_inverse_in_time] = compute_bn_fir_filters(sh_order, fs, b_n_len, c, rmic, isRigid);

% Process SRIRs
rir_list = dir(fullfile(rir_dir,"room*.mat"));
for i=1:length(rir_list)
    [~, rir_name, ~] = fileparts(rir_list(i).name);
    disp(rir_name);

    rir_data = matfile(fullfile(rir_dir,sprintf("%s.mat",rir_name)));
    var_names = who(rir_data);
    if any(strcmp(var_names, 'sim_rir'))
        rir = double(rir_data.sim_rir);
    elseif any(strcmp(var_names, 'sim_rir_clean'))
        rir = double(rir_data.sim_rir_clean);
    else
        error('No valid RIR variable found in the .mat file.');
    end
    nsamples = size(rir,2);


    %% B-Format Encoding
    Y = SHTools.getRealSH(sh_order, mic_el, mic_az);
    S = pinv(Y');
    rir_SH = tensorprod(S,rir,2,1);

    rir_hoa = zeros(ncoeff,nsamples+b_n_len); % more length filled with zeros at the end for the delay in the filters
    for shc=1:ncoeff
        rir_hoa(shc,1:nsamples) = rir_SH(shc,:);
        rir_hoa(shc,:) = filter(b_n_inverse_in_time(shc,:),1,rir_hoa(shc,:));
    end

    save(fullfile(rir_dir,['hoa_', rir_name]),'rir_hoa');
end

end