%% demo_A2B2A_from_rir_mat.m
% Simulate A-Format to B-Format encoding and back to A-Format using SMIR-generated RIRs stored in .mat files.
% Computes the SNR between the original and reconstructed A-Format signals.
% Requires:
%   - SMIR-Generator: addpath('src/SMIR-Generator/');
%   - SHTools: addpath('src/audioprocessing/scripts/library/');
%   - tprod: addpath('src/tprod/');
%   Author: Julio J. Carabias-Orti
%   Date: June, 2025

addpath('src/audioprocessing/scripts/library/');
addpath('src/SMIR-Generator/'); 
addpath('src/tprod/');

% Load parameters
P = readyaml('config.yaml');

%% Paths
audio_dir = P.rutas.audio_dir;
csv_room = dir([P.rutas.csv_dir, filesep, 'room_*.csv']);
csv_room = fullfile(csv_room(1).folder, csv_room(1).name);
csv_rir = dir([P.rutas.csv_dir, filesep, 'rir*.csv']);
csv_rir = fullfile(csv_rir(1).folder, csv_rir(1).name);
srir_dir = P.rutas.srir_dir;
srir_type = P.rutas.srir_type;

%% INPUT PARAMETERS
target_room_id = 1;
target_rir_id = 4;
% --
c = P.general.c;
fs = P.general.fs;
isRigid = strcmp(P.eigenmike.micType,'rigid');
sh_order = P.hoa_encoding.sh_order; %max(ceil((2*pi*(fs/2)/c)*rmic)); %4;
ncoeff = (sh_order+1)^2;
% Eigenmike
rmic = P.eigenmike.mic_r;
mic_az = P.eigenmike.mic_az;
mic_el = P.eigenmike.mic_el;

% Tetrahedral mic
% 6, 8, 10,12, 22, 24,26, 28
% 12, 8, 28, 24
% sampling_mics = [8,12,24,28];
% sampling_mics = [6,8,10,12,22,24,26,28];
sampling_mics = 1:32;
mic_az = mic_az(sampling_mics);
mic_el = mic_el(sampling_mics);

%% Define STFT/ISTFT
winlen = P.general.winlen;
freqs_stft = fs/2*linspace(0,1,winlen+1);
mystft = @(x) stft(x, fs, 'Window', hamming(winlen, 'periodic'), ...
    'FrequencyRange', 'onesided', 'FFTLength', 2^nextpow2(winlen+1));

myistft = @(x) real(istft(x, fs, 'Window', hamming(winlen, 'periodic'), ...
    'FrequencyRange', 'onesided', 'FFTLength', 2^nextpow2(winlen+1)));

%% List audio folders
items = dir(audio_dir);
subfolders = items([items.isdir] & ~ismember({items.name}, {'.', '..'}));
audio_dir = fullfile(audio_dir, {subfolders.name})';

room_info = readtable(csv_room);
rir_info = readtable(csv_rir);
rir_list = dir(fullfile(srir_dir, "*.mat"));

%% Display info about azimuth and elevation
row_idx = find(rir_info.room_id == target_room_id & rir_info.rir_id == target_rir_id);
fprintf('Current azimuth = %.2fº, elevation = %.2fº\n', rad2deg(rir_info.az(row_idx)), rad2deg(rir_info.el(row_idx)));

%% === Load audio ===
i = 1;
song_id = ['train_', num2str(i)];
[vocals, ~] = audioread(fullfile(audio_dir{i}, 'vocals.wav'));
if size(vocals, 2) > 1, vocals = vocals(:,1); end

%% === Extract 5 seconds of activity ===
thresh = 0.001;
duration_sec = 5;
win_len = fs;
hop = fs / 2;
num_frames_needed = floor(duration_sec * fs / hop);

rms_activity = @(x) sqrt(movmean(x.^2, win_len));
vocals_mask = rms_activity(vocals) > thresh;
logical_frames = vocals_mask(1:hop:end);
start_idx = -1;
for j = 1:(length(logical_frames) - num_frames_needed)
    if all(logical_frames(j:j + num_frames_needed - 1))
        start_idx = (j - 1) * hop + 1;
        break;
    end
end
if start_idx == -1, error('No 5-second segment with full activity found.'); end
end_idx = start_idx + duration_sec * fs - 1;
signal = vocals(start_idx:end_idx);

%% === Load RIRs ===
rir_name = 'room' + string(target_room_id) + '_' + string(target_rir_id);
if strcmp(srir_type, 'clean')
    sim_rir = load(fullfile(srir_dir, sprintf("%s_clean.mat", rir_name))).sim_rir_clean;
else
    sim_rir = load(fullfile(srir_dir, sprintf("%s.mat", rir_name))).sim_rir;
end
sim_rir = double(sim_rir);

%% === Simulate multichannel capture ===
nmic = size(sim_rir,1);
signal_ref = zeros(nmic, length(signal) + size(sim_rir,2) - 1);
for m = 1:nmic
    signal_ref(m,:) = conv(sim_rir(m,:), signal);
end

% Tetrahedrical mic
nmic = length(sampling_mics);
sim_rir = sim_rir(sampling_mics,:);

% RIR HOA ENCODING
if strcmp(srir_type, 'clean')
    rir_hoa = load(fullfile(srir_dir, sprintf("hoa_%s_clean.mat", rir_name))).rir_hoa;
else
    rir_hoa = load(fullfile(srir_dir, sprintf("hoa_%s.mat", rir_name))).rir_hoa;
end
signal_hoa = zeros(ncoeff, length(signal) + size(rir_hoa,2) - 1);
for shc=1:ncoeff
    signal_hoa(shc,:) = conv(rir_hoa(shc,:), signal);
end

%% Beamformer
beamformer_order = P.hoa_encoding.beamformer_order;
E = HOAbeamformer(signal_hoa, beamformer_order);

%% BACK TO A-Format 
b_n_len = P.hoa_encoding.b_n_len;
[b_n_in_time, ~] = compute_bn_fir_filters(sh_order, fs, b_n_len, c, rmic, isRigid);
nsamples = size(signal_hoa,2);
audio_SH = zeros(ncoeff,nsamples+b_n_len);
for shc=1:ncoeff
    audio_SH(shc,1:nsamples) = signal_hoa(shc,:);
    audio_SH(shc,:) = filter(b_n_in_time(shc,:),1,audio_SH(shc,:));
end

% Estima la presión a partir de los coeficientes SH (version SHTools)
Y = SHTools.getRealSH(sh_order, mic_el, mic_az);
S = pinv(Y');
signal_est = tensorprod(Y',audio_SH,2,1);
% Compenso el retardo de las convoluciones
signal_est = signal_est(:,b_n_len+1:end);

% Ensure both signals are column vectors
min_len = min(size(signal_ref,2),size(signal_est,2));
signal_ref = signal_ref(:,1:min_len);
signal_est = signal_est(:,1:min_len);

% Compute noise
noise = signal_ref - signal_est;

% Compute SNR in dB
snr_value = 10 * log10(sum(signal_ref(:).^2) / sum(noise(:).^2));

fprintf('SNR = %.2f dB\n', snr_value);
