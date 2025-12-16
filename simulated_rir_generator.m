function simulated_rir_generator(start_id,nrooms,nworkers)
% SIMULATED_RIR_GENERATOR Generate simulated RIRs using SMIR-generator
%   simulated_rir_generator(dataset_dir, start_id,nrooms,nworkers)
%   generates nrooms simulated RIRs using SMIR-generator and saves them in
%   dataset_dir/rir and dataset_dir/rir_clean. Information about the rooms
%   and RIRs is saved in dataset_dir/info.
%  INPUT:
%   dataset_dir : directory where to save the RIRs and info files
%   start_id    : starting room ID (number)
%   nrooms      : number of rooms to simulate
%   nworkers    : number of parallel workers to use (optional, default: 1)
% Example:
%   simulated_rir_generator('/mnt/share/carabias/datasets/simulated',1,100,4)
%   generates 100 simulated RIRs starting from room ID 1 using 4 parallel
%   workers and saves them in /mnt/share/carabias/datasets/simulated
%
% Author: Julio J. Carabias-Orti
% Date: June, 2025

% PARALLEL POOL
if nargin == 3, nworkers = 1; end

% Load parameters
P = readyaml('config.yaml');

dataset_dir= P.rutas.dataset_dir;

% DIRECTORIES
if ~exist(fullfile(dataset_dir,'rir'),'dir')
    mkdir(fullfile(dataset_dir,'rir'))
end
if ~exist(fullfile(dataset_dir,'rir_clean'),'dir')
    mkdir(fullfile(dataset_dir,'rir_clean'))
end
if ~exist(fullfile(dataset_dir,'info'),'dir')
    mkdir(fullfile(dataset_dir,'info'))
end

% SMIR-generator parameters
c = P.general.c;   % speed of sound m/s
fs = P.general.fs; % processing sampling frequency in Hz (MUSDB sampling frequency)
                                        
N_harm = P.rir_generator.N_harm; % Maximum number of harmonics for the spherical harmonic decomposition
K = P.rir_generator.K; % oversampling factor
reflectionOrder = P.rir_generator.reflectionOrder; % reflection order (default=-1, maximum reflection order)
src_type = P.rir_generator.src_type; % source type: 'o' for omnidirectional, 'd' for dipole
src_dist = P.rir_generator.src_dist; % interval of source distance from SMA center in meters (value will be randomly chosen in this interval)
HP=P.rir_generator.HP;  % high-pass filter flag (1: enable, 0: disable)

% ROOM
% room parameters as in 'Robust Source Counting and DOA Estimation Using Spatial Pseudo-Spectrum and Convolutional Neural Network'
room_width = P.rir_generator.room_width;   % room width interval in meters
room_length = P.rir_generator.room_length; % room length interval in meters
room_height = P.rir_generator.room_height; % room height interval in meters

% xy_src_dist_pool = 1:0.5:3;                     % source xy distance pool in meters
% z_src_pos_pool = [0.3 0.7 1.0 1.3 1.6 1.9 2.3]; % source height pool in meters

% REVERBERATION TIMES
t60 = P.rir_generator.t60; % seconds

% MICROPHONE ARRAY 
micType = P.rir_generator.micType; % microphone type: 'open' or 'rigid'

% EIGENMIKE PARAMETERS 
micR = P.eigenmike.mic_r; % Radius of the Eigenmike in meters
mic_az = P.eigenmike.mic_az; % azimuth of the microphones
mic_el = P.eigenmike.mic_el; % inclination of the microphones
mic_height = P.eigenmike.mic_height;  

% DOA RESOLUTION
% az_res = 5;    % azimuth resolution (degrees) [-az_res,az_res]
% el_res = 5;    % inclination resolution (degrees) [-in_res,in_res]
% nb_az = 360/(az_res*2);
% nb_el = (180/(el_res*2))+1;

% SPHERE SAMPLING
npoints = P.rir_generator.npoints; % 46 points on the sphere
%npoints = nb_az*(nb_el-(30/el_res));
%spl = 'uniform';

%% Store room and RIR info
room_info = array2table(nan(0,5), 'VariableNames', {'room_id', 'x', 'y', 'z','t60'});
rir_info = array2table(nan(0,7), 'VariableNames', {'room_id','rir_id', 'x', 'y', 'z','az','el'});

% Create reverberation time vector
rt60_nrooms = round(nrooms/numel(t60));
rts = []; % Vector of reverberation times for each room
for i = 1:numel(t60)
    rts = [rts, t60(i)*ones(1,rt60_nrooms)];
end

%% GENERATE SIMULATED RIRS
parfor (r_id = 1:nrooms,nworkers)
% for r_id = 1:nrooms
    rooms = start_id:start_id+nrooms-1;
    room_i = rooms(r_id);

    % Sample points on sphere
    usphere = RandSampleSphere(npoints,'stratified');
    % polSphere = cart2polSphere(usphere(:,1),usphere(:,2),usphere(:,3));

    % Sample room dimensions
    room_x = urand(room_width);
    room_y = urand(room_length);
    room_z = urand(room_height);
    L = [room_x,room_y,room_z];
    
    % Sample reverberation time
    rt60 = rts(r_id);
    nsample = round(rt60*fs);
    
    % Display info
    w = getCurrentWorker;
    n = w.ProcessId;
    fprintf('\nRoom %d:\tL: %.2f %.2f %.2f \trt60: %.2f\t process: %d',room_i,L(1),L(2),L(3),rt60,n);
    room_info = [room_info;{room_i,L(1),L(2),L(3),rt60}];
    
    % Microphone position (center of the room at mic_height)
    mic_x = L(1) / 2;
    mic_y = L(2) / 2;
    micPos = [mic_x,mic_y,mic_height];
    
    % For each point on the sphere, generate RIR
    rir_cnt = 0;
    for rir_i = 1:length(usphere)
        s = usphere(rir_i,:);
        [az_tmp, el_tmp, ~] = mycart2sph(s(1),s(2),s(3));

        % Limit elevation to [60,130] degrees
        if rad2deg(el_tmp) < 60 || rad2deg(el_tmp) > 130
            continue
        end

        % Check if source is inside the room (possible problems due to random values)
        inside = false;
        while not(inside)
            r_tmp = urand(src_dist);
            [xtmp,ytmp,ztmp] = mysph2cart(az_tmp,el_tmp,r_tmp);

            src_x = micPos(1)+xtmp;
            src_y = micPos(2)+ytmp;
            src_z = micPos(3)+ztmp;
            src = [src_x,src_y,src_z];

            inside = insideRoom(src,L);
        end

        % Store RIR info
        rir_info = [rir_info;{room_i,rir_cnt,xtmp,ytmp,ztmp,az_tmp,el_tmp}];
        
        % [src_ang(1),src_ang(2)] = mycart2sph(micPos(1)-src(1),micPos(2)-src(2),micPos(3)-src(3));
        [a,b] = mycart2sph(micPos(1)-src(1),micPos(2)-src(2),micPos(3)-src(3));
        src_ang = [a,b];
        
        rir_name = sprintf('room%d_%d',room_i,rir_cnt);
        fprintf('\n\t%d: %s',rir_cnt,rir_name);
        % Call SMIR-generator
        [rir, ~] = smir_generator(c,fs,micPos,src,L,rt60,micType,micR,[mic_az,mic_el],N_harm,nsample,K,reflectionOrder,0,HP,src_type,src_ang);
        [rir_clean, ~] = smir_generator(c,fs,micPos,src,L,0.2,micType,micR,[mic_az,mic_el],N_harm,nsample,K,0,0,HP,src_type,src_ang);

        savefiles(rir_name,rir,rir_clean,dataset_dir);
        rir_cnt = rir_cnt+1;
    end
end

% SAVE ROOM AND RIR INFO
writetable(room_info,fullfile(dataset_dir,'info',sprintf('room_info_%d_%d.csv',start_id,start_id+nrooms-1)));
writetable(rir_info,fullfile(dataset_dir,'info',sprintf('rir_info_%d_%d.csv',start_id,start_id+nrooms-1)));

end

%% HELPER FUNCTIONS
function savefiles(filename, sim_rir, sim_rir_clean, dataset_dir)

    save(fullfile(dataset_dir,'rir',filename),'sim_rir');
    save(fullfile(dataset_dir,'rir_clean',sprintf('%s_clean',filename)),'sim_rir_clean');

end

function [isInside] = insideRoom(point,room)
% INPUT
% point = [x,y,z] point coordinates 
% room = [x,y,z] room dimensions
% OUTPUT
% isInside = True, if point inside room, False otherwise

    xpp = point(1);
    ypp = point(2);
    zpp = point(3);
    
    if 0<=xpp && xpp<=room(1) && 0<=ypp && ypp<=room(2) && 0<=zpp && zpp<=room(3)
        isInside = true;
    else 
        isInside = false;
    end
end


function [x] = urand(interval)
% INPUT
% interval = [min, max]
% OUTPUT
% x = sample uniformly chosen from interval

    x = rand()*(interval(2)- interval(1))+interval(1);
end

function [polar] = cart2polSphere(x,y,z)
    polar = zeros(length(x),2);
    [polar(:,1),polar(:,2)] = mycart2sph(x,y,z);

    polar = rad2deg(polar);
    polar = mod(polar+360,360);
end
