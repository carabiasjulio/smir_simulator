function [array_x, array_y, array_z, mic_dirs_rad] = get_eigemike_pos()

% Simulate a nearly uniform array of 32 microphones on a rigid baffle, with
% the specifications of the Eigenmike array [ref2].

mic_dirs_deg = ...
    [0    21;
    32     0;
     0   -21;
   328     0;
     0    58;
    45    35;
    69     0;
    45   -35;
     0   -58;
   315   -35;
   291     0;
   315    35;
    91    69;
    90    32;
    90   -31;
    89   -69;
   180    21;
   212     0;
   180   -21;
   148     0;
   180    58;
   225    35;
   249     0;
   225   -35;
   180   -58;
   135   -35;
   111     0;
   135    35;
   269    69;
   270    32;
   270   -32;
   271   -69];

mic_dirs_rad = mic_dirs_deg*pi/180;

% Eigenmike radius
R = 0.042;

array_x = R*cos(mic_dirs_rad(:,1)).*cos(mic_dirs_rad(:,2));
array_y = R*sin(mic_dirs_rad(:,1)).*cos(mic_dirs_rad(:,2));
array_z = R*sin(mic_dirs_rad(:,2));