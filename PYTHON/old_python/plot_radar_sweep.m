% plot_radar_sweep.m
% ------------------------------------------------
% Make sure this file lives in the same folder as your .nc

clc
clear 
close all

%% PARAMETERS
ncfile    = 'radar_all_sweeps.nc';
sweep_idx = 3;    % which time‐slice (1…Nt)

%% Discover variable info
info    = ncinfo(ncfile, 'echo_strength');
dimInfo = info.Dimensions;     % struct array with fields .Name and .Length
varSize = info.Size;           % e.g. [270 2048 1024]
dimNames = {dimInfo.Name};     

% find indices of each dim
iTime  = find(strcmp(dimNames,'time'),1);
iAngle = find(strcmp(dimNames,'angle'),1);
iRange = find(strcmp(dimNames,'range'),1);

% sanity check
Nt = varSize(iTime);
if sweep_idx<1 || sweep_idx>Nt
    error('sweep_idx must be between 1 and %d',Nt);
end

%% Read one sweep slab
nd    = numel(varSize);
start = ones(1,nd);
count = varSize;
start(iTime) = sweep_idx;  
count(iTime) = 1;          
echo      = ncread(ncfile,'echo_strength',start,count);

%% Read and cast coords
time_vals  = double(ncread(ncfile,'time'));   % Nt×1
angle_vals = double(ncread(ncfile,'angle'));  % Na×1, in degrees
range_vals = double(ncread(ncfile,'range'));  % Nr×1, in same units as echo2D

%% 1) Standard image‐matrix plot
figure;
imagesc(angle_vals, range_vals, echo);
set(gca,'YDir','normal');
ylabel('Range');
xlabel('Angle (°)');
title(sprintf('Echo Strength (sweep %d of %d) – Matrix View', sweep_idx, Nt));
colorbar;
axis tight;

%% 2) “Real-world” polar plot in Cartesian coords
% Build 2D grids of R and A (now as doubles)
[R_mat, A_mat] = meshgrid(range_vals, angle_vals);  

% Convert polar → Cartesian
X = -R_mat .* cosd(A_mat); % negative here just to correct for rotational direction. 
Y = R_mat .* sind(A_mat);

figure;
pcolor(X, Y, echo');
shading interp;       % smooth color transitions
axis equal tight;     % preserve aspect ratio
xlabel('X (range × cos θ)');
ylabel('Y (range × sin θ)');
title(sprintf('Echo Strength (sweep %d) – Polar View', sweep_idx));
colorbar;
