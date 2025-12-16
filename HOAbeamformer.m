function E = HOAbeamformer(hoa, sh_order) 
    % Grid (theta in [0,pi], phi in [-pi,pi])
    theta = linspace(0, pi, 60);           % elevation (theta)
    phi   = linspace(-pi, pi, 120);        % azimuth (phi)
    nPhi   = numel(phi);
    nTheta = numel(theta);

    [theta_mesh, phi_mesh] = meshgrid(theta, phi);
    
    % Build SH matrix at all positions (M x Qpos)
    %Y_l = get_real_sh(sh_order, theta_mesh(:), phi_mesh(:));
    Y_l = SHTools.getRealSH(sh_order, theta_mesh(:), phi_mesh(:));

    % Beamformer output at each position: s = y^T * hoa
    % (Qpos x 1)
    ncoeff = (sh_order+1)^2;
    s = (Y_l.' * hoa(1:ncoeff,:));
    
    % Energy map (recommended): average power across time frames if T>1
    if size(s,2) > 1
        energy = mean(abs(s).^2, 2);   % [Qpos x 1]
    else
        energy = abs(s).^2;            % [Qpos x 1]
    end
    
    E = reshape(energy, nPhi, nTheta); % matches size(phi_mesh)
    
    % Plot like contourf in degrees, similar orientation flips
    figure;
    contourf(rad2deg(phi_mesh), rad2deg(theta_mesh), E, 'LineStyle', 'none');
    colormap bone;
    title('3DOF conductor signal energy distribution');
    xlabel('Azimuth (º)');
    ylabel('Elevation (º)');
    set(gca,'XDir','reverse','YDir','reverse');  % invert axes
    axis tight;
end