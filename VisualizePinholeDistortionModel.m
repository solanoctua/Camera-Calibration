
close all;
clear;
clc;

%% Camera Parameters
image_width = 1280;
image_height = 960;
%%Pinhole
K = [892.0 0.0  640.0; 0.0 890.0 480.0; 0.0 0.0 1.0]; % mean model
distCoeffs = [-0.33 0.15 0.001 0.00065 -0.045]; % mean model
if numel(distCoeffs) < 14, distCoeffs(14) = 0; end

%% https://github.com/kyamagu/mexopencv/blob/master/samples/calibration_demo.m#L352
% cameraMatrix = [fx 0 cx; 0 fy cy; 0 0 1]
%
% * focal lengths   : fx, fy
% * aspect ratio    : a = fy/fx
% * principal point : cx, cy

% distCoeffs = [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy]
%
% * radial distortion     : k1, k2, k3
% * tangential distortion : p1, p2
% * rational distortion   : k4, k5, k6
% * thin prism distortion : s1, s2, s3, s4
% * tilted distortion     : taux, tauy (ignored here)

nstep = 20;
[u,v] = meshgrid(linspace(0, image_width-1,nstep), linspace(0, image_height-1,nstep));
grid = [u(:) v(:) ones(numel(u),1)].';
%% reverse perspective projection to find location of pixel points in camera coordinate system 
xyz = K \ grid;
xp = xyz(1,:) ./ xyz(3,:);
yp = xyz(2,:) ./ xyz(3,:);
%% polynomial model for distortion
r2 = xp.^2 + yp.^2;
r4 = r2.^2;
r6 = r2.^3;
coef = (1 + distCoeffs(1)*r2 + distCoeffs(2)*r4 + distCoeffs(5)*r6) ./ (1 + distCoeffs(6)*r2 + distCoeffs(7)*r4 + distCoeffs(8)*r6); 
%% undistorted sensor points
xpp = xp.*coef + 2*distCoeffs(3)*(xp.*yp) + distCoeffs(4)*(r2 + 2*xp.^2) + distCoeffs(9)*r2 + distCoeffs(10)*r4;
ypp = yp.*coef + distCoeffs(3)*(r2 + 2*yp.^2) + 2*distCoeffs(4)*(xp.*yp) + distCoeffs(11)*r2 + distCoeffs(12)*r4;
%% re-project undistorted sensor coordinates to image
u2 = K(1,1)*xpp + K(1,3);  % u2 = f_x * x + c_x
v2 = K(2,2)*ypp + K(2,3);  % v2 = f_y * y + c_y
%% re-projection error
du = u2(:) - u(:); 
dv = v2(:) - v(:);
dr = reshape(hypot(du,dv), size(u));
%% plot
figure;
subplot(1,1,1);
quiver(u(:)+1, v(:)+1, du, dv)
hold on
plot(image_width/2, image_height/2, 'x', K(1,3), K(2,3), 'o')
[C, hC] = contour(u(1,:)+1, v(:,1)+1, dr, 'k');
clabel(C, hC)
hold off, axis ij equal tight
title('Pinhole Distortion Model'), xlabel('u'), ylabel('v')

distorted_x = 0; % px
distorted_y = 0; % px

% for i = 0:640
%     [undistorted_x,undistorted_y] = undistort(distorted_x, distorted_y, K, distCoeffs);
%     fprintf("distorted coordinates (%d, %d) ---> undistorted coordinates (%f, %f)\n", distorted_x,distorted_y,undistorted_x,undistorted_y);
%     distorted_x = undistorted_x;
%     distorted_y = undistorted_y;
% end 


function [undistorted_x, undistorted_y] = undistort(distorted_x, distorted_y, K, distCoeffs)
    %%pixel mapping
    f_x = K(1,1);
    c_x = K(1,3);
    f_y = K(2,2);
    c_y = K(2,3);
    focal_length = 1; % mm
    distorted_sensor_coord_x = (distorted_x - c_x)*focal_length/f_x ; % mm
    distorted_sensor_coord_y = (distorted_y - c_y)*focal_length/f_y ; % mm
    %% polynomial model for distortion
    r2 = distorted_sensor_coord_x.^2 + distorted_sensor_coord_y.^2;
    r4 = r2.^2;
    r6 = r2.^3;
    coef = (1 + distCoeffs(1)*r2 + distCoeffs(2)*r4 + distCoeffs(5)*r6) ./ (1 + distCoeffs(6)*r2 + distCoeffs(7)*r4 + distCoeffs(8)*r6); 
    %% undistorted sensor points
    undistorted_sensor_coord_x = distorted_sensor_coord_x.*coef + 2*distCoeffs(3)*(distorted_sensor_coord_x.*distorted_sensor_coord_y) + distCoeffs(4)*(r2 + 2*distorted_sensor_coord_x.^2) + distCoeffs(9)*r2 + distCoeffs(10)*r4;
    undistorted_sensor_coord_y = distorted_sensor_coord_y.*coef + distCoeffs(3)*(r2 + 2*distorted_sensor_coord_y.^2) + 2*distCoeffs(4)*(distorted_sensor_coord_x.*distorted_sensor_coord_y) + distCoeffs(11)*r2 + distCoeffs(12)*r4;
    %% re-project undistorted sensor coordinates to image
    undistorted_x = f_x * undistorted_sensor_coord_x + c_x/focal_length;
    undistorted_y = f_y * undistorted_sensor_coord_y + c_y/focal_length;
     
end


