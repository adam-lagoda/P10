% colormap_directory = fullfile(matlabroot, 'toolbox', 'matlab', 'codetools', 'cmap');
% addpath(colormap_directory);

% Load the data from the CSV file
data = csvread('first_frame.csv');

% Get the dimensions of the data
[rows, cols] = size(data);

% Create a meshgrid of coordinates
[X, Y] = meshgrid(linspace(-10, 10, cols), linspace(-10, 10, rows));

% Plot the 3D surface
figure;
surf(X, Y, data,'LineWidth', 0.5);
colormap jet;
xlabel('X [m]','interpreter','latex', 'FontSize', 22, 'FontWeight','bold');
ylabel('Y [m]','interpreter','latex', 'FontSize', 22,'FontWeight','bold');
zlabel('Z [m]','interpreter','latex','FontSize', 22, 'FontWeight','bold');
set(zlim([-1.5, 1.5]))
ax = gca;
ax.FontSize = 18;
% set(xlabel('X [m]','FontSize', 14), 'interpreter', 'latex');
% set(ylabel('Y [m]','FontSize', 14), 'interpreter', 'latex');
% set(zlabel('Z [m]','FontSize', 14), 'interpreter', 'latex');
% xlabel('X [m]');
% ylabel('Y [m]');
% zlabel('Z [m]');
% title('First Frame of 3D Wave Animation');
% Normalize the axis size
axis equal;