% File name parameters
n           = 100;
q           = 0.8;
beta_min    = 0.00;
beta_max    = 0.02;
D_min       = 1;
D_max       = 30;
c           = 1e-3;
num_samples = 100;
T_max       = 20000;
corruption_type = 'sup_c';

% Load and display heat map
filename = sprintf('__n=%u__q=%u__beta_min=%u__beta_max=%u__D_min=%u__D_max=%u__c=%1.0d__num_samples=%u__T_max=%u__corruption_type=%s.txt',n,q*100,beta_min*100,beta_max*100,D_min,D_max,c,num_samples,T_max,corruption_type);
filename_heat_map = sprintf('\\heat_map_raw_data\\D_vs_beta%s',filename);
filename_D_min_vals = sprintf('\\heat_map_raw_data\\D_vs_beta__D_min%s',filename);
filename_D_samples = sprintf('\\heat_map_raw_data\\D_vs_beta__D_samples%s',filename);
filename_beta_samples = sprintf('\\heat_map_raw_data\\D_vs_beta__beta_samples%s',filename);

A=readmatrix(filename_heat_map);
D_min_vals=readmatrix(filename_D_min_vals);
D_samples = readmatrix(filename_D_samples);
beta_samples = readmatrix(filename_beta_samples);

figure
imagesc('XData', beta_samples, 'YData', D_samples, 'CData', A, [0 1])
colormap("jet")
hold on 
num_min_D = max(size(D_min_vals));
plot(beta_samples(1:num_min_D),D_min_vals(1:num_min_D),'LineWidth',4,'Color','black')

colorbar
hold off 