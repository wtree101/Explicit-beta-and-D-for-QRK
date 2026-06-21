% File name parameters
n               = 100;
q               = 0.8;
beta            = 0.01;
D_min           = 1;
D_max           = 30;
c               = 1e-3;
num_samples     = 5;
T_intervals     = 100;
T_max           = 10000;
corruption_type = 'sup_c';

% Load and display heat map
filename_heat_map = sprintf('\\heat_map_data\\D_vs_T__n=%u__q=%u__beta=%u__D_min=%u__D_max=%u__c=%1.0d__num_samples=%u__T_intervals=%u__T_max=%u__corruption_type=%s.txt',n,q*100,beta*100,D_min,D_max,c,num_samples,T_intervals,T_max,corruption_type);
filename_D_min_vals = sprintf('\\heat_map_data\\D_vs_T__D_MIN__n=%u__q=%u__beta=%u__D_min=%u__D_max=%u__c=%1.0d__num_samples=%u__T_intervals=%u__T_max=%u__corruption_type=%s.txt',n,q*100,beta*100,D_min,D_max,c,num_samples,T_intervals,T_max,corruption_type);
A=readmatrix(filename_heat_map);
D_min_vals=readmatrix(filename_D_min_vals);

figure
imagesc('XData', (T_intervals:T_intervals:T_max)', 'YData', (D_min:D_max)', 'CData', A)
colormap("jet")
hold on 
plot((T_intervals:T_intervals:T_max)',D_min_vals,'LineWidth',4,'Color','black')

colorbar
hold off 