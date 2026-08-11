% % File name parameters
% n               = 100;
% q               = 0.8;
% beta            = 0.005;
% D_min           = 1;
% D_max           = 30;
% c               = 1e-3;
% num_samples     = 100;
% T_intervals     = 100;
% T_max           = 20000;
% corruption_type = 'adversarial';

% File name parameters
n               = 100;
q               = 0.8;
beta            = 0.005;
D_min           = 1;
D_max           = 30;
c               = 1e-3;
num_samples     = 5;
T_intervals     = 1000;
T_max           = 10000;
corruption_type = 'sup_c';
c_success_mode = 'bound';
maximize_c_for_success = true;



beta = floor(beta*100) / 100;
script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir,'..','heat_map_raw_data');

% Load and display heat map
filename = sprintf('__n=%u__q=%u__beta=%u__D_min=%u__D_max=%u__c=%1.0d__num_samples=%u__T_intervals=%u__T_max=%u__corruption_type=%s.txt',n,q*100,beta*100,D_min,D_max,c,num_samples,T_intervals,T_max,corruption_type);
if strcmp(c_success_mode, 'fixed')
    c_success_suffix = '';
else
    if maximize_c_for_success
        max_c_str = 'True';
    else
        max_c_str = 'False';
    end
    c_success_suffix = sprintf('__c_success=%s__max_c=%s', c_success_mode, max_c_str);
end

filename_with_c_success = sprintf('__n=%u__q=%u__beta=%u__D_min=%u__D_max=%u__c=%1.0d__num_samples=%u__T_intervals=%u__T_max=%u__corruption_type=%s%s.txt',n,q*100,beta*100,D_min,D_max,c,num_samples,T_intervals,T_max,corruption_type,c_success_suffix);
filename_heat_map = fullfile(data_dir,sprintf('D_vs_T%s',filename_with_c_success));
filename_D_min_vals = fullfile(data_dir,sprintf('D_vs_T__D_MIN%s',filename));
filename_c_bound = fullfile(data_dir,sprintf('D_vs_T__c_bound%s',filename_with_c_success));
disp(filename_heat_map)
disp(filename_D_min_vals)
A=readmatrix(filename_heat_map);
D_min_vals=readmatrix(filename_D_min_vals);

figure
imagesc('XData', (T_intervals:T_intervals:T_max)', 'YData', (D_min:D_max)', 'CData', A, [0 1])
colormap("jet")
hold on 
plot((T_intervals:T_intervals:T_max)',D_min_vals,'LineWidth',4,'Color','black')

colorbar
title(sprintf('Success heatmap (%s, c success = %s)', corruption_type, c_success_mode))
xlabel('T')
ylabel('D')
hold off 

if isfile(filename_c_bound)
    C_bound=readmatrix(filename_c_bound);

    figure
    imagesc('XData', (T_intervals:T_intervals:T_max)', 'YData', (D_min:D_max)', 'CData', C_bound)
    colormap("jet")
    hold on
    plot((T_intervals:T_intervals:T_max)',D_min_vals,'LineWidth',4,'Color','black')
    colorbar
    title('Bound c used for success judgement')
    xlabel('T')
    ylabel('D')
    hold off
end
