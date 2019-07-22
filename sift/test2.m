%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading  Data =============
path = 'D:/experiment/sift_feature_extraction/sift/result/';
path2 = [path 'sift_labels0.5.csv'];
file_names = dir(path2);
for i = 1:length(file_names)
    file_name = file_names(i).name;
    mat_name = file_name(1:find(file_name == '.') - 1);
    file_name = [path file_name];
    material = csvread(file_name);  
    mat_name = ['D:/experiment/sift_feature_extraction/sift/result/' mat_name '.mat'];    
    save(mat_name,'material');
    disp(mat_name);  
end

