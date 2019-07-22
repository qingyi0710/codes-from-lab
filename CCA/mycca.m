%% initializing
clc;clear;

%% loading data

gist = load('./ddsm_feature/gist.mat','GistFeats');
hog =  load('./ddsm_feature/hog.mat','hog');
sift =  load('./ddsm_feature/sift.mat','countVectors');
vgg =  load('./ddsm_feature/vgg.mat','vgg16');
emk_sift =  load('./ddsm_feature/pca_emk_ddsmfea.mat','ddsmfea');


%% <1> SG fuse sift & gist

SG_cca_concat = ccaFuse(sift.countVectors,gist.GistFeats,'concat');   
% SG_cca_sum = myccaFuse(sift.countVectors,gist.GistFeats,'sum'); 

% save('./cca_result/SG_cca_concat.mat','SG_cca_concat');
% save('./cca_result/SG_cca_sum.mat','SG_cca_sum');
% 
% %% <2> SH fuse sift & hog
% 
% SH_cca_concat = myccaFuse(sift.countVectors,hog.hog,'concat');   
% SH_cca_sum = myccaFuse(sift.countVectors,hog.hog,'sum'); 
% 
% save('./cca_result/SH_cca_concat.mat','SH_cca_concat');
% save('./cca_result/SH_cca_sum.mat','SH_cca_sum');
% 
% %% <3> SV fuse sift & vgg
% 
% SV_cca_concat = myccaFuse(sift.countVectors,vgg.vgg16,'concat');   
% SV_cca_sum = myccaFuse(sift.countVectors,vgg.vgg16,'sum'); 
% 
% save('./cca_result/SV_cca_concat.mat','SV_cca_concat');
% save('./cca_result/SV_cca_sum.mat','SV_cca_sum');
% 
% %% <4> GH fuse gist & hog
% GH_cca_concat = myccaFuse(gist.GistFeats,hog.hog,'concat');   
% GH_cca_sum = myccaFuse(gist.GistFeats,hog.hog,'sum'); 
% 
% save('./cca_result/GH_cca_concat.mat','GH_cca_concat');
% save('./cca_result/GH_cca_sum.mat','GH_cca_sum');
% 
% %% <5> GV fuse gist & vgg
% GV_cca_concat = myccaFuse(gist.GistFeats,vgg.vgg16,'concat');   
% GV_cca_sum = myccaFuse(gist.GistFeats,vgg.vgg16,'sum'); 
% 
% save('./cca_result/GV_cca_concat.mat','GV_cca_concat');
% save('./cca_result/GV_cca_sum.mat','GV_cca_sum');
% 
% %% <6> HV fuse hog & vgg
% 
% HV_cca_concat = myccaFuse(hog.hog,vgg.vgg16,'concat');   
% HV_cca_sum = myccaFuse(hog.hog,vgg.vgg16,'sum'); 
% 
% save('./cca_result/HV_cca_concat.mat','HV_cca_concat');
% save('./cca_result/HV_cca_sum.mat','HV_cca_sum');
% 

