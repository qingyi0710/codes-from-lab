%% initializing
clc;clear;

%% loading data

gist = load('../data/ddsm/feature_mat/gist.mat','GistFeats');
hog =  load('../data/ddsm/feature_mat/hog.mat','hog');
sift =  load('../data/ddsm/feature_mat/sift.mat','countVectors');
vgg =  load('../data/ddsm/feature_mat/vgg.mat','vgg16');
% emk_sift =  load('./ddsm_feature/pca_emk_ddsmfea.mat','ddsmfea');
load('../data/ddsm/train&test.mat','Test','Train');

label1  =  find(Train == 1);
label2 = find(Train ==0);

% FilterValue = 0.0;
if ~exist('./cca_result0.0')
    mkdir('./cca_result0.0')
end

%% <1> SG fuse sift & gist
trainX = sift.countVectors(label1,:);
testX = sift.countVectors(label2,:);  

trainY = gist.GistFeats(label1,:);
testY  = gist.GistFeats(label2,:);

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'sum');
temp = [trainZ1' testZ1']';
SG_cca_sum = zeros(size(temp));
SG_cca_sum(label1,:) =  temp(1:1945,:);
SG_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/SG_cca_sum.mat','SG_cca_sum');

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'concat');
temp = [trainZ1' testZ1']';
SG_cca_concat = zeros(size(temp));
SG_cca_concat(label1,:) =  temp(1:1945,:);
SG_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/SG_cca_concat.mat','SG_cca_concat');


% <2> SH fuse sift & hog
trainX = sift.countVectors(label1,:);
testX = sift.countVectors(label2,:);  

trainY = hog.hog(label1,:);
testY  = hog.hog(label2,:);

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'sum');
temp = [trainZ1' testZ1']';
SH_cca_sum = zeros(size(temp));
SH_cca_sum(label1,:) =  temp(1:1945,:);
SH_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/SH_cca_sum.mat','SH_cca_sum');

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'concat');
temp = [trainZ1' testZ1']';
SH_cca_concat = zeros(size(temp));
SH_cca_concat(label1,:) =  temp(1:1945,:);
SH_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/SH_cca_concat.mat','SH_cca_concat');


% <3> SV fuse sift & vgg
trainX = sift.countVectors(label1,:);
testX = sift.countVectors(label2,:);  

trainY = vgg.vgg16(label1,:);
testY  = vgg.vgg16(label2,:);

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'sum');
temp = [trainZ1' testZ1']';
SV_cca_sum = zeros(size(temp));
SV_cca_sum(label1,:) =  temp(1:1945,:);
SV_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/SV_cca_sum.mat','SV_cca_sum');

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'concat');
temp = [trainZ1' testZ1']';
SV_cca_concat = zeros(size(temp));
SV_cca_concat(label1,:) =  temp(1:1945,:);
SV_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/SV_cca_concat.mat','SV_cca_concat');


% <4> GH fuse gist & hog
trainX = gist.GistFeats(label1,:);
testX = gist.GistFeats(label2,:);  

trainY = hog.hog(label1,:);
testY  = hog.hog(label2,:);

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'sum');
temp = [trainZ1' testZ1']';
GH_cca_sum = zeros(size(temp));
GH_cca_sum(label1,:) =  temp(1:1945,:);
GH_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/GH_cca_sum.mat','GH_cca_sum');

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'concat');
temp = [trainZ1' testZ1']';
GH_cca_concat = zeros(size(temp));
GH_cca_concat(label1,:) =  temp(1:1945,:);
GH_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/GH_cca_concat.mat','GH_cca_concat');


% <5> GV fuse gist & vgg
trainX = gist.GistFeats(label1,:);
testX = gist.GistFeats(label2,:);  
 
trainY = vgg.vgg16(label1,:);
testY  = vgg.vgg16(label2,:);

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'sum');
temp = [trainZ1' testZ1']';
GV_cca_sum = zeros(size(temp));
GV_cca_sum(label1,:) =  temp(1:1945,:);
GV_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/GV_cca_sum.mat','GV_cca_sum');

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'concat');
temp = [trainZ1' testZ1']';
GV_cca_concat = zeros(size(temp));
GV_cca_concat(label1,:) =  temp(1:1945,:);
GV_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/GV_cca_concat.mat','GV_cca_concat');

% <6> HV fuse gist & vgg
trainX = hog.hog(label1,:);
testX = hog.hog(label2,:);  

trainY = vgg.vgg16(label1,:);
testY  = vgg.vgg16(label2,:);

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'sum');
temp = [trainZ1' testZ1']';
HV_cca_sum = zeros(size(temp));
HV_cca_sum(label1,:) =  temp(1:1945,:);
HV_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/HV_cca_sum.mat','HV_cca_sum');

[trainZ1,testZ1] = ccaFuse(trainX, trainY, testX, testY, 'concat');
temp = [trainZ1' testZ1']';
HV_cca_concat = zeros(size(temp));
HV_cca_concat(label1,:) =  temp(1:1945,:);
HV_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result0.0/HV_cca_concat.mat','HV_cca_concat');


    