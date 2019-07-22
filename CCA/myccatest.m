%% initializing
clc;clear;

%% loading data


sift =  load('../data/ddsm/feature_mat/sift.mat','countVectors');
gist = load('../data/ddsm/feature_mat/gist.mat','GistFeats');
vgg16 =  load('../data/ddsm/dl_feature_mat/ddsm_vgg16.mat','ddsm_vgg16');
resnet50 = load('../data/ddsm/dl_feature_mat/ddsm_resnet50.mat','ddsm_resnet50');
load('../data/ddsm/train&test.mat','Test','Train');

label1  =  find(Train == 1);
label2 = find(Train ==0);

FilterValue = 0.9;
if ~exist('./cca_result_fuse0.9')
    mkdir('./cca_result_fuse0.9')
end

%% <1> SG fuse sift & gist
trainX = sift.countVectors(label1,:);
testX = sift.countVectors(label2,:);  

trainY = gist.GistFeats(label1,:);
testY  = gist.GistFeats(label2,:);

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'sum',FilterValue);
temp = [trainZ1' testZ1']';
SG_cca_sum = zeros(size(temp));
SG_cca_sum(label1,:) =  temp(1:1945,:);
SG_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/SG_cca_sum.mat','SG_cca_sum');

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'concat',FilterValue);
temp = [trainZ1' testZ1']';
SG_cca_concat = zeros(size(temp));
SG_cca_concat(label1,:) =  temp(1:1945,:);
SG_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/SG_cca_concat.mat','SG_cca_concat');


%% <2> S Vgg16 fuse sift & vgg16
trainX = sift.countVectors(label1,:);
testX = sift.countVectors(label2,:);  

trainY = vgg16.ddsm_vgg16(label1,:);
testY  = vgg16.ddsm_vgg16(label2,:);

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'sum',FilterValue);
temp = [trainZ1' testZ1']';
siftvgg16_cca_sum = zeros(size(temp));
siftvgg16_cca_sum(label1,:) =  temp(1:1945,:);
siftvgg16_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/siftvgg16_cca_sum.mat','siftvgg16_cca_sum');

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'concat',FilterValue);
temp = [trainZ1' testZ1']';
siftvgg16_cca_concat = zeros(size(temp));
siftvgg16_cca_concat(label1,:) =  temp(1:1945,:);
siftvgg16_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/siftvgg16_cca_concat.mat','siftvgg16_cca_concat');


%% <3> S res50 :fuse sift & res50
trainX = sift.countVectors(label1,:);
testX = sift.countVectors(label2,:);  

trainY = resnet50.ddsm_resnet50(label1,:);
testY  = resnet50.ddsm_resnet50(label2,:);

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'sum',FilterValue);
temp = [trainZ1' testZ1']';
siftres50_cca_sum = zeros(size(temp));
siftres50_cca_sum(label1,:) =  temp(1:1945,:);
siftres50_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/siftres50_cca_sum.mat','siftres50_cca_sum');

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'concat',FilterValue);
temp = [trainZ1' testZ1']';
siftres50_cca_concat = zeros(size(temp));
siftres50_cca_concat(label1,:) =  temp(1:1945,:);
siftres50_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/siftres50_cca_concat.mat','siftres50_cca_concat');


%% <4> G vgg16: fuse gist & vgg16
trainX = gist.GistFeats(label1,:);
testX = gist.GistFeats(label2,:);  

trainY = vgg16.ddsm_vgg16(label1,:);
testY  = vgg16.ddsm_vgg16(label2,:);

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'sum',FilterValue);
temp = [trainZ1' testZ1']';
gistvgg16_cca_sum = zeros(size(temp));
gistvgg16_cca_sum(label1,:) =  temp(1:1945,:);
gistvgg16_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/gistvgg16_cca_sum.mat','gistvgg16_cca_sum');

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'concat',FilterValue);
temp = [trainZ1' testZ1']';
gistvgg16_cca_concat = zeros(size(temp));
gistvgg16_cca_concat(label1,:) =  temp(1:1945,:);
gistvgg16_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/gistvgg16_cca_concat.mat','gistvgg16_cca_concat');


%% <5> G resnet50: fuse gist & resnet50
trainX = gist.GistFeats(label1,:);
testX = gist.GistFeats(label2,:);  
 
trainY = resnet50.ddsm_resnet50(label1,:);
testY  = resnet50.ddsm_resnet50(label2,:);

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'sum',FilterValue);
temp = [trainZ1' testZ1']';
gistres50_cca_sum = zeros(size(temp));
gistres50_cca_sum(label1,:) =  temp(1:1945,:);
gistres50_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/gistres50_cca_sum.mat','gistres50_cca_sum');

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'concat',FilterValue);
temp = [trainZ1' testZ1']';
gistres50_cca_concat = zeros(size(temp));
gistres50_cca_concat(label1,:) =  temp(1:1945,:);
gistres50_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/gistres50_cca_concat.mat','gistres50_cca_concat');

%% <6> vgg16 resnet50: fuse vgg16 & resnet50
trainX = vgg16.ddsm_vgg16(label1,:);
testX = vgg16.ddsm_vgg16(label2,:);  

trainY = resnet50.ddsm_resnet50(label1,:);
testY  = resnet50.ddsm_resnet50(label2,:);

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'sum',FilterValue);
temp = [trainZ1' testZ1']';
vgg16res50_cca_sum = zeros(size(temp));
vgg16res50_cca_sum(label1,:) =  temp(1:1945,:);
vgg16res50_cca_sum(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/vgg16res50_cca_sum.mat','vgg16res50_cca_sum');

[trainZ1,testZ1] = ccaFuse_filter(trainX, trainY, testX, testY, 'concat',FilterValue);
temp = [trainZ1' testZ1']';
vgg16res50_cca_concat = zeros(size(temp));
vgg16res50_cca_concat(label1,:) =  temp(1:1945,:);
vgg16res50_cca_concat(label2,:) =  temp(1946:end,:);
save('./cca_result_fuse0.9/vgg16res50_cca_concat.mat','vgg16res50_cca_concat');


    