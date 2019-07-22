clc;clear;




trainX=load('C:\Users\Administrator\Desktop\数据\ccaFuse-master\训练\SIFT+RGB.mat');
trainY=load('C:\Users\Administrator\Desktop\数据\ccaFuse-master\训练\LBP.mat');

testX=load('C:\Users\Administrator\Desktop\数据\ccaFuse-master\测试\SIFT+RGB.mat');
testY=load('C:\Users\Administrator\Desktop\数据\ccaFuse-master\测试\LBP.mat');

[trainZ,testZ]=ccaFuse(trainX.Z, trainY.LBPFeats_j, testX.Z, testY.LBPFeats_j, 'sum');