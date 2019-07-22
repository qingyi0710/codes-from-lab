%% Initialization

clear ; close all; clc
vecLBPMap = makeLBPMap;

%% =========== Part 1: Loading  Data =============
LBPFeats_Neg=[];        LBPFeats_Pos=[];     
for j=1:2
    fprintf('Loading  Data ...\n')
    if(j==1)
        P = '../INbreast/Neg/';
    elseif(j==2)
        P = '../INbreast/Pos/';
    
    end
    D = dir([P '*.jpg']);
    %% =========== Part 2: Extract  LBPFea =============
    
    for i = 1:length(D)
        a = imread([P D(i).name]);
        if ndims(a) == 3,
            I = im2double(rgb2gray(a));
        else
            I = im2double(a);
        end;
        
        [m,n]=size(I);
        if m*n>=1000*1000
            histLBP = getLBPHist(I, 5, 4, 10);
        elseif m*n>500*500
            histLBP = getLBPHist(I, 5, 4, 5);
        elseif m*n<=500*500
            histLBP = getLBPHist(I, 5, 4, 2);
        end
        if(j==1)
            LBPFeats_Neg=[LBPFeats_Neg;histLBP];
            material1{i,1}='Neg';
%             material1=material1';
            D1=D;
        elseif(j==2)
            LBPFeats_Pos=[LBPFeats_Pos;histLBP];
            material2{i,1}='Pos';
%             material2=material2';
            D2=D;
        
        end
        
        fprintf('Finished:%d--%d\n',j,i);
    end
end
%% =========== Part 3: Saving  Data =============

LBPFeats=[LBPFeats_Neg;LBPFeats_Pos];
material=[material1;material2];
Dir=[D1;D2];
save('../result/ALLLBPFea.mat','-v7.3','material','Dir','LBPFeats','LBPFeats_Neg','LBPFeats_Pos');

