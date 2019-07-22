%% Initialization

clear ; close all; clc
vecLBPMap = makeLBPMap;

%% =========== Part 1: Loading  Data =============
LBPFeats_canvas=[];        LBPFeats_nylon=[];        LBPFeats_polyester=[];
LBPFeats_pu=[];     
for j=1:4
    fprintf('Loading  Data ...\n')
    if(j==1)
        P = 'E:\working\gistdescriptor\MattrSet\canvas\';
    elseif(j==2)
        P = 'E:\working\gistdescriptor\MattrSet\nylon\';
    elseif(j==3)
        P = 'E:\working\gistdescriptor\MattrSet\polyester\';
    elseif(j==4)
        P = 'E:\working\gistdescriptor\MattrSet\pu\';
  
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
            LBPFeats_canvas=[LBPFeats_canvas;histLBP];
            material1{i,1}='canvas';
            material1=material1;
            D1=D;
        elseif(j==2)
            LBPFeats_nylon=[LBPFeats_nylon;histLBP];
            material2{i,1}='nylon';
            material2=material2;
            D2=D;
        elseif(j==3)
            LBPFeats_polyester=[LBPFeats_polyester;histLBP];
            material3{i,1}='polyester';
            material3=material3;
            D3=D;
        elseif(j==4)
            LBPFeats_pu=[LBPFeats_pu;histLBP];
            material4{i,1}='pu'
            material4=material4;
            D4=D;
        end
        fprintf('Finished:%d£¬%d\n',j,i);
    end
end
%% =========== Part 3: Saving  Data =============

LBPFeats=[LBPFeats_canvas;LBPFeats_nylon;LBPFeats_polyester;LBPFeats_pu];
material=[material1;material2;material3;material4];
Dir=[D1;D2;D3;D4];
save('E:\working\ALLLBPFea.mat','-v7.3','material','Dir','LBPFeats','LBPFeats_canvas','LBPFeats_nylon','LBPFeats_polyester','LBPFeats_pu');

