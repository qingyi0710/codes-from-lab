%% Initialization

clear ; close all; clc
vecLBPMap = makeLBPMap;

%% =========== Part 1: Loading  Data =============
LBPFeats_pu_b=[];        LBPFeats_canvas_b=[];        LBPFeats_polyester_b=[];
LBPFeats_nylon_b=[];       LBPFeats_pu_s=[];        LBPFeats_canvas_s=[];
for j=1:6
    fprintf('Loading  Data ...\n')
    if(j==1)
        P = 'D:\BaiduYunDownload\MattrSet\Image\bag\material_pu\';
    elseif(j==2)
        P = 'D:\BaiduYunDownload\MattrSet\Image\bag\material-canvas\';
    elseif(j==3)
        P = 'D:\BaiduYunDownload\MattrSet\Image\bag\material_polyester\';
    elseif(j==4)
        P = 'D:\BaiduYunDownload\MattrSet\Image\bag\material_nylon\';
    elseif(j==5)
        P = 'D:\BaiduYunDownload\MattrSet\Image\shoes\material-pu\';
    elseif(j==6)
        P = 'D:\BaiduYunDownload\MattrSet\Image\shoes\material-canvas\';
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
            LBPFeats_pu_b=[LBPFeats_pu_b;histLBP];
            material1{i,1}='pu';
            material1=material1;
            D1=D;
        elseif(j==2)
            LBPFeats_canvas_b=[LBPFeats_canvas_b;histLBP];
            material2{i,1}='canvas';
            material2=material2;
            D2=D;
        elseif(j==3)
            LBPFeats_polyester_b=[LBPFeats_polyester_b;histLBP];
            material3{i,1}='polyester';
            material3=material3;
            D3=D;
        elseif(j==4)
            LBPFeats_nylon_b=[LBPFeats_nylon_b;histLBP];
            material4{i,1}='nylon';
            material4=material4;
            D4=D;
        elseif(j==5)
            LBPFeats_pu_s=[LBPFeats_pu_s;histLBP];
            material5{i,1}='pu';
            material5=material5;
            D5=D;
        elseif(j==6)
            LBPFeats_canvas_s=[LBPFeats_canvas_s;histLBP];
            material6{i,1}='canvas';
            material6=material6;
            D6=D;
        end
        fprintf('Finished:%d£¬%d\n',j,i);
    end
end
%% =========== Part 3: Saving  Data =============

LBPFeats=[LBPFeats_pu_b;LBPFeats_canvas_b;LBPFeats_polyester_b;LBPFeats_nylon_b;LBPFeats_pu_s;LBPFeats_canvas_s];
material=[material1;material2;material3;material4;material5;material6];
Dir=[D1;D2;D3;D4;D5;D6];
save('D:\MATLAB\test\LBP\Mat\ALLLBPFea.mat','-v7.3','material','Dir','LBPFeats','LBPFeats_pu_b','LBPFeats_canvas_b','LBPFeats_polyester_b','LBPFeats_nylon_b','LBPFeats_pu_s','LBPFeats_canvas_s');

