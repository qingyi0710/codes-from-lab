
%% Initialization
clear ;
close all;
clc;

%% =========== Part 1: Loading  Data =============
SiftFeats_0=[];        SiftFeats_1=[];      SiftFeats_2=[];
for j = 1:3
    fprintf('Loading  Data ...\n')
    if(j==1)
        P = '../../../dataset/InBreast/227_classified_image/0/';
    elseif(j==2)
        P = '../../../dataset/InBreast/227_classified_image/1/';
    elseif(j==3)
        P = '../../../dataset/InBreast/227_classified_image/2/';
    end
    D = dir([P '*.png']);
    
    %% =========== Part 2: Extract  SiftFea =============
    for i = 1 : length(D)
        a = imread([P D(i).name]);
        if ndims(a) == 3
            I = im2double(rgb2gray(a));
        else
            I = im2double(a);
        end
        %     imwrite(I,'I.jpg','quality',80);
        I = single(I) ;
        [frames, descrips] = vl_sift(I);
        descrips=double(descrips');
        k = size(descrips,1);
        fprintf("the %d descrips is : %d-by-128 .",i,k);
        
        if(size(descrips,1)>100||size(descrips,1)==100)
            descrips=descrips(randperm(size(descrips,1),100),:);
            disp(size(descrips))
           
        end
        
        
         
        if(j==1)
            disp(size(descrips))
            count1=ones(size(descrips,1),1)*i;
            descrips=[descrips count1];
            SiftFeats_0=[SiftFeats_0;descrips];
            material1{i}='0';
            material1=material1';
            D1=D;
        elseif(j==2)
            disp(size(descrips))
            count2=ones(size(descrips,1),1)*(i+count1(1));
            descrips=[descrips count2];
            SiftFeats_1=[SiftFeats_1;descrips];
            material2{i}='1';
            material2=material2';
            D2=D;
       elseif(j==3)
            disp(size(descrips))
            count3=ones(size(descrips,1),1)*(i+count2(1));
            descrips=[descrips count3];
            SiftFeats_2=[SiftFeats_2;descrips];
            material3{i}='2';
            material3=material3';
            D3=D;
        end
        fprintf('Finished:%d--%d\n',j,i);
    end
end
SiftFeats=[SiftFeats_0;SiftFeats_1;SiftFeats_2];
material=[material1';material2';material3];
Dir=[D1;D2;D3];
save('../data/InBreast_SiftFea.mat','-v7.3','SiftFeats','SiftFeats_0','SiftFeats_1','SiftFeats_2');
%% =========== Part 3: K-means =============




% K is the dimension of the feature that you want to get in the countVectors
K=500;


%
initMeans = SiftFeats(randperm(size(SiftFeats,1),K),:);

%
[KMeans] = K_Means(SiftFeats,K,initMeans);



[countVectors] = get_countVectors(KMeans,K,410);

% =========== Part 4: Saving  Data =============

save('../data/InBreast_ALLSiftVectors.mat','Dir','countVectors','material');



