%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading  Data =============
SiftFeats_cotton=[];        SiftFeats_denim=[];       SiftFeats_fleece=[];     SiftFeats_polyester=[];
SiftFeats_nylon=[];  SiftFeats_silk =[];    SiftFeats_terrycloth=[];    SiftFeats_viscose=[];   SiftFeats_wool =[];
for j=1:9
    fprintf('Loading  Data ...\n')
    if(j==1)
        P = '../add2/Cotton/';
      
    elseif(j==2)
        P = '../add2/Denim/';
    elseif(j==3)
        P ='../add2/Fleece/';
    elseif(j==4)
        P = '../add2/Nylon/';
    elseif(j==5)
        P = '../add2/Polyester/';
    elseif(j==6)
        P = '../add2/Silk/';
    elseif(j==7)
        P = '../add2/Terrycloth/';
    elseif(j==8)
        P = '../add2/Viscose/';
    elseif(j==9)
        P = '../add2/Wool/';
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
        %
        %     %记录所有sift特征
        %     SiftFea=[SiftFea;descrips];
        
        %每幅图片随机抽取100以内条特征数据
        if(size(descrips,1)>100||size(descrips,1)==100)
            descrips=descrips(randperm(size(descrips,1),100),:);
            % 在行中选择100条特征
        end
        % size(descrips,1) Returns the number of rows in the matrix descrips
        %randperm(n,k) 返回行向量，其中包含从 1 到 n（包括二者）之间的整数随机置换
        % k 表示只取1—n中的k个
        
        
        
        if(j==1)
            count1=ones(size(descrips,1),1)*i;
            descrips=[descrips count1];
            SiftFeats_cotton=[SiftFeats_cotton;descrips];
            material1{i}='cotton';
            material1=material1';
            D1=D;
        elseif(j==2)
            count2=ones(size(descrips,1),1)*(i+count1(1));
            descrips=[descrips count2];
            SiftFeats_denim=[SiftFeats_denim;descrips];
            material2{i}='denim';
            material2=material2';
            D2=D;
        elseif(j==3)
            count3=ones(size(descrips,1),1)*(i+count2(1));
            descrips=[descrips count3];
            SiftFeats_fleece=[SiftFeats_fleece;descrips];
            material3{i}='fleece';
            material3=material3';
            D3=D;
        elseif(j==4)
            count4=ones(size(descrips,1),1)*(i+count3(1));
            descrips=[descrips count4];
            SiftFeats_nylon=[SiftFeats_nylon;descrips];
            material4{i}='nylon';
            material4=material4';
            D4=D;
        elseif(j==5)
            count5=ones(size(descrips,1),1)*(i+count4(1));
            descrips=[descrips count5];
            SiftFeats_polyester=[SiftFeats_polyester;descrips];
            material5{i}='polyester';
            material5=material5';
            D5=D;
        elseif(j==6)
            count6=ones(size(descrips,1),1)*(i+count5(1));
            descrips=[descrips count6];
            SiftFeats_silk=[SiftFeats_silk;descrips];
            material6{i}='silk';
            material6=material6';
            D6=D;
        elseif(j==7)
            count7=ones(size(descrips,1),1)*(i+count6(1));
            descrips=[descrips count7];
            SiftFeats_terrycloth=[SiftFeats_terrycloth;descrips];
            material7{i}='terrycloth';
%             material7=material7';         
            D7=D;
        elseif(j==8)
            count8=ones(size(descrips,1),1)*(i+count7(1));
            descrips=[descrips count8];
            SiftFeats_viscose=[SiftFeats_viscose;descrips];
            material8{i}='viscose';      
            material8=material8';
            D8=D;
        elseif(j==9)
            count9=ones(size(descrips,1),1)*(i+count8(1));
            descrips=[descrips count9];
            SiftFeats_wool=[SiftFeats_wool;descrips];
            material9{i}='wool';
            material9=material9';
            D9=D;
        end
        fprintf('Finished:%d，%d\n',j,i);
    end
end
SiftFeats=[SiftFeats_cotton;SiftFeats_denim;SiftFeats_fleece;SiftFeats_nylon;SiftFeats_polyester;SiftFeats_silk;SiftFeats_terrycloth;SiftFeats_viscose;SiftFeats_wool];
material=[material1;material2;material3;material4;material5;material6;material7';material8;material9];
Dir=[D1;D2;D3;D4;D5;D6;D7;D8;D9];
save('.\result\ALLSiftFea_add2.mat','-v7.3','SiftFeats','SiftFeats_cotton','SiftFeats_denim','SiftFeats_fleece','SiftFeats_nylon','SiftFeats_polyester','SiftFeats_silk','SiftFeats_terrycloth','SiftFeats_viscose','SiftFeats_wool');
%% =========== Part 3: K-means =============




%选择聚类个数 K is the dimension of tfeatures that you want to get in the countVectors
K=500;

% % 提取图片库中所有图片的SIFT特征
% [img_paths,Feats] = get_sifts('C:/Users/yinyi/Documents/MATLAB/k-means+BOF/img_paths.txt');

% 随机生成K个初始类心
initMeans = SiftFeats(randperm(size(SiftFeats,1),K),:);

% 根据生成的初始类心对所有SIFT特征进行聚类
[KMeans] = K_Means(SiftFeats,K,initMeans);


% 统计图片库每张图片每个聚类中特征点个数，每张图片对应一个K维向量
[countVectors] = get_countVectors(KMeans,K,12660);

%% =========== Part 4: Saving  Data =============

save('.\result\ALLSiftVectors_add2.mat','Dir','countVectors','material');




