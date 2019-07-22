%% Initialization

clear ; close all; clc

%% =========== Part 1: Loading  Data =============

GistFeats_Neg=[];        GistFeats_Pos=[]; 

for j=1:2
    fprintf('Loading  Data ...\n')
    if(j==1)
        P = '../INbreast/Neg/';
    elseif(j==2)
        P = '../INbreast/Pos/';
    
    end
    D = dir([P '*.jpg']);


%% =========== Part 2: Extract  GistFea =============

    GistFeats=[];
    for i = 1 : length(D)
        a = imread([P D(i).name]);
        % Parameters:
        clear param 
        %param.imageSize. If we do not specify the image size, the function LMgist
        %   will use the current image size. If we specify a size, the function will
        %   resize and crop the input to match the specified size. This is better when
        %   trying to compute image similarities.
        param.orientationsPerScale = [8 8 8 8];
        param.numberBlocks = 4;
        param.fc_prefilt = 4;

        % Computing gist requires 1) prefilter image, 2) filter image and collect
        % output energies
        [gist, param] = LMgist(a, '', param);
         if(j==1)
             GistFeats_Neg=[GistFeats_Neg;gist];
             material1{i}='Neg';
             material1=material1';
             D1=D;
        elseif(j==2)
            GistFeats_Pos=[GistFeats_Pos;gist];
            material2{i}='Pos';
            material2=material2';
            D2=D;
        
         end
         fprintf('Finished:%d--%d\n',j,i);   
    end

end
%% =========== Part 3: Saving  Data =============

GistFeats=[GistFeats_Neg;GistFeats_Pos];
material=[material1';material2];
Dir=[D1;D2];
save('./results/gist.mat','-v7.3','material','Dir','GistFeats','GistFeats_Neg','GistFeats_Pos');
