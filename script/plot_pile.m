%+FHDR//////////////////////////////////////////////////////////////////////////////
% Company: Shanghai Jiao Tong University
% Engineer: Yu Huang
% Coding: UTF-8
% Create Date: 2025.3.21
% Description:
% Plot the results of sandpile fractal
%
% Revision:
% ---------------------------------------------------------------------------------
% [Date]         [By]         [Version]         [Change Log]
% ---------------------------------------------------------------------------------
% 2025/3/21      Yu Huang     1.0               First implmentation
% ---------------------------------------------------------------------------------
%
%-FHDR//////////////////////////////////////////////////////////////////////////////
clc
clear
%% load
d_size = [2000, 2000];
file = fopen('..\0.bin', "r");
data0 = fread(file, d_size, "int")';
fclose(file);
type_ = 3;
switch type_
    case 1
        file = fopen('..\1_tri.bin', "r");
        data1 = fread(file, d_size, "int")';
    case 2
        file = fopen('..\1_quad.bin', "r");
        data1 = fread(file, d_size, "int")';
    case 3
        file = fopen('..\1_hex.bin', "r");
        data1 = fread(file, d_size, "int")';
end
fclose(file);
data1(d_size / 2) = 0;
%% select
selec = [998, 998];
data_raw = data1(d_size(1) / 2 - selec(1):d_size(1) / 2 + selec(1), d_size(2) / 2 - selec(2):d_size(2) / 2 + selec(2));
s_size = size(data_raw);
%% plot
switch type_
    case 1
        c = [182, 78, 62;...
            240, 120, 61;...
            247, 235, 141;];
    case 2
        c = [182, 78, 62;...
            255, 255, 255;...
            161, 193, 159;...
            240, 193, 92];
    case 3
        c = [182, 78, 62;...
            240, 120, 61;...
            247, 235, 141;...
            255, 255, 255;...
            76, 139, 60;...
            79, 171, 181];
end
img_out = zeros([s_size, 3], 'uint8');
for i = 1:s_size(1)
    for j = 1:s_size(2)
        idx = data_raw(i ,j) + 1;
        switch type_
            case 1
                if idx > 3
                    idx = 3;
                end
            case 2
                if idx > 4
                    idx = 4;
                end
            case 3
                if idx > 6
                    idx = 6;
                end
        end
        img_out(i, j, :) = c(idx, :);
    end
end
imwrite(img_out, "pile.bmp")