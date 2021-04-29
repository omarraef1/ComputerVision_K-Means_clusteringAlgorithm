function hw10()

close all;
%Part A 1
sunset = 'DD_19.tiff';
tiger1 = 'tiger-1.tiff';
tiger2 = 'tiger-2.tiff';
%part A1
computeMeans(sunset,5)
computeMeans(sunset,10)

computeMeans(sunset,5)
computeMeans(sunset,10)
computeMeans(tiger1,5)
computeMeans(tiger1,10)
computeMeans(tiger2,5)
computeMeans(tiger2,10)
%}
%part A2
%
computeMeans2(sunset,5)
computeMeans2(sunset,10)
computeMeans2(tiger1,5)
computeMeans2(tiger1,10)
computeMeans2(tiger2,5)
computeMeans2(tiger2,10)
%}
%part A3
% A3a
computeTexturesA(sunset);
computeTexturesA(tiger1);
computeTexturesA(tiger2);
%A3b
computeTexturesB(sunset);
computeTexturesB(tiger1);
computeTexturesB(tiger2);
%A3c
computeTexturesC(sunset, 5);
computeTexturesC(sunset, 10);
computeTexturesC(tiger1, 5);
computeTexturesC(tiger1, 10);
computeTexturesC(tiger2, 5);
computeTexturesC(tiger2, 10);

%Part B 4


%THIS COMMENT STUB STATES THAT 
%THIS CODE IS THE PROPERTY OF OMAR R.G. (UofA Student)



end


function computeTexturesC(img,k)

newImg = computeMeans3(img, k);

%imgC = imread(newImg);
imgG = rgb2gray(uint8(newImg));
figure, imshow(imgG)
whos('imgC')

[mag, dir] = imgradient(imgG); %for mere comparison
whos('grayImg')
cEd = edge(imgG, 'canny'); % for mere comparison
figure, imshow(cEd);
mean(mean(mag))
%white: 255 255 255
%black: 0 0 0
edgeImg = imgG;
w = 2;
for i = 1:w:364
    for j = 1:w:236
        if(mag(j,i) > 0.429)    %Threshold 0.429?      
            edgeImg(j, i, 1) = 255;
            edgeImg(j, i, 2) = 255;
            edgeImg(j, i, 3) = 255;
        else
            edgeImg(j, i, 1) = 0;
            edgeImg(j, i, 2) = 0;
            edgeImg(j, i, 3) = 0;
        end
    end
end

figure, imshow(edgeImg);

end


function final_image = computeMeans3(img, k)
X = double(imread(img));
original = X;
K = k; % 32 works beautifully
max_iteration = 10;
size_X = size(X);
X = reshape(X, size_X(1)*size_X(2), 3);

initcent = initcentroidss(X, K);

[finalcent, indez] = computekmeans(initcent, X, max_iteration);
whos('indez')
lamda = 2;
final_image_vector =  finalcent(indez,:);
whos('final_image_vector')
final_image_vector =  final_image_vector;
%final_image_vector(:,4) = final_image_vector(:,1)*lamda;
%final_image_vector(:,5) = final_image_vector(:,2)*lamda;
whos('final_image_vector')

final_image = reshape(final_image_vector.*lamda, size_X(1), size_X(2), 3);
whos('final_image')
figure, subimage(uint8(final_image));

end

function computeTexturesB(img) %check again that ur doing correctly

imgC = imread(img);
imgG = rgb2gray(imgC);
whos('imgC')

[mag, dir] = imgradient(imgG); %for mere comparison
whos('grayImg')
cEd = edge(imgG, 'canny'); % for mere comparison
figure, imshow(cEd);
mean(mean(mag))
%white: 255 255 255
%black: 0 0 0
edgeImg = imgC;
w = 2;
for i = 1:w:364
    for j = 1:w:236
        if(mag(j,i) > 0.429)    %Threshold 0.429?      
            edgeImg(j, i, 1) = 255;
            edgeImg(j, i, 2) = 255;
            edgeImg(j, i, 3) = 255;
        else
            edgeImg(j, i, 1) = 0;
            edgeImg(j, i, 2) = 0;
            edgeImg(j, i, 3) = 0;
        end
    end
end

figure, imshow(edgeImg);

end

function computeTexturesA(img)

imgC = imread(img);
imgG = rgb2gray(imgC);
whos('imgC')

[mag, dir] = imgradient(imgG); %for mere comparison
whos('grayImg')
cEd = edge(imgG, 'canny'); % for mere comparison
figure, imshow(cEd);
mean(mean(mag))
%white: 255 255 255
%black: 0 0 0
edgeImg = imgG;
w = 2;
for i = 1:w:364
    for j = 1:w:236
        if(mag(j,i) > 0.429)    %Threshold 0.429?      
            edgeImg(j, i, 1) = 255;
            edgeImg(j, i, 2) = 255;
            edgeImg(j, i, 3) = 255;
        else
            edgeImg(j, i, 1) = 0;
            edgeImg(j, i, 2) = 0;
            edgeImg(j, i, 3) = 0;
        end
    end
end

figure, imshow(edgeImg);

end

function computeMeans2(img, k)
X = double(imread(img));
original = X;
K = k;
max_iteration = 10;

size_X = size(X);
X = reshape(X, size_X(1)*size_X(2), 3);

initcent = initcentroidss(X, K);

[finalcent, indez] = computekmeans(initcent, X, max_iteration);
whos('indez')
lamda = 2;
final_image_vector =  finalcent(indez,:);
whos('final_image_vector')
final_image_vector =  final_image_vector;
%final_image_vector(:,4) = final_image_vector(:,1)*lamda;
%final_image_vector(:,5) = final_image_vector(:,2)*lamda;
whos('final_image_vector')

final_image = reshape(final_image_vector.*lamda, size_X(1), size_X(2), 3);
whos('final_image')
%display image
figure, subimage(uint8(original));
figure, subimage(uint8(final_image));

end


function computeMeans(img, k)
X = double(imread(img));
original = X;
K = k;
max_iteration = 10;

size_X = size(X);
X = reshape(X, size_X(1)*size_X(2), 3);

initcent = initcentroidss(X, K);

[finalcent, indez] = computekmeans(initcent, X, max_iteration);

final_image_vector =  finalcent(indez,:);
whos('final_image_vector')
final_image_vector =  final_image_vector;
%final_image_vector(:,4) = final_image_vector(:,1)*lamda;
%final_image_vector(:,5) = final_image_vector(:,2)*lamda;
whos('final_image_vector')

final_image = reshape(final_image_vector, size_X(1), size_X(2), 3);
whos('final_image')
figure, subimage(uint8(original));
figure, subimage(uint8(final_image));

end


  
function [indez] = compnearcent(centroids, X);

indez = [];

for i = 1:size(X, 1)
    distances = [];
    for k = 1:size(centroids, 1)
        distances(k) = sqrt(sum(((X(i, :) - centroids(k, :)).^2)))^2;
    end
    [minVall, indez(i)] = min(distances);
end

end


function [centroids] = initcentroidss(X, K)
    shuffldX = X(randperm(size(X,1)),:);
    centroids = shuffldX(1:K, :);
end


function [finalcent, indez] = computekmeans(initcent, X,maxiterr)

centroids = initcent;

for i = 1:maxiterr
    indez = compnearcent(centroids, X);
    centroids = updatecentpos(centroids, X, indez);
end

finalcent = centroids;

end

function [centroids] = updatecentpos(centroids, X, indez)

currCent = [];
for k = 1:size(centroids, 1)
    indez_logical = (indez == k)';
    currCent(k,:) = sum(X .* indez_logical, 1)/sum(indez_logical);
end

centroids = currCent;
end