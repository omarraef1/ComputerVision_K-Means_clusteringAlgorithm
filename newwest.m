function mainBruh()

close all;
%Part A 1
%computeClusters('DD_19.tiff', 5);
%computeClusters('DD_19.tiff', 10);
%computeClusters('tiger-1.tiff', 5);
%computeClusters('tiger-1.tiff', 10);
%computeClusters('tiger-2.tiff', 5);
%computeClusters('tiger-2.tiff', 10);

%Part A 2

%computeClusters2('DD_19.tiff', 5);
computeClusters2('tiger-1.tiff', 5);
computeClusters2('tiger-1.tiff', 10);



%Part A 3


%THIS COMMENT STUB STATES THAT 
%THIS CODE IS THE PROPERTY OF OMAR R.G. (UofA Student)




%Part B



end



function [final_image] = CreateKColourImage2(cluster_array, mean_values)

%this function creates a k-colour image that has its pixels divided into k
%clusters. All pixels in a given cluster will be recoloured using the mean
%colour values for that cluster
%Author: Steven Ho
%Inputs: cluster_array: A 2D array specifying which cluster each pixel
%                       belongs to.
%        mean_values: A 3D array where each row contains the mean colour
%                     values for the cluster corresponding to that row number.
%Outputs: final_image: A 3D array of unsigned 8-bit integers representing
%                      an RGB image. The colour of each pixel is determined
%                      by the colour associated with that cluster.

%obtain cluster array dimensions
lambda1 = 1;
lambda10 = 5;
whos('cluster_array')
whos('cluster_array')


number_of_rows = size(cluster_array, 1);
number_of_columns = size(cluster_array, 2);

%set up dimensions for final_image
final_image = zeros(number_of_rows*lambda10, number_of_columns*lambda10, 3);

%cycle through each pixel and convert to repsective cluster RGB value
for row = 1:number_of_rows
    for column = 1:number_of_columns
        for layer = 1:3
            cluster = cluster_array(row, column);
            final_image(row*lambda10, column*lambda10, layer) = mean_values(cluster, 1, layer);
            
        end
    end
end

%convert to 8-bit unsigned intergers
final_image = uint8(final_image);
end


function computeClusters2(imago, kk)
% This script converts an image containing many colours
% into one containing just k colours (where k is a small number)
% It does this using by using the k-means algorithm, a general purpose
% data science algorithm which can be used to sort data into k clusters.
% Before this script can be used the following functions must be
% implemented: computeKRandPs, GetRGBValuesForPoints, KMeansRGB and
% CreateKColourImage
% Read in an image to convert
% If enter is hit the image to read will default to clocktower.jpg

imgName = imago;
A = imread(imgName);

% get the number of colours and maximum number of iterations from the user
k = kk;
maxIter = 200;

% display the original image in figure 1
figure
imshow(A)
title(['Original image: ' imgName]);

% convert image data to double format so we can do calculations with it
A=double(A);

% visualise 3D colour space data
figure
plot3(A(:,:,1),A(:,:,2),A(:,:,3),'+b')
title(['Colour space data for ' imgName])
xlabel('red'); ylabel('green'); zlabel('blue');
axis tight
grid on

% select k points at random from the image
[points] = computeKRandPs(A,k);

% use selected points to get the colour values for our seed means
initSeedMeans = GetRGBValuesForPoints(A,points);

% use the k means algorithm to segment all pixels in the image
% into one of k clusters and calculate the corresponding means
[clusters, means] = KMeansRGB(A,initSeedMeans,maxIter);

% convert the cluster data into an image by using the corresponding colour
% for each cluster (i.e. the mean colour for that cluster)
% the output will be an unsigned 8 bit integer array
B = CreateKColourImage2(clusters,means);
whos('B')
% display the resulting k colour image and write it to a file
figure
imshow(B);
title([num2str(k) ' colour version of ' imgName ])

end


function computeClusters(imago, kk)
% This script converts an image containing many colours
% into one containing just k colours (where k is a small number)
% It does this using by using the k-means algorithm, a general purpose
% data science algorithm which can be used to sort data into k clusters.
% Before this script can be used the following functions must be
% implemented: computeKRandPs, GetRGBValuesForPoints, KMeansRGB and
% CreateKColourImage
% Read in an image to convert
% If enter is hit the image to read will default to clocktower.jpg

imgName = imago;
A = imread(imgName);

% get the number of colours and maximum number of iterations from the user
k = kk;
maxIter = 200;

% display the original image in figure 1
figure
imshow(A)
title(['Original image: ' imgName]);

% convert image data to double format so we can do calculations with it
A=double(A);

% visualise 3D colour space data
figure
plot3(A(:,:,1),A(:,:,2),A(:,:,3),'+b')
title(['Colour space data for ' imgName])
xlabel('red'); ylabel('green'); zlabel('blue');
axis tight
grid on

% select k points at random from the image
[points] = computeKRandPs(A,k);

% use selected points to get the colour values for our seed means
initSeedMeans = GetRGBValuesForPoints(A,points);

% use the k means algorithm to segment all pixels in the image
% into one of k clusters and calculate the corresponding means
[clusters, means] = KMeansRGB(A,initSeedMeans,maxIter);

% convert the cluster data into an image by using the corresponding colour
% for each cluster (i.e. the mean colour for that cluster)
% the output will be an unsigned 8 bit integer array
B = CreateKColourImage(clusters,means);

% display the resulting k colour image and write it to a file
figure
imshow(B);
title([num2str(k) ' colour version of ' imgName ])

end

function [final_image] = CreateKColourImage(cluster_array, mean_values)

%this function creates a k-colour image that has its pixels divided into k
%clusters. All pixels in a given cluster will be recoloured using the mean
%colour values for that cluster
%Author: Steven Ho
%Inputs: cluster_array: A 2D array specifying which cluster each pixel
%                       belongs to.
%        mean_values: A 3D array where each row contains the mean colour
%                     values for the cluster corresponding to that row number.
%Outputs: final_image: A 3D array of unsigned 8-bit integers representing
%                      an RGB image. The colour of each pixel is determined
%                      by the colour associated with that cluster.

%obtain cluster array dimensions
number_of_rows = size(cluster_array, 1);
number_of_columns = size(cluster_array, 2);

%set up dimensions for final_image
final_image = zeros(number_of_rows, number_of_columns, 3);

%cycle through each pixel and convert to repsective cluster RGB value
for row = 1:number_of_rows
    for column = 1:number_of_columns
        for layer = 1:3
            cluster = cluster_array(row, column);
            final_image(row, column, layer) = mean_values(cluster, 1, layer);
            
        end
    end
end

%convert to 8-bit unsigned intergers
final_image = uint8(final_image);
end

function [RGB_values] = GetRGBValuesForPoints(imgMat, points)

%This function returns the RGB colour values for a list of specified points
%from an image
%Author: Steven Ho
%Inputs: imgMat: A 3D image array from which to obtain RGB values from
%        points: A 2D array of k rows and 2 columns containing the points
%                to extract colour values for
%Outputs: RGB_values: A 3D array containing k rows, 1 column and 3 layers,
%                     representing a list of k points from the image.

%set up RGB_values array size
number_of_rows = size(points);
RGB_values = zeros(number_of_rows(1), 1, 3);

%cycle through each pixel and obtain their RGB values respectively
for row = 1:number_of_rows
    
    %saves the row and column data of 'points' to a 1x2 array 'position'.
    %This will be used for indexing.
    position = points(row,:);
    
    %extracts the RGB colour values of 'imgMat' and stores in a 3D
    %array'colourValues'.
    RGB_values(row,1,:) = imgMat(position(1),position(2),:);
    
end
end

function [clusters, newMeans] = KMeansRGB(imgMat, initialMeans, maxIterations)

%this function partitions the points in an image into k clusters, using the
%k means algorithm to do so.
%Author: Steven Ho
%Inputs: imgMat: A 3D array containing an RGB image
%        initialMeans: A 3D array containing the seed mean values which will
%                    be used to initialise the k-means algorithm.
%        maxIterations: the maximum number ofiterations to perform.
%Outputs: clusters: A 2D array specifying which cluster each pixel belongs
%                   to.
%         newMeans: A 3D array where each row contains the mean colour
%                      values for the cluster corresponding to that row number.

%set up iterations
clusterCount = size(initialMeans, 1);
iterations = 0;
newMeans = initialMeans;
prevMeans = zeros(size(initialMeans));
identical_means = isequal(newMeans, prevMeans); %check if max iterations reached

%continue looping until max iterations reached or mean values identical
while ~identical_means && iterations <= maxIterations
    
    %assignment step
    prevMeans = newMeans;
    [clusters] = AssignToClusters(imgMat, prevMeans);
    
    iterations = iterations + 1; %increase iteration number by 1
    
    %update means
    newMeans = clustersMeans(imgMat, clusterCount, clusters);
    
    %if max iterations reached, stop code and return message to user
    if iterations == maxIterations
        return
    end
    
    %update status of identical means
    identical_means = isequal(newMeans, prevMeans);
    
end

end

function [uniquePMat] = computeKRandPs(imgMat, number_of_points)

%This function, computeKRandPs generates a list of randomly
%selected pixels from an image array.

%Inputs: imgMat: 3D image from which to select points from
%        number_of_points: number of points to randomly select
%Output: selected_pixels_array: 2D array containing 'number_of_points'
%        rows and 2 columns, representing the randomly selected points.

%Reads the size of the image data
[rows, columns,~] = size(imgMat);

%out of rows*columns combinations, selects 'number_of_points' unique points
%and saves into variable 'p'
unique_points_array = randperm(rows*columns, number_of_points);

%searches for indexes (i, j)for the unique integers generated 
%in unique_points_array
[i, j] = ind2sub([rows,columns], unique_points_array);

%merges 'i' and 'j' into a 2D array 'points' containing k rows and 2
%columns by using transpose operator
uniquePMat = [(i)',(j)'];
end

function [distSquared] = SquaredDistance(frstP, scndP)

%this function calculates the square of the distance between two points in
%3D space
%Author: Steven Ho
%Inputs: frstP: an array containing three elements representing a
%                     point in 3D space
%        scndP: an array containing three elements representing a
%                      second point in 3D space
%Outputs: distSquared: the square of the distance between the two
%                           points in 3D space


%find squared distance between two repsective points
distSquared = sum((frstP - scndP) .^2);
end



function [clusterMeanMat] = clustersMeans(imgMat, clusterCount, cluster_array)

%this function calculates the mean values for each cluster group.

%Inputs: imgMat: A 3D array of an RGB image
%        clusterCount: single value of how many clusters there are
%        cluster_array: A 2D array specifying which cluster each pixel
%                       belongs to.
%Outputs: clusterMeanMat: A 3D array containing mean values for each
%                             cluster.

%preallocate clusterMeanMat size
clusterMeanMat = zeros(clusterCount, 1, 3);

%set up clusterNum array for vectorisation
clusterNum = 1:clusterCount;
[~, column] = size(clusterNum);
clusterNum = reshape(clusterNum, [1, 1, column]);

%find every point in each cluster
logMat = (cluster_array == clusterNum);
logMatDoubled = double(logMat); %convert logical array to double array

for i = 1:clusterCount %cycle through values of each cluster
    new_array = logMatDoubled(:, :, i) .* imgMat;
    number_of_points = length(find(logMat(:, :, i)));
    layer_means = sum(new_array) ./ number_of_points; %find new mean for each layer
    clusterMeanMat(i, :, :) = sum(layer_means);%find new mean for each cluster
end
%whos('clusterMeanMat')

end


function [cluster_array] = AssignToClusters(imgMat, colour_array)

%this function assigns each pixel in an image to a cluster, based on which
%mean that point is closest to.
%Author: Steven Ho
%Input: RGB_imgMat: A 3D array with m rows, n columns and 3 layers,
%                        containing an RGB image
%       colour_array: A 3D array containing k rows, 1 column and 3 layers
%                     containing the colour information of each of k means.
%Outputs: cluster_array: A 2D array with m rows and n columns, containing
%                        the corresponding cluster number for each pixel in
%                        the image.

%obtain dimensions of imgMat
[rows, columns, ~] = size(imgMat);

%rearrange arrays for vectorization
permute_imgMat = permute(imgMat, [3 2 1]);

reshaped_imgMat = reshape(permute_imgMat, [3, rows * columns]);

rearranged_colour_array = permute(colour_array, [3 2 1]);

%calculate squared distance between point and selected mean point
distance_squared_array = (reshaped_imgMat - ...
    rearranged_colour_array).^2;

%find total squared distance between each point
total_distance = sum(distance_squared_array);

%rearrange total_distance array into a 2D array
distance_array = permute(total_distance, [3 2 1]);

%find designated cluster for each point
[~, index] = min(distance_array, [], 1);

cluster = index; %in event of tie, assign to smallest cluster

%rearrange array back into original form
cluster_array = reshape(cluster, [columns, rows]);
cluster_array = permute(cluster_array, [2 1]);


end


