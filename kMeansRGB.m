function [clusteredImage, pointColours] = KMeansRGB(imageData, pointColours, iterations) 
    % 'KMeansRGB': A function that given a NxMx3 matrix of image data,
    % inital seed point colours and a maximum number of iterations, will
    % preform the K-means algorithm to cluster the image in a set number of
    % specific colours, returning a NxM matrix of the indices to the
    % colours, which are stored in a seprate output.
    % Inputs: imageData (NxMx3 image data), pointColours (Kx3 matrix of
    % point colours), iterations (integer number of iterations).
    % Outputs: clusteredImage (NxM uint8 map of indices to final
    % pointColours), pointColours(final point colours for image).
    
    
    % Convert image to (NxM)x3 matrix of colours, each column is one
    % colour. Set to double for calculations.
    imageColours = double(reshape(imageData(:), [], 3));
    
    % Create a copy of imageColours with type uint8 for fast retrieval of
    % data in the update means part of the algorithm.
    imageColours2 = uint8(imageColours);
    
    % Preallocate meanColours array with size Kx1x3.
    meanColours = zeros(size(pointColours, 1), 1, 3);
    
    % Vectorise each colour into a seperate column vector so that the
    % squaredDistances section can run instantly and preform the implicit
    % expansion quickly.
    imageRed = reshape(imageColours(:, 1), [], 1);
    imageGreen = reshape(imageColours(:, 2), [], 1);
    imageBlue = reshape(imageColours(:, 3), [], 1);

    % Begin iterating algorithm
    for i = 1:iterations
        % Vectorise the pointColours (which is the current meanColours)
        % into row vectors, which allows for the implicit expansion in the
        % squaredDistances section to work.
        meanRed = pointColours(:, 1)';
        meanGreen = pointColours(:, 2)';
        meanBlue = pointColours(:, 3)';
        
        % Calculate the distances squared to each point through implicit
        % expansion of the column and row vectors.
        squaredDistances = ((imageRed - meanRed).^2 + (imageGreen - meanGreen).^2 + (imageBlue - meanBlue).^2);
        
        % Only get indices from minimum of each row of squaredDistance
        % matrix to find assigned cluster
        [~, indices] = min(squaredDistances, [], 2);
        
        % Go over each cluster to find new mean colour of assigned points
        for j = 1:size(pointColours, 1)
            % Create logical column vector of when indices are at value
            % 'j', and cast to uint8 to be used to multiply with another
            % uint8 matrix.
            tempIndices = uint8(indices == j);
            
            % Calculate new mean colours by multiplying logical vector by
            % the imageColours matrix. Only the elements represented by a
            % logical '1' will remain. Get sum of these elements and divide
            % by the sum of the logical array (the number of elements
            % selected) to get the mean colours.
            meanColours(j, 1, 1:3) = sum(imageColours2 .* tempIndices, 1) ./ sum(tempIndices, 1);
        end
        
        % Check for convergance. Round function is not used here as
        % convergance is checked to the limit of MATLAB's precision so that
        % no pixel colours can be incorrectly identified.
        if meanColours == pointColours
            % The output clustered image is the indices matrix reshaped into 
            % the correct image size and converted to uint8 type.
            clusteredImage = uint8(reshape(indices, size(imageData, 1), size(imageData, 2)));
            
            % Exit for loop as algorithm has converged.
            return;
        else
            % Set the old pointColours to the new mean colours.
            pointColours = meanColours;
        end
        
        if i == iterations
            % Display warning message stating that the alogrithm did not
            % converge in the required number of iterations.
            warning('Maximum number of iterations was reached before convergence was achieved.');
            
            % The output clustered image is the indices matrix reshaped into 
            % the correct image size and converted to uint8 type.
            clusteredImage = uint8(reshape(indices, size(imageData, 1), size(imageData, 2)));
        end
    end
end

function randomPoints = SelectKRandomPoints(imageData, K)
    % 'SelectKRandomPoints': A function which given an image matrix and
    % number of clusters will generate a random starting point for each
    % cluster within the image data without replacement.
    % Inputs: imageData (NxMx3 image data), K (integer number of clusters)
    % Output: randomPoints (Kx2 matrix of row and column indices to random
    % points)
    
    % Author: Jack Yarndley

    % Get total number of pixels in image so that the randperm function can be used
    totalPixels = size(imageData, 1) * size(imageData, 2);
    % Use randperm function to get K unique random pixels from number of
    % total pixels.
    randPixels = randperm(totalPixels, K);
    
    % Create Kx2 matrix of row and column indices by using ind2sub on the
    % width and height of image matrix, and the random pixels temporary
    % array.
    [randomPoints(:, 1), randomPoints(:, 2)] = ind2sub([size(imageData, 1) size(imageData, 2)], randPixels);
end
function clusteredImage = AssignToClusters(imageData, pointColours)
    % 'AssignToClusters': A function that given an image matrix and a set
    % of points will find out which point each pixel is closest to,
    % assigning each pixel to a 'cluster'.
    % Inputs: imageData (NxMx3 image data), pointColours (Kx3 array of
    % RGB point colours)
    % Output: clusteredImage (NxM matrix of which cluster each pixel is 
    % assigned to)
    
    % Author: Jack Yarndley

    % Get total number of pixels in imageData
    imageSize = size(imageData, 1) * size(imageData, 2);
    
    % Create a new array of all of the colours in the image in a (N*M)x3
    % array
    imageColours = double(reshape(imageData(:), [], 3));
    
    % Initialize an empty matrix to fill with each squared distance. Matrix 
    % The array has K columns, derived from the size of the pointColours
    % array, which allows for the squaredDistance from each pixel to the
    % mean to be stored in the matrix.
    squaredDistances = zeros(imageSize, size(pointColours, 1));
    
    % Go through each RGB triplet in the pointColours matrix
    for k = 1:size(pointColours, 1)
        % Vector based implementation of the squaredDistances function. Get
        % the sum of the distance between the imageColours array and the
        % specific pointColour squared. Inserts into specific column in the
        % squaredDistances matrix.
        squaredDistances(:, k) = sum((imageColours - pointColours(k, 1:3)).^2, 2);
    end

    % Calculates minimum of each row in the squaredDistances matrix
    % (closest point) and get the indice of it. This is the index to the
    % closest mean point in the pointColours matrix.
    [~, indices] = min(squaredDistances, [], 2);

    % Reshapes the indices array into the shape of the
    % image. Returns a clustered image of type uint8 to save space 
    % (limited to 255 colours in total). 
    clusteredImage = uint8(reshape(indices, size(imageData, 1), size(imageData, 2)));
end

function newImage = CreateKColourImage(clusteredImage, pointColours)
    % 'CreateKColourImage': A function which given a NxM matrix of colour
    % indexes and a RGB index list matrix will recreate the NxMx3 array in
    % order to be displayed and saved as an image.
    % Inputs: clusteredImage (NxM matrix of indices), pointColours (Kx3
    % matrix of RGB colours points for each cluster)
    % Output: newImage (NxMx3 matrix in image format)
    
    % Author: Jack Yarndley

    % Select appropiate RGB triplet for each index in clusteredImage from 
    % the pointColours matrix and convert to uint8.
    newImage(:, 1, 1:3) = uint8(pointColours(clusteredImage(:), 1, 1:3));
    
    % Reshape the newImage matrix into a NxMx3 image matrix of the correct
    % size to be displayed.
    newImage = reshape(newImage, [size(clusteredImage) 3]);
end

function pointColours = GetRGBValuesForPoints(imageData, randomPoints)
    % 'GetRGBValuesForPoints': A function that given a matrix of row and
    % column indices corresponding to an image matrix will retun a Ax1x3
    % matrix of the RGB values for each point the the matrix.
    % Inputs: imageData (NxMx3 image data), randomPoints (Kx2 array of row
    % and column indices)
    % Output: pointColours (Kx1x3 matrix of RGB points for each requested
    % set of row and column indices)
    
    % Author: Jack Yarndley
    
    % Get total number of pixels in imageData array.
    imageSize = size(imageData, 1) * size(imageData, 2);
    
    % Convert indices given in randomPoints array to indices in the
    % imageData array. This will only given the red colour indices.
    redIndices = sub2ind(size(imageData), randomPoints(:, 1), randomPoints(:, 2));
    
    % To get the green and blue indices, the red indices are incremented by
    % multiples of the image size, as each layer of the image is stored incrementally.
    rgbIndices = [(redIndices), (redIndices + imageSize), (redIndices + 2*imageSize)];
    
    % Get the colours of each point from the indices array, convert to
    % double for calculations and store in Ax1x3 matrix.
    pointColours(:, 1, 1:3) = double(imageData(rgbIndices));
end

function squaredDistance = SquaredDistance(firstPoint, secondPoint)
    % 'SquaredDistance': A function that returns the distance squared between two
    % points in 3D space. Can be used within the AssignToClusters function in
    % order to find nearest cluster.
    % Inputs: firstPoint, secondPoint as 1x1x3, 3x1, 1x3 matrices.
    % Output: squaredDistance.
    
    % Author: Jack Yarndley

    % Return the sum of the difference squared between the two different input
    % matrices, giving the squared distance. The : operator is used to
    % return the values of the indices of each input array allowing for all
    % required shapes and sizes to be input.
    squaredDistance = sum((firstPoint(:) - secondPoint(:)).^2);
end

function meanColours = UpdateMeans(imageData, K, clusteredImage)
    % 'UpdateMeans': A function which given the image matrice, number of
    % clusters and the temporary clustered image will return the new mean
    % colour (location) values of each cluster.
    % Inputs: imageData (MxNx3 image data), K (integer number of clusters), clusteredImage
    % (MxN integer matrice of assigned clusters)
    % Output: meanColours (Kx1x3 matrice of new mean colours)
    
    % Author: Jack Yarndley

    % Initialize meanColours array with zeros.
    meanColours = zeros(K, 1, 3);
    % Reshape imageData array into a Nx3 array of each RGB triplet.
    imageColours = reshape(imageData(:), [], 3);
    
    % Go over each cluster index
    for i = 1:K
        % Select all RGB triplets from imageColours array that are assigned
        % to the cluster i in the clusteredImage matrice. Take mean of all
        % these to get RGB triplet of new mean colours, which is the new
        % mean.
        meanColours(i, 1, 1:3) = mean(imageColours(clusteredImage == i, 1:3), 1);
    end
end
