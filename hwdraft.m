function hw10()

close all;

sunset = imread('DD_19.tiff');
tiger1 = imread('tiger-1.tiff');
tiger2 = imread('tiger-2.tiff');
%figure(1), imshow(sunset);
%figure(2), imshow(tiger1);
%figure(3), imshow(tiger2);
whos('sunset')
sunset
xP = sunset(:,1);
yP = sunset(:,2);
whos('yP')
whos('xP')
whos('sunset')
k = 5;
%xP = xMax * rand(1,numP);
%yP = yMax * rand(1,numP);
%points = [xP; yP];
points = double(sunset);


%% run kMeans.m and measure/print performance

tic;
[cluster, centr] = kMeans(k, points); % my k-means
myPerform = toc;
fprintf('Computation time for kMeans.m: %d seconds.\n', myPerform);


%% run MATLAB's function kmeans(P,k) and measure/print performance

tic;
[cluster_mT, centr_m] = kmeans(points',k); % MATLAB's k-means
matlabsPerform = toc;
cluster_m = cluster_mT';
fprintf('Computation time for MATLABs kmeans: %d seconds.\n', matlabsPerform);


%% Compare performances

frac = matlabsPerform/myPerform;
fprintf('MATLAB uses %d of the time kMeans.m uses.\n' ,frac);


%% All visualizations

figure('Name','Visualizations','units','normalized','outerposition',[0 0 1 1]);

% visualize the clustering
subplot(2,2,1);
scatter(xP,yP,200,cluster,'.');
hold on;
scatter(centr(1,:),centr(2,:),'xk','LineWidth',1.5);
axis([0 xMax 0 yMax]);
daspect([1 1 1]);
xlabel('x');
ylabel('y');
title('Random data points clustered (own implementation)');
grid on;

% number of points in each cluster
subplot(2,2,2);
histogram(cluster);
axis tight;
[num,~] = histcounts(cluster);
yticks(round(linspace(0,max(num),k)));
xlabel('Clusters');
ylabel('Number of data points');
title('Histogram of the cluster points (own implementation)');

% visualize MATLAB's clustering
subplot(2,2,3);
scatter(xP,yP,200,cluster_m,'.');
hold on;
scatter(centr_m(:,1),centr_m(:,2),'xk','LineWidth',1.5);
axis([0 xMax 0 yMax]);
daspect([1 1 1]);
xlabel('x');
ylabel('y');
title('Random data points clustered (MATLABs implementation)');
grid on;

% number of points in each MATLAB cluster
subplot(2,2,4);
histogram(cluster_m);
axis tight;
[num_m,~] = histcounts(cluster_m);
yticks(round(linspace(0,max(num_m),k)));
xlabel('Clusters');
ylabel('Number of data points');
title('Histogram of the cluster points (MATLABs implementation)');

end

function [ cluster, centr ] = kMeans( k, P )

%kMeans Clusters data points into k clusters.
%   Input args: k: number of clusters; 
%   points: m-by-n matrix of n m-dimensional data points.
%   Output args: cluster: 1-by-n array with values of 0,...,k-1
%   representing in which cluster the corresponding point lies in
%   centr: m-by-k matrix of the m-dimensional centroids of the k clusters


numP = size(P,2); % number of points
dimP = size(P,1); % dimension of points


%% Choose k data points as initial centroids

% choose k unique random indices between 1 and size(P,2) (number of points)
randIdx = randperm(numP,k);
% initial centroids
centr = P(:,randIdx);


%THIS COMMENT STUB STATES THAT 
%THIS CODE IS THE PROPERTY OF OMAR R.G. (UofA Student)

%% Repeat until stopping criterion is met

% init cluster array
cluster = zeros(1,numP);

% init previous cluster array clusterPrev (for stopping criterion)
clusterPrev = cluster;

% for reference: count the iterations
iterations = 0;

% init stopping criterion
stop = false; % if stopping criterion met, it changes to true

while stop == false
    
    % for each data point 
    for idxP = 1:numP
        % init distance array dist
        dist = zeros(1,k);
        % compute distance to each centroid
        for idxC=1:k
            dist(idxC) = norm(P(:,idxP)-centr(:,idxC));
        end
        % find index of closest centroid (= find the cluster)
        [~, clusterP] = min(dist);
        cluster(idxP) = clusterP;
    end
    
    % Recompute centroids using current cluster memberships:
        
    % init centroid array centr
    centr = zeros(dimP,k);
    % for every cluster compute new centroid
    for idxC = 1:k
        % find the points in cluster number idxC and compute row-wise mean
        centr(:,idxC) = mean(P(:,cluster==idxC),2);
    end
    
    % Checking for stopping criterion: Clusters do not chnage anymore
    if clusterPrev==cluster
        stop = true;
    end
    % update previous cluster clusterPrev
    clusterPrev = cluster;
    
    iterations = iterations + 1;
    
end


% for reference: print number of iterations
fprintf('kMeans.m used %d iterations of changing centroids.\n',iterations);
end
