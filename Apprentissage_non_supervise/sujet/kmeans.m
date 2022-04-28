function [centroids,labels] = kmeans(data, K, init_centroids, niter)
%%%%%%%%%%%%%%%%%%%% k-means clustering algorithm %%%%%%%%%%%%%%%%%%%%%%%%%
% Description : clusters N D-dimensional data points into K classes.
%%%%%% Input:
% - data (N x D, float) : input data (N samples of dimension D)
% - K (int) : desired number of clusters
% - init_centroids (K x D, float) : initial positions of the K centroids
% - niter (int) : number of iterations
%%%%%% Output:
% - centroids (K x D, float) : output positions of the K centroids
% - labels (N x 1, int) : label of each point in 1:K
%%%%%% Author:
% antoine.deleforge@inria.fr (2021)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i=1:niter
    %%% Update labels:
    %   TODO exo 3)
    %   - Compute the centroid-to-point distance matrix (N x K)
    %   - Assign points to their closest centroid
    
    %%% Update centroids
    %   TODO exo 3)
    %   - For each label, compute the mean of points with these labels
end

end
