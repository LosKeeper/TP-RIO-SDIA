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

[N,D]=size(data);
centroids=init_centroids;

for i=1:niter
    %%% Update labels:
    %   - Compute the centroid-to-point distance matrix (N x K)
    %   - Assign each point to the class of the closest centroid
    diffs=(reshape(data,[N,1,D])-reshape(centroids,[1,K,D])).^2;
    dist2=sum(diffs,3);
    [~,labels]=min(dist2,[],2);
    
    %%% Update centroids
    %   - For each label, compute the mean of points with these labels
    %   - Update the centroids
    for k=1:K
        centroids(k,:)=mean(data(labels==k,:),1);
    end
end

end
