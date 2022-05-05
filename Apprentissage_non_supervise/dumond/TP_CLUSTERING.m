close all;

%%%%%% Parameters of the script
K = 4; % Number of clusters
iter_per_step = 1; % Number of iterations between each visualization step
max_steps = 30; % Maximum number of steps

%%%%%% Generate dataset [Partie I]
[data, true_centroids, true_labels] = dataset_square(4);
% [data, true_centroids, true_labels] = dataset_mickeymouse(10);
% [data, true_centroids, true_labels] = dataset_flower(10);
% [data, true_centroids, true_labels] = dataset_pancakes(10);

%%%%%% Visualize dataset [Partie I]
fig1 = figure("Name","DATA");
clf(fig1);
movegui("northwest");
visualize_2Dclustering(fig1,data);

%%%%%% Visualize true clusters and centroids [Partie I]
fig2 = figure("Name","GROUND TRUTH");
clf(fig2);
movegui("southwest");
visualize_2Dclustering(fig2,data,true_centroids,true_labels);


%%%%% Initialize parameters
% Centroids (K x D) [Partie II]

N=size(data,1);
centroids = data(randperm(N,K),:); % K x D
% centroids=kmeansplusplus_init(data,K);

% Covariance matrices (K x D x D) [Partie III]
% TODO exo 9)

%%%%%% MAIN LOOP of clustering

labels=0;new_labels=0;
fig3=figure('Name','CLUSTERING');
clf(fig3);
movegui('northeast');
for i = 1:max_steps
    fprintf('Iteration: %d\n',i);
    
    %%%%% Run k-means for niter iterations [Partie II]
    [new_centroids,new_labels] = kmeans(data, K, centroids, iter_per_step);
    
    %%%%% Early stopping: stop the loop if labels have not changed
    if(all(new_labels==labels))
        break;
    end
    
    %%%%% Visualize clustering in current iteration
    visualize_2Dclustering(fig3,data,centroids,new_labels);   
    visualize_2Dclustering(fig3,data,new_centroids,new_labels);
    
    %%%% Update labels and parameters
    labels = new_labels;
    centroids = new_centroids;
end


% %%%%%% Main Loop to Visualize the Evolution of Clustering
% fig3=figure('Name','CLUSTERING');
% clf(fig3);
% movegui('northeast');
% for i = 1:max_steps
%     fprintf('Iteration: %d\n',i);
    
% %     %%%%% Run kmeans or gmm_em for iter_per_step iterations
% %     TODO exo 3) or 9)
    
%       %%%%% Early stopping: stop the loop if labels have not changed
% %     TODO exo 4)
    
% %     %%%%% Visualize clustering in current iteration
% %     TODO exo 3) or 9)
    
% %     %%%% Update labels and parameters
% %     TODO exo 3) or 9)
% %     labels = new_labels;
% %     centroids = new_centroids;
% end




