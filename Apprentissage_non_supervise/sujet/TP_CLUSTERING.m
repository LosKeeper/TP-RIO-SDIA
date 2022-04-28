close all;

%%%%%% Parameters of the script
K = 0; % Number of clusters
iter_per_step = 1; % Number of iterations between each visualization step
max_steps = 30; % Maximum number of steps

%%%%%% Generate dataset [Partie I]
% TODO exos 1),2)

%%%%%% Visualize dataset [Partie I] 
% TODO exos 1),2)

%%%%%% Visualize true clusters and centroids [Partie I]
% TODO exos 1),2)

%%%%% Initialize parameters
% Centroids (K x D) [Partie II]
% TODO exo 3)

% Covariance matrices (K x D x D) [Partie III]
% TODO exo 9)

%%%%%% Main Loop to Visualize the Evolution of Clustering
fig3=figure('Name','CLUSTERING');
clf(fig3);
movegui('northeast');
for i = 1:max_steps
    fprintf('Iteration: %d\n',i);
    
%     %%%%% Run kmeans or gmm_em for iter_per_step iterations
%     TODO exo 3) or 9)
    
      %%%%% Early stopping: stop the loop if labels have not changed
%     TODO exo 4)
    
%     %%%%% Visualize clustering in current iteration
%     TODO exo 3) or 9)
    
%     %%%% Update labels and parameters
%     TODO exo 3) or 9)
%     labels = new_labels;
%     centroids = new_centroids;
end




