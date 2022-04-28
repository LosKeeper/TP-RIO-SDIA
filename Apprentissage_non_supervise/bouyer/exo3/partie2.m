[new_centroids, new_labels] = kmeans(data, K, centroids, iter_per_step);

fig3 = figure('Name', 'CLUSTERING');
clf(fig3);
movegui('northeast');

visualize_2Dclustering(fig3, data, centroids, new_labels); % pause;
visualize_2Dclustering(fig3, data, new_centroids, new_labels); % pause;
