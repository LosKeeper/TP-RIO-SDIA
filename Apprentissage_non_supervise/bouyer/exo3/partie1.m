[data, true_centroids, true_labels] = dataset_square(10);

fig1 = figure("Name", "DATA");
clf(fig1);
movegui("northwest");
visualize_2Dclustering(fig1, data);
% pause;

fig2 = figure("Name", "GROUND TRUTH");
clf(fig2);
movegui("southwest");
visualize_2Dclustering(fig2, data, true_centroids, true_labels);
% pause;
