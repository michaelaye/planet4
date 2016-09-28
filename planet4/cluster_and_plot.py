from planet4 import io, clustering, plotting
import numpy as np
import matplotlib.pyplot as plt

db = io.DBManager()

def cluster_and_plot(image_id, hdbscan_min_samples=None, scaler='robust', normalize=False,
                     min_samples=4, proba_cut=0.9, dir_ext='ms4_pc0.9'):
    from planet4 import plotting, clustering
    cm = clustering.ClusteringManager(do_dynamic_min_samples=False,
                                      include_angle=True,
                                      quiet=True,
                                      output_dir='new_clustering_'+dir_ext,
                                      normalize=normalize,
                                      scaler=scaler,
                                      min_samples_factor=0.1,
                                      use_DBSCAN=False,
                                      hdbscan_min_samples=hdbscan_min_samples,
                                      min_samples=min_samples,
                                      proba_cut=proba_cut)
    cm.cluster_image_id(image_id)
    plotting.plot_image_id_pipeline(image_id,
                                    save=True,
                                    savetitle='hdbscan_',
                                    cm=cm,
                                   )
    plt.close('all')
    return dict(id_=image_id)


myids = np.load('../notebooks/myids.npy')

for id_ in myids:
    print(id_)
    cluster_and_plot(id_, min_samples=4, proba_cut=0.9, dir_ext='ms4_pc0.9_',
                     normalize=True, scaler=)
