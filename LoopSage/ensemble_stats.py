from utils import *
from sklearn.manifold import MDS
from scipy import optimize
from sklearn.cluster import DBSCAN
import seaborn as sns
import calculate_rmsd as rmsd # check https://github.com/charnley/rmsd
import os

class EnsembleStats:

    def __init__(self,path):
        self.path = path
        self.other_path = path.replace(path.split('/')[-1],'other')
        self.plots_path = path.replace(path.split('/')[-1],'plots')
        self.N = len([entry for entry in os.listdir(path) if os.path.isfile(os.path.join(path, entry))])

    def rmsd_heatmap(self,viz=True):
        '''
        This function computes a matrix with the RMSD values between 
        each pair of models in our ensemble. It finally plots a heatmap.
        Output:
        RMSD (numpy array): an array with all RMSD values between every pair of models.
        '''
        RMSD = np.zeros([self.N,self.N])

        for i in tqdm(range(self.N),desc="Loading…",ascii=False,ncols=75):
            for j in range(self.N):
                try:
                    V1 = get_coordinates_cif(file=self.path+f'/MDLE_{i}.cif')
                    V2 = get_coordinates_cif(file=self.path+f'/MDLE_{j}.cif')
                    RMSD[i,j] = rmsd.quaternion_rmsd(V1,V2)
                except:
                    raise Exception('Something is wrong with your cif files. :(')
        
        np.save(self.other_path+'/rmsd.npy',RMSD)

        if viz:
            figure(figsize=(55,30))
            sns.heatmap(RMSD, annot_kws={'size': 100}, cmap='gnuplot')
            plt.title('RMSD heatmap')
            plt.savefig(self.plots_path+'/rmsd_heatmap_N{}.svg'.format(self.N),dpi=200,format='svg')
            plt.show()

        return RMSD

    def apply_MDS(self,H,metric=False,n_init=4,max_iter=300,verbose=0,eps=1e-3,dissimilarity='euclidean'):
        '''
        This function takes the RMSD matrix as input and reduces the dimension with MDS.
        
        Input:
        H (np.array): An self.N X self.N matrix of RMSD distances.
        (the other ones are parameters of MDS sklearn class)
        metric (logical): If True, perform metric MDS; otherwise, perform nonmetric MDS.
        n_init (int): Number of times the SMACOF algorithm will be run with different initializations. The final results 
                        will be the best output of the runs, determined by the run with the smallest final stress.
        max_iter (int): Maximum number of iterations of the SMACOF algorithm for a single run.
        verbose (int): Level of verbosity.
        eps (float): Relative tolerance with respect to stress at which to declare convergence.
        dissimilarity (str): Dissimilarity measure to use:
            --> ‘euclidean’:
            Pairwise Euclidean distances between points in the dataset.
            --> ‘precomputed’:
            Pre-computed dissimilarities are passed directly to fit and fit_transform.
        Output:
        x, y (np.arrays): Each one of the dimensions of the two dimensional representation
                            of our RMSD matrix.
        '''
        embedding = MDS(n_components=2,metric=metric,n_init=n_init,max_iter=max_iter,verbose=verbose,eps=eps,dissimilarity=dissimilarity)
        H_transformed = embedding.fit_transform(H)
        print('Initial shape:',H.shape)
        print('Transformed shape:',H_transformed.shape)

        x,y = H_transformed[:,0],H_transformed[:,1]
        plt.plot(x,y,'go')
        plt.grid()
        if metric==True:
            plt.title('metric MDS scatter plot of RMSD matrix')
            plt.savefig(self.plots_path+'/metric MDS_scat_N{}.svg'.format(self.N),dpi=200,format='svg')
        else:
            plt.title('non metric MDS scatter plot of RMSD matrix')
            plt.savefig(self.plots_path+'/non metric MDS_scat_N{}.svg'.format(self.N),dpi=200,format='svg')
        plt.show()

        return H_transformed

    def apply_DBSCAN(self,X,eps=0.3,min_samples=10):
        '''
        This function applies DBASCAN algorithm so as to identify the clusters in 2-dimensional space.
        The main point of this function is to find the most representative models of each cluster.
        For each cluster we identify the representative point/model as its center.
        Input:
        X (np.array): dimension (N,2), it is our data.
        eps (float): parameter of radius of DBSCAN.
        min_samples (int): parameter of minimum cluster samples for DBSCAN
        Output:
        centroids (np.array): dimension (n_clusters,2), in each row of the matrix we have the centers of each cluster.
        repr_mds (list): A list with the most representative models of each class.
        '''

        # #############################################################################
        # Compute DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        f = open(self.other_path+'/clustering_info.txt', "w")
        f.write("Estimated number of clusters: %d\n" % n_clusters_)
        f.write("Estimated number of noise points: %d\n\n" % n_noise_)

        # #############################################################################

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )
        plt.grid()
        plt.title(r"Estimated number of clusters: {}, (for $\epsilon$={}, min_Pts={})".format(n_clusters_,eps,min_samples))
        plt.savefig(self.plots_path+'/cluster_N{}.svg'.format(self.N),dpi=200,format='svg')
        plt.show()

        def closest_node(node, nodes):
            nodes = np.asarray(nodes)
            dist_2 = np.sum((nodes - node)**2, axis=1)
            return np.argmin(dist_2)+1, dist_2
        
        centroids=[]
        repr_mds=[]
        distance_distribution = []
        
        f.write(f'Centroids of each cluster (DBSCAN with eps={eps}, min_Pts={min_samples}):\n')
        for i in range(n_clusters_):
            points_of_cluster = X[labels==i,:]
            centroid_of_cluster = np.mean(points_of_cluster, axis=0)
            repr_m, dists = closest_node(centroid_of_cluster, X)
            repr_mds.append(repr_m)
            f.write('Centroid of the Cluster {} is {}. '.format(i+1,centroid_of_cluster))
            f.write('More representative model of the cluster {} is {}.\n'.format(i+1,repr_m,))
            centroids.append(centroid_of_cluster)
            distance_distribution.append(dists)
            plt.plot(dists,label='Cluster No{}'.format(i+1))
        plt.xlabel('Model',fontsize=16)
        plt.ylabel('Distance from the Center',fontsize=16)
        if n_clusters_>1:
            plt.legend()
        plt.grid()
        plt.savefig(self.plots_path+'/clust_distances_from_center_N{}.svg'.format(self.N),dpi=200,format='svg')
        plt.show()

        plt.hist(distance_distribution,bins=self.N//10)
        plt.xlabel('Distance from Centroid',fontsize=16)
        plt.ylabel('Frequency',fontsize=16)
        plt.grid()
        plt.savefig(self.plots_path+'/clust_dist_hist_N{}.svg'.format(self.N),dpi=200,format='svg')
        plt.show()

        return np.array(centroids), repr_mds

def main():
    ens = EnsembleStats('/mnt/raid/codes/LoCR-main/stochastic_model_Nbeads_2400_chr_chr10_region_70000000_71000000/pdbs')
    R = ens.rmsd_heatmap()
    M = ens.apply_MDS(R)
    centrs, repr_mds = ens.apply_DBSCAN(M,eps=0.15,min_samples=10)
