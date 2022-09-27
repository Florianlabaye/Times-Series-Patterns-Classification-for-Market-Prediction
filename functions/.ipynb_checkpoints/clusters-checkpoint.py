import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def filtrerClusters1(
    A, itA, B, seuil
):  # On va faire l'intersection de deux sets de clusters, A et B sont des tableaux de centroids, on raffine le set A
    # itA est le nombre de raffinements que A a déjà subi
    """
    Operates intersections between two set of centroids to form a new set of centroids
    :param A: Set of centroids
    :param itA: number of iterations
    :param B: another set of centroids
    :param seuil: the threshold used to filter controids
    :return: the final set of centroids obtained after n_it intesections
    """
    new_centroids = []
    for centroidA in A:
        for centroidB in B:
            dist = np.linalg.norm(centroidA - centroidB)
            if dist < seuil:
                new_centroids.append(((itA + 1) * centroidA + centroidB) / (itA + 2))
                break
    return new_centroids


# On a les données de départ dans Liste, une marge de tolérance seuil pour la filtration.
# On fait nit fois appel à k-means, et on garde l'intersection de tous les sets créés
def nRaffinements(nit, nbClusters, seuil, data):
    """
    Builds a model of classification using intersections between centroids of KMeans' models
    :param nit: number of iterations
    :param nbClusters: the parameter k of KMEans
    :param seuil: the threshold used for intersections
    :param data_set: The input to be classified
    :return: clustering model
    """
    kmeans = KMeans(nbClusters)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    for i in range(nit):
        kmeans2 = KMeans(nbClusters)
        kmeans2.fit(data)
        temp_centroids = kmeans2.cluster_centers_
        centroids = filtrerClusters1(centroids, i, temp_centroids, seuil)
    model = KMeans(n_clusters=len(centroids), init=np.array(centroids), n_init=1)
    return model


def nRaffinements2(nit, nbClusters, seuil, data_set1, data_set2):
    """
    Builds a model of classification using intersections between centroids of KMeans' models trained using two different data_sets
    :param nit: number of iterations
    :param nbClusters: the parameter k of KMEans
    :param seuil: the threshold used for intersections
    :param data_set1: The first input to be classified
    :param data_set1: The second input to be classified
    :return: clustering model
    """
    kmeans = KMeans(nbClusters)
    kmeans.fit(data_set1)
    centroids1 = kmeans.cluster_centers_
    for i in range(nit):
        kmeans2 = KMeans(nbClusters)
        kmeans2.fit(data_set2)
        centroids2 = kmeans2.cluster_centers_
        centroids1 = filtrerClusters1(centroids1, i, centroids2, seuil)
    model = KMeans(n_clusters=len(centroids1), init=np.array(centroids1), n_init=1)
    return model


########################## Not USED FUNCTION IN THE MODEL###########################################
def optimize_clusters(data, seuil):
    """
    Optimize the number of clusters
    :param data: pandas' dataframe
    :param seuil: threshold used in the merge procedure of clusters - nRaffinements
    """
    nb = [i * 50 for i in range(1, 24)]
    var = []
    for i in nb:
        model, centroids = nRaffinements2(
            nit=10, nbClusters=i, seuil=seuil, data_set=data
        )
        model.fit(data)
        var.append(model.inertia_)
    plt.plot(nb, data)
    plt.title("intravariance des clusters")
    plt.xlabel("nombre de clusters")
    plt.ylabel("Intravariance")
    plt.savefig("Intravariace des clusters")