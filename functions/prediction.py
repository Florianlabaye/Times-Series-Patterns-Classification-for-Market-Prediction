import matplotlib.pyplot as plt
import numpy as np

# A dictionary for computing PIP values
PIP = {"NONFOREX": 100, "FOREX": 10000}


def cluster_dates(classifier, dates_set):
    """
    Classer les dates de la prédiction selon le cluster auquel elles appertiennent
    """
    clust_dates = {}
    for i in set(classifier.labels_):
        clust_dates[i] = []
    for i in range(len(classifier.labels_)):
        clust_dates[classifier.labels_[i]].append(dates_set[i])
    return clust_dates


def track_cluster(clusters_dates, history, nb_cluster, t_tracking, inde):
    """
    Measures the PIP if we track nb_cluster_th cluster during t_tracking period of time
    :param clusters_dates: dates respect to clusters
    :param history: pandas' dataframe
    :param nb_cluster: the number of the cluster we are tracking
    :param t_trucking:
    :return: a table of PIP values in each periode of time
    """
    pip = [0] * t_tracking
    count = 0
    for date in clusters_dates[nb_cluster]:
        index_start = history.index[history["date"] == date][0]
        prices = np.array(
            history.iloc[index_start : index_start + t_tracking - 1]["open"]
        )
        prices = prices / prices[0] - 1
        pip = np.array([sum(x) for x in zip(prices, pip)])
        count += 1
    return np.array(pip) / count * PIP[inde]


def predictive_index_1(classifier, clusters_dates, history, t_tracking, min_pip, inde):
    """
    Finds the predictive clusters with respect only to their PIPs
    :return: a list of the indexes of the predictive clusters
    """
    indexes = []
    for i in set(classifier.labels_):
        pip = track_cluster(clusters_dates, history, i, t_tracking, inde)
        if np.abs(pip[-1]) > min_pip:
            if pip[-1] > 0:
                indexes.append(i)
            else:
                indexes.append(-i)
    return indexes


def predictive_indexe_2(clusters_dates, pred_ind, min_element):
    """
    Computes the predictive clusters with respect to their PIPs & their the number of elements
    :param pred_ind:
    :param min_element:
    :param clusters_dates:
    :return:
    """
    indexes_2 = [k for k, v in clusters_dates.items() if len(v) > min_element]
    return [ind for ind in pred_ind if np.abs(ind) in indexes_2]


########################## Not USED FUNCTION IN THE MODEL###########################################


def plot_pips(clusters_dates, history, t_trucking, classifier):
    """
    Plots the PIP variation for different periods of time up to t_tracking
    """
    plt.figure(figsize=(10, 6))
    plt.grid()
    for i in range(max(classifier.labels_) + 1):
        pip = track_cluster(clusters_dates, history, i, t_trucking)
        plt.plot(pip)