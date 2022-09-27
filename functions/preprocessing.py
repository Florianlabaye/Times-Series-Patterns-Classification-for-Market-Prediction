import sqlite3

import numpy as np
import pandas as pd
from sklearn import preprocessing


def History(path_db, symbol, start, end):
    """
    Return a pandas' dataframe containing the index we are looking for
    :param path_db: system path where the data base file is stored
    :param symbol: the symbol of the index
    :param start: starting date with the format "%Y-%M-%D %H-%M-%S"
    :param end: ending date with the format "%Y-%M-%D %H-%M-%S"
    :return: a pandas' dataframe
    """
    connection = sqlite3.connect(path_db)
    query = f"""SELECT * FROM {symbol}"""
    df = pd.read_sql(query, connection)
    connection.close()
    index_start, index_end = (
        df.index[df["date"] == start][0],
        df.index[df["date"] == end][0],
    )
    return df.iloc[index_start:index_end, :].reset_index()


def process_data(history, longueur, echantillon, input_dim, target):
    """
    Do the preprocessing step: normalize data & make it in a series of longueur-points
    :param history: dataframe
    :param longueur: length of the series of points
    :param echantillon: the number of samples taken
    :param input_dim
    :param target: the parameter we base our analysis in, it's either open or close
    :return: dates_set & data_set
    """
    # data = np.array([data_set[i:i+longueur:].tolist() for i in range(0,len(his)-longueur, longueur//echantillon)])
    dates_set = np.array(
        [history["date"].iloc[i] for i in range(0, len(history) - longueur, longueur)]
    )
    targetDf = history[target].to_list()
    data = []
    for i in range(0, len(targetDf) - longueur, longueur):
        l = targetDf[i : i + longueur]
        lComp = []
        for k in range(0, len(l), int(np.floor(len(l) / echantillon))):
            lComp.append(l[k])
        data.append(np.array(lComp))

    # On normalise les données et on les redimensionne
    data = np.array([preprocessing.scale(l) for l in data])
    data = data.reshape((len(data), -1, input_dim))
    return dates_set, data