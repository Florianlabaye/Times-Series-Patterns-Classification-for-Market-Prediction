import sys

sys.path.append("./functions")


from pdf import *
from run import *

path = ".\\"
index0 = "NONFOREX"
index = "FRA40"
date_start = "2012-01-06 07:00:00"
date_end_train = "2016-01-06 07:00:00"
date_end_test = "2017-01-06 07:00:00"
nb_clusters = 500
# 100 meilleur choix
longueur = 8
echantillon = 8

n_in = echantillon
latent_dim = 3
input_dim = 1

seuil = 0.4
nb_iter = 25

t_tracking = 20
min_pip = 0.05
min_element = 45
spread = 0.01
int_rate = 0.1
trade_init = 10

output = run(
    path,
    index0,
    index,
    date_start,
    date_end_train,
    date_end_test,
    nb_clusters,
    longueur,
    echantillon,
    n_in=8,
    latent_dim=3,
    input_dim=1,
    seuil=seuil,
    nb_iter=nb_iter,
    t_tracking=t_tracking,
    min_pip=min_pip,
    min_element=min_element,
    spread=0.01,
    int_rate=0.1,
    trade_init=10,
)


BACKTEST_REPORT(output)