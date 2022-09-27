import datetime

from backtesting import *
from clusters import *
from encoder import *
from prediction import *
from preprocessing import *


def run(
    path,
    index0,
    index,
    date_start,
    date_end_train,
    date_end_test,
    nb_clusters,
    longueur,
    echantillon,
    n_in,
    latent_dim,
    input_dim,
    seuil,
    nb_iter,
    t_tracking,
    min_pip,
    min_element,
    spread=0.01,
    int_rate=0.1,
    trade_init=10,
):
    print("#################PREPOCESSING DATA...#####################")
    his = History(path + "marketdata.db", index, date_start, date_end_train)
    dates_set, data_set = process_data(his, longueur, echantillon, input_dim, "open")
    # Encoding data
    encoder_model, decoder_model, model = encoder(n_in, latent_dim, input_dim)
    model_history = model.fit(
        data_set,
        data_set,
        validation_split=0.05,
        epochs=40,
        batch_size=130,
        verbose=0,
        shuffle=True,
    )
    data_set_encoded = encoder_model.predict(data_set)
    print("#################PREPOCESSING DONE#####################")

    print("#################BUILDING THE CLASSIFIER...#####################")

    Kmeans = nRaffinements(5, nb_clusters, seuil, data_set_encoded)
    Kmeans.fit(data_set_encoded)
    print("#################CLASSIFIER IS READY#####################")

    print("#################FILTERING CLUSTERS...#####################")
    clusters_dates = cluster_dates(Kmeans, np.array(dates_set))

    pred_indexes_beta = predictive_index_1(
        Kmeans, clusters_dates, his, t_tracking=t_tracking, min_pip=min_pip, inde=index0
    )
    indexes_2 = [k for k, v in clusters_dates.items() if len(v) > min_element]
    pred_indexes = [ind for ind in pred_indexes_beta if np.abs(ind) in indexes_2]
    # pred_indexes = predictive_indexe_2(Kmeans,  pred_indexes_beta, min_element=min_element)
    print("#################FILTERATION DONE#####################")

    print("#################RUNNING BACKTEST...#####################")
    history_test = History(path + "marketdata.db", index, date_end_train, date_end_test)

    dates_test, data_test = process_data(
        history_test, longueur, echantillon, input_dim, "open"
    )
    data_test_encoded = encoder_model.predict(data_test)
    equity, leverage_buy, leverage_sel, briefing = back_testing(
        Kmeans,
        t_tracking=t_tracking,
        testing_set=data_test_encoded,
        spread=spread,
        int_rate=int_rate,
        trade_init=trade_init,
        history=history_test,
        predictive_clust=pred_indexes,
    )
    maxdrawdown = max_drawdown(equity)
    bench_return = (
        history_test["open"].iloc[len(history_test) - 1] / history_test["open"].iloc[0]
        - 1
    )

    # Pie chart: long-short chart
    # short : sell / long : buy
    labels = "LONG", "SHORT"
    sizes = [
        len(briefing.loc[briefing["position"] == "buy"]),
        len(briefing.loc[briefing["position"] == "sell"]),
    ]
    colors = ["blue", "red"]
    fig1, ax1 = plt.subplots()
    ax1.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax1.axis("equal")
    plt.title("LONG/SHORT RATIO")
    plt.savefig(path + "\\temp_orders.png")
    plt.grid()
    plt.close()

    # Equity chart
    plt.plot(equity, color="red")
    plt.ylabel("Equity")
    plt.xlabel("Time")
    plt.grid()
    plt.savefig(path + "\\temp_return.png")
    plt.close()

    # Leverage chart
    l = [lb + ls for lb, ls in zip(leverage_buy, leverage_sel)]
    plt.plot(l, color="blue")
    plt.ylabel("Leverage")
    plt.grid()
    plt.savefig(path + "\\temp_leverage.png")
    plt.close()

    output = {}
    output["SYMBOL"] = index
    output["START_DATE"] = date_end_train
    output["END_DATE"] = date_end_test
    output["DATES"] = [date_start, date_end_train, date_end_test]
    output["SPREAD"] = spread
    output["N_CLUSTERS"] = nb_clusters
    output["PREDICTIVE_CLUSTERS"] = len(pred_indexes)
    output["RETURN"] = (equity[-1] / equity[0] - 1) * 100
    output["MIN_PIPS"] = min_pip
    output["N_TRADE"] = len(briefing)
    output["WIN_RATE"] = len(briefing.loc[briefing["PnL"] > 0]) / len(briefing)
    output["MAX_DRAWDOWN"] = maxdrawdown
    s = datetime.datetime.strptime(date_end_train, "%Y-%m-%d %H:%M:%S")
    e = datetime.datetime.strptime(date_end_test, "%Y-%m-%d %H:%M:%S")

    output["SHARPE"] = (
        equity[-1] - equity[0] - (1.01 ** ((e - s).days // 365) - 1) * equity[0]
    ) / np.array(equity).std()
    output["PATH"] = path
    output["BENCHMARK_RETURN"] = bench_return * 100

    return output