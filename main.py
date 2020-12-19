
from itertools import combinations, chain
from sklearn.preprocessing import StandardScaler


class SetWithSubset(set):
    def subsets(self):
        s1 = []
        s2 = list(self)

        def recfunc(i=0):
            if i == len(s2):
                yield frozenset(s1)
            else:
                yield from recfunc(i + 1)
                s1.append(s2[ i ])
                yield from recfunc(i + 1)
                s1.pop()

        yield from recfunc()
def get_training_data(df , factors, league_name=''):
    if league_name == '':
        df2 = df
    else:
        df2 = df.loc[df['league'] == league_name]
    df1 = df2.loc[df2['playoffs'] == 0]
    df1 = df1.reset_index()

    group = df1.groupby(['gameid'])


    inputs = []
    results = []
    for name, dfgroup in group:
        dfgroup = dfgroup.fillna(0)
        ndf1 = dfgroup.head(1)
        tempdf1 = ndf1[factors].values
        tempresult1 = ndf1['result'].values[0]
        ndf2 = dfgroup.tail(1)
        tempdf2 = ndf2[factors].values

        C1 = [a - b for a, b in zip(tempdf1, tempdf2)]

        inputs.append(C1[0])
        results.append(tempresult1)
    return inputs, results

def calc_score(row, score_multiplier_arr):

    nrow = list(row.tolist()[2:10])

    score = 0
    for i in range(len(score_multiplier_arr)):
        score = score + score_multiplier_arr[i]*nrow[i]

    return score


def get_unique_subsets(n):
    return list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))

def proccess_data2(sample_size, target_vars, score_multiplier_arr):
    import plotly as plt
    dfinit = pd.read_csv('full_lol.csv')
    ss = StandardScaler()



    df0 = dfinit[dfinit['league'] == "NA LCS"]

    df = df0[['team', 'result', 'gameid'] + target_vars]

    df1 = df.dropna()

    tempteams = df1['team']

    tempid = df1['gameid']
    df3 = df1.drop('team', 1)
    df3 = df3.drop('gameid', 1)
    df2 = pd.DataFrame(ss.fit_transform(df3), columns=df3.columns)

    df2['team'] = tempteams.tolist()
    df2['gameid'] = tempid.tolist()


    dfteams = df2.groupby('team')# df2['team'])

    team_data = pd.DataFrame()
    for name, team in dfteams:

        team1 = team.reset_index()
        if len(team1) > 100:
            team1['score'] = team1.apply(lambda x: calc_score(x, score_multiplier_arr),axis=1)
            team_data = team_data.append(team1)


    lossmean = team_data['score'].loc[team_data['result'] == -1].mean()
    winmean = team_data['score'].loc[team_data['result'] == 1].mean()
    shiftamount = winmean - lossmean
    print(">>>", lossmean, winmean, shiftamount)
    import numpy as np
    def losshift(row):
        if row['result'] == -1:
            return row['score'] + shiftamount
        return row['score']
    dfteams = team_data.groupby(team_data['team'])
    shifted_data = pd.DataFrame()
    td = []
    for name, team in dfteams:
        team1 = team.reset_index()
        team1['score'] = team1.apply(lambda x: losshift(x),axis=1)

        # team1['prev_match'] = team1['score'].shift(1)
        # team1 = team1[1:]
        td.append(team1['score'].tolist())
        team1['prev_match'] = team1['score'].rolling(window=5).mean()
        team1 = team1[4:]
        print(team1['prev_match'].head())


        shifted_data = shifted_data.append(team1)

    dfgameid = shifted_data.groupby(shifted_data['gameid'])
    final_data = []
    result_data = []
    td = []
    tr1 = []
    tr2 = []
    rr = []
    for id, game in dfgameid:
        if len(game) == 2:

            pm1 = game['prev_match'].iloc[0]
            pm2 = game['prev_match'].iloc[1]
            score1 = game['prev_match'].iloc[0]
            score2 = game['prev_match'].iloc[1]
            s1 = game['result'].iloc[0]
            final_data.append([pm1, pm2])
            tr1.append(score1)
            tr2.append(score2)
            rr.append([pm1 - pm2, s1])
            if s1 == 1:
                result_data.append(1)

                td.append(1)
            else:
                result_data.append(0)

                td.append(0)


    import matplotlib.pyplot as plt
    x = tr1
    y = tr2
    labl = td
    color = ['red' if l == 0 else 'green' for l in labl]
    plt.scatter(x, y, color=color)
    # plt.show()
    return final_data, result_data

def proccess_data(sample_size, target_vars, score_multipliers):
    import plotly as plt
    dfinit = pd.read_csv('full_lol.csv')
    ss = StandardScaler()

    df = dfinit[['team', 'result'] + target_vars]
    df1 = df.dropna()
    tempteams = df1['team']
    df3 = df1.drop('team', 1)
    df2 = pd.DataFrame(ss.fit_transform(df3), columns=df3.columns)
    df2['team'] = tempteams
    dfteams = df2.groupby(df2['team'])
    final_data = []
    result_data = []
    team_data = []
    for name, team in dfteams:

        team1 = team.reset_index()
        team1 = team1.drop('team', 1)

        data = team1.values.tolist()

        if len(data) > 50:

            seql = sample_size
            input_data = data[:-seql]

            targets = data[seql:]

            dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                input_data, targets, sequence_length=seql)

            team_stuff = []
            team_wl = []
            for batch in dataset:
                inputs, targets = batch


                tnp = list(targets)
                inp = list(inputs)
                for ind in range(0, len(tnp)):
                    gamestats = tnp[ind][2:]

                    fantasy_score = sum(gamestats * score_multipliers)
                    team_stuff.append(float(fantasy_score))
                    team_wl.append(int(tnp[ind][1]))
                    preproccessed_input = list(inp[ind])
                    proccessed_input = []
                    for p in preproccessed_input:
                        proccessed_input.append(p[2:])
                    final_data.append(proccessed_input)
                    result_data.append(float(fantasy_score))
            team_stuff = np.asarray(team_stuff, dtype=np.float32)
            team_stuff = team_stuff + -1 * team_stuff.min()
            team_data.append(team_stuff)
            team_data.append(team_wl)

    return final_data, result_data



if __name__ == "__main__":

    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import statistics as st

    score_multiplier = {'barons': 41.84,
                        'dragons': 22.58,
                        'golddiffat15': 11.95,
                        'xpdiffat15': 10.25,
                        'firsttower': 5.50,
                        'csdiffat15': 4.43,
                        'firstblood': 2.19,
                        'heralds': 1.23}


    score_multiplier_arr = [41.84, 22.58, 11.95, 10.25,
                        5.50,  4.43,  2.19,  1.23]
    target_vars = list(score_multiplier.keys())
    # target_vars = ['firstblood', 'firstdragon', 'dragons', 'goldat10', 'xpat10', 'csat10', 'goldat15', 'xpat15',
    #                'csat15']

    sample_size = 1
    final_data, result_data = proccess_data2(sample_size, target_vars, score_multiplier_arr)
    result_datanp = np.asarray(result_data, dtype=np.float32)
    result_datanp = result_datanp + -1*result_datanp.min()
    print("mean", result_datanp.mean(), "std", result_datanp.std())
    print("sample size", len(final_data) )
    partition_size = 300
    train_data = np.asarray(final_data[partition_size:], dtype=np.float32)

    train_labels = np.asarray(result_datanp[partition_size:], dtype=np.float32)
    print(list(train_data[0]))
    print(train_labels[0])
    test_data = np.asarray(final_data[:partition_size], dtype=np.float32)
    test_labels = np.asarray(result_datanp[:partition_size], dtype=np.float32)


    import pandas
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = train_data
    Y = train_labels
    Xd = train_data

    print("Input shape:", X.shape)
    print("Target shape:", Y.shape)
    learning_rate = 0.003
    def make_model(inputs):

        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        # lstm_out = keras.layers.LSTM(32)(inputs)

        lstm_out = keras.layers.LSTM(32)(inputs)
        outputs = keras.layers.Dense(1)(lstm_out)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
        model.summary()
        return model


    def baseline_model(inputs):
        # create model
        model = Sequential()
        model.add(Dense(len(inputs[1]), input_dim=len(inputs[1]), kernel_initializer='normal'))
        # model.add(Dense(8, input_dim=8, kernel_initializer='normal'))

        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


    test_results = []
    # X = [item for sublist in X for item in sublist]
    # X = np.asarray(X)


    model = baseline_model(X)

    estimator = model.fit(
        X,Y, epochs=1000
    )
    #
    for layer in model.layers:
        print("weights", layer.get_weights())
    ypred = model.predict(test_data)
    total_diff = 0

    print("Prediciton vs Actual")
    for i in range(len(test_labels)):
        print(ypred[i][0], test_labels[i])
        sqd= (ypred[i][0] - test_labels[i])**2
        if sqd != 0:
            mse = (sqd)**(0.5)
            print(mse)
            total_diff = total_diff + mse
            test_results.append([ypred[i][0], test_labels[i]])
    print("Average MSE", total_diff/len(test_labels))

    sdafgdsfg


    model = make_model(Xd)



    estimator = model.fit(
        Xd, Y, epochs=10000
    )
    #
    for layer in model.layers:
        print("weights", layer.get_weights())
    # print("score", estimator.score(test_data, test_labels))
    ypred = model.predict(test_data)
    total_diff = 0
    print("Prediciton vs Actual")

    for i in range(len(test_labels)):
        print(ypred[i][0], test_labels[i])
        sqd = (ypred[i][0] - test_labels[i]) ** 2
        if sqd != 0:
            mse = (sqd) ** (0.5)
            print(mse)
            total_diff = total_diff + mse
            test_results[i].append(ypred[i][0])
    print("Average MSE", total_diff / len(test_labels))
    print(test_results)
    import csv
    wtr = csv.writer(open('out.csv', 'w'), delimiter=',', lineterminator='\n')
    for x in test_results: wtr.writerow(x)






