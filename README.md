# LoLMachineLearning
This project aimed to predict the results of professional League of Legends games, using past data, with the ultimate goal of beating betting odds. To do this a variety of machine learning techniques were employed, but none were ultimately successful. This lack of success can be traced back by the weak link between past results and prediction on the next game. 

First dominance analysis was used to determine the key factors in winning. In order to eliminate noisy stats, only the most predictive stats were used in the end, weighted to their influence on the game. The stats and weights are:
'barons': 41.84,
'dragons': 22.58,
'golddiffat15': 11.95,
'xpdiffat15': 10.25,
'firsttower': 5.50,
'csdiffat15': 4.43,
'firstblood': 2.19,
'heralds': 1.23


By multiplying the normalized versions of these stats for a game by the weights we come up with a score that is meant to represent how well a team played. 

At first I tried to use previous game stats, along with a LSTM model to predict a score for the next game. Ultimately this was unsuccessful with the model simply guessing the mean each time. I tried a simple NN and got the same result.

Having failed to predict the score I attempted to instead see if past scores could be used to predict results. The first thing I did was to adjust past scores for the result of the game, since while both wins and losses showed a normal distribution, they were shifted. By shifting the losses upward I hoped to remove the actual outcome of the game from the stats. 

Ultimately this approach also failed, past scores did not have predictive power on the outcome of future games. The error on predictions was ~49% ie equivalent to random guessing.

Example teams and distributions before loss shifting

Score comparison before loss shifting

After loss shifting

![als](https://user-images.githubusercontent.com/16391164/102693426-7836af80-41e8-11eb-91b9-56d5b1e2b48d.png)



