# AutoTradingBot with KNN
Auto trading bot for Forex using KNN classifier on python (scikit-learn). On Python 2.7

## Installation: 
`pip install pandas`

`pip install numpy`

`pip install scipy`

`pip install sklearn`

`pip install oandapyV20`

I'll recommend you to download datasets from Dukascopy and convert it to .npy using the same numpy

## Execute:
`python autotradeKnn.py {Complete path to your token.dat} {The pair of which you have data} {Number of Neighbors} {Take Profit} {Stop Loss}`

Where token.dat is the file where you'll write your accountID and your access token, provided by Oanda.

Disclaimer: This bot is built for test purposes, do not use in real forex accounts.



