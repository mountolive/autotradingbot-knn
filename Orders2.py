import json
import pandas as pd
import numpy as np
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
from sklearn.neighbors import KNeighborsClassifier
from datetime import *
from time import sleep
import threading

# Technical Analysis tools

def momentum(win):
    return clp[-1]/clp[-1-win]-1.0

def SMA(win):
    return clp[-1]/clp[N-win:N].mean()-1.0

def BB(win):
    num = clp[-1]-clp[N-win:N].mean()
    return num/(2*clp[N-win:N].std())

## Token data and account ID

file = open("/home/leo/Desktop/machine learning/fx/TradingOanda/tokenID.dat","r")
accountID = file.readline().rstrip()
accessToken = file.readline().rstrip()
file.close()
api = API(access_token=accessToken)

## Lock

lock = threading.Lock()

## Reading file with the data

pairs = {"GBP_USD":{'TP':60,'SL':20,'K':56,'CLF':''},"EUR_USD":{'TP':50,'SL':15,'K':11,'CLF':''}}

t1=''

for pair in pairs:
	Xtrain = np.load(pair+"_dataX.npy")
	ytrain = np.load(pair+"_datay.npy")

	## Train KNN algorythm for each pair
	pairs[pair]['CLF'] = KNeighborsClassifier(n_neighbors=pairs[pair]['K']) # <-- Modify the number of neighbors
	pairs[pair]['CLF'].fit(Xtrain, ytrain)

	params ={"instruments": pair}

	r = pricing.PricingInfo(accountID=accountID, params=params)
	rv = api.request(r)
	df=pd.DataFrame(r.response['prices'])
	print "Current %s price = %0.5f" % (pair,float(df.closeoutAsk[0]))
	t1=datetime.strptime(str(df.time[0])[0:26],"%Y-%m-%dT%H:%M:%S.%f")
	print "Current broker time = %d:%02d" % (t1.hour,t1.minute)

ha = t1.hour
mtn=63 # minutes to next operation
minToNext = mtn-t1.minute
sleep(60*minToNext+(60-t1.second))
## sleep(30)
nt = 0

while True:
    for pair in pairs:
		lock.acquire()
		N=20
		TakeProfit = pairs[pair]['TP']*0.0001 # Take Profit
		StopLoss = pairs[pair]['SL']*0.0001   # Stop Loss 
		params = { "count":N+1,"granularity":"H1"}
		client = oandapyV20.API(access_token=accessToken)
		r = instruments.InstrumentsCandles(instrument=pair, params=params)
		try:
    	    client.request(r)
		except:
		    nt+=1
            print "Exception candles request, trial number = ",nt
            if (nt>=10):
                print "No conection, wait till next hour *** "
                minToNext = mtn-t1.minute
                sleep(60*minToNext+(60-t1.second))
                nt=0
            continue
    

		dfc=pd.DataFrame(r.response['candles'])
		clp = []
		for i in range(N):
		    clp.append(float(dfc.mid[i]['c']))
    
		clp = np.array(clp)

		Xtest = np.array([[ momentum(3)*100, SMA(N), BB(N)]])
        
   	## print current price and time
		params ={"instruments": pair}
		r = pricing.PricingInfo(accountID=accountID, params=params)
		try:
		    rv = api.request(r)
		except:
		    nt+=1
		    print "Exception current price request, trial number = ",nt
   	        if (nt>=10):
				print "No conection, wait till next hour *** "
				minToNext = mtn-t1.minute
				sleep(60*minToNext+(60-t1.second)
				nt=0
			continue
    
    
		df=pd.DataFrame(r.response['prices'])
		print "Current "+pair+" price = %0.5f" % float(df.closeoutAsk[0])
		t1=datetime.strptime(str(df.time[0])[0:26],"%Y-%m-%dT%H:%M:%S.%f")
		print "current broker time = %d:%02d" % (t1.hour,t1.minute)
		print "Momentum, SMA, BB = ",Xtest
		pred = pairs[pair]['CLF'].predict(Xtest)
		s = "* %d/%02d/%02d %d:%02d " % (t1.year,t1.month,t1.day,t1.hour,t1.minute)
		sx = s +"Xtest = %0.4f, %0.4f, %0.4f pred = %d \n" % (Xtest[0,0],Xtest[0,1],
                                                        Xtest[0,2],int(pred[0]))
		s+= pair
		s+= " Close = %0.5f " % float(df.closeoutAsk[0])

	##    break
    
    	if (t1.hour == ha):
            print "Weekend?"
            minToNext = mtn-t1.minute
            sleep(60*minToNext+(60-t1.second))
            continue
        
    	ha = t1.hour
    
    	print "Prediction using KNN (0:do nothing,1: buy, 2:sell) = ",int(pred[0])
    	action = {0: "Nothing", 1: "Buy", 2: "Sell"}
    	s+= "pred = "+action[int(pred[0])]+" \n"
    	prob = pairs[pair]['CLF'].predict_proba(Xtest)
    	print "Probability of prediction = %0.1f percent" % (prob.max()*100)



    	## MARQUET ORDER REQUEST
    	## Buy Order:
    	if int(pred[0])==1:  ## BUY
        	TP = "{:.5f}".format(float(df.closeoutAsk[0])+TakeProfit)
        	SL = "{:.5f}".format(float(df.closeoutAsk[0])-StopLoss)
        	mktOrder = {'order': {
        	    		'timeInForce': 'FOK',
        	    		'instrument': pair,
        	    		'positionFill': 'DEFAULT',
        	    		'units': '10000',
        	    		'type': 'MARKET',
        	    		'takeProfitOnFill': {
        		        	'timeInForce': 'GTC',
        	        		'price': TP},
        	    		'stopLossOnFill': {
        	        		'timeInForce': 'GTC',
        	        		'price': SL}
        	    		}
        		}
			r = orders.OrderCreate(accountID, data=mktOrder)
        
			try:
				client.request(r)
				print r.response
				s+="       Order BUY TP = "+TP+" SL = "+SL+" \n"
			except:
				nt+=1
				print "Exception (buy) trial number = ",nt
				if (nt>=10):
					print "No conection, wait till next hour *** "
					minToNext = mtn-t1.minute
					sleep(60*minToNext+(60-t1.second))
					nt=0
				continue
        
    	## Sell Order:
    	if int(pred[0])==2:  ## SELL
        	TP = "{:.5f}".format(float(df.closeoutAsk[0])-TakeProfit)
        	SL = "{:.5f}".format(float(df.closeoutAsk[0])+StopLoss)
        	mktOrder = {'order': {
        				'timeInForce': 'FOK',
        		    	'instrument': pair,
        		    	'positionFill': 'DEFAULT',
        		    	'units': '-10000',
        		    	'type': 'MARKET',
        		    	'takeProfitOnFill': {
        		    	    'timeInForce': 'GTC',
        		    	    'price': TP},
        		    	'stopLossOnFill': {
        		    	    'timeInForce': 'GTC',
        		    	    'price': SL}
        				}
        		}
        	r = orders.OrderCreate(accountID, data=mktOrder)
        	try:
        	   	client.request(r)
        	   	print r.response
        	   	s+="       Order SELL TP = "+TP+" SL = "+SL+" \n"
        	except:
        	   	nt+=1
        	   	print "Exception (sell) trial number = ",nt
        	   	if (nt>=10):
                   	       	print "No conection, wait till next hour *** "
                	       	minToNext = mtn-t1.minute
        	           	sleep(60*minToNext+(60-t1.second))
        	           	nt=0
        	   	continue
    	
	## Log File
		with open("operations.log", "a") as f:
            f.write(s)
    	f.close()
    
		with open("prediction.log", "a") as fx:
    	    fx.write(sx)
    	fx.close()
    	minToNext = mtn-t1.minute
    	nt=0
	##    sleep(30)
		lock.release()

    sleep(60*minToNext+(60-t1.second))

