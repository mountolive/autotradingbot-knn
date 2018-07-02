import datetime
import logging
import threading
import sys
from time import sleep
import pandas as pd
import numpy as np
from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
from sklearn.neighbors import KNeighborsClassifier


# Functions of the module

def setup_logger(name, log_file, level=logging.INFO):
    """Function for logger construction"""

    handler = logging.FileHandler(log_file)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def model_trainer(pair, classifier):
    """Function that loads the data for training the ML model and executes the fit"""

    try:
        Xtrain = np.load(pair + "_dataX.npy")
        ytrain = np.load(pair + "_datay.npy")
    except IOError:
        logging.info('File not found for the pair: %s' % pair)
        raise OperationInvalidException('File not found for pair: %s' % pair)

    # Train KNN algorithm for each pair
    # Modify the number of neighbors
    classifier.fit(Xtrain, ytrain)
    return classifier


def create_operation(prediction, pair, last_price, take_profit_pips=50, stop_loss_pips=20, units='10000'):
    """Function that creates the necessary data for the request of an operation"""

    tp = "{:.5f}".format(last_price + float(take_profit_pips)*0.0001 if pair != 'USD_JPY'
                         else float(take_profit_pips)*0.01)
    sl = "{:.5f}".format(last_price - float(stop_loss_pips)*0.0001 if pair != 'USD_JPY'
                         else float(stop_loss_pips)*0.01)
    return {'order': {
        'timeInForce': 'FOK',
        'instrument': pair,
        'positionFill': 'DEFAULT',
        'units': units if prediction == 1 else '-'.join(units),
        'type': 'MARKET',
        'takeProfitOnFill': {
            'timeInForce': 'GTC',
            'price': tp},
        'stopLossOnFill': {
            'timeInForce': 'GTC',
            'price': sl}
        }
    }


def main(*args):
    """Main function for program's execution"""

    if args.count() != 4:
        logging.info('You need 4 arguments for running this script: location of token.dat, '
                     'pair (e.g. GBP_USD), Number of Neighbors, Take Profit level (in pips) and '
                     'Stop Loss level (in pips)')
        raise OperationInvalidException('Not enough arguments')

    with open(args[0], 'r') as tokenfile:
        accountID = tokenfile.readline().rstrip()
        access_token = tokenfile.readline().rstrip()

    # Training the KNN
    classifier = model_trainer(args[1], KNeighborsClassifier(n_neighbors=args[2]))

    # We need to instantiate the Oanda API
    api = API(access_token=access_token)
    price_info = pricing.PricingInfo(accountID=accountID, params={'instruments': args[1]})
    res = api.request(price_info)
    df = pd.DataFrame(res.response['prices'])
    print 'Current %s price = %0.5f' % (args[1], float(df.closeoutAsk[0]))
    t1 = datetime.strptime(str(df.time[0])[0:26], "%Y-%m-%dT%H:%M:%S.%f")
    print 'Current broker time = %d:%02d' % (t1.hour, t1.minute)


    current_hour = t1.hour
    time_per_operation = 63  # minutes to next operation
    sleep_time = 60 * (time_per_operation - t1.minute) + (60 - t1.second)
    sleep(sleep_time)
    number_of_tries = 0
    # Logger
    logger = TradesLogger()

    while True:

        n = 20
        params = {'count': n + 1, 'granularity': 'H1'}
        candles = instruments.InstrumentsCandles(instrument=args[1], params=params)
        if number_of_tries < 10:
            try:
                api.request(candles)
            except Exception:
                number_of_tries += 1
                logging.warn('Exception candles request, trial number = %s') % number_of_tries
        else:
            logging.error('No connection, wait till next hour *** ')
            number_of_tries = 0
            sleep(sleep_time)

        # Instantiating indicator handler
        indicator_handler = TechinicalIndicatorBuilder(14, n)

        # Converting data to array
        candles_data = np.array([pd.DataFrame(candles.response['candles']).mid[x]['c'] for x in range(n)])

        Xtest = np.array([[indicator_handler.momentum(candles_data) * 100, indicator_handler.sma(candles_data),
                           indicator_handler.bollinger_bands(candles_data)]])

        price_info = pricing.PricingInfo(accountID=accountID, params={'instruments': args[1]})
        if number_of_tries < 10:
            try:
                res = api.request(price_info)
            except Exception:
                number_of_tries += 1
                logging.warn('Exception candles request, trial number = %s') % number_of_tries
        else:
            logging.error('No connection, wait until next hour *** ')
            number_of_tries = 0
            sleep(sleep_time)

        df = pd.DataFrame(res.response['prices'])
        print 'Current %s price = %0.5f' % (args[1], float(df.closeoutAsk[0]))
        t1 = datetime.strptime(str(df.time[0])[0:26], "%Y-%m-%dT%H:%M:%S.%f")
        print 'Current broker time = %d:%02d' % (t1.hour, t1.minute)
        print 'Momentum, SMA, BB =  ', Xtest
        prediction = classifier.predict(Xtest)
        s_oper = '* %d/%02d/%02d %d:%02d ' % (t1.year, t1.month, t1.day, t1.hour, t1.minute)
        s_pred = 'Xtest = %0.4f, %0.4f, %0.4f pred = %d \n' % (Xtest[0, 0], Xtest[0, 1],
                                                        Xtest[0, 2], int(prediction[0]))
        s_oper.join(args[1])
        s_oper.join(' Close = %0.5f ' % float(df.closeoutAsk[0]))

        if t1.hour == current_hour:
            print 'Weekend?'
            sleep(sleep_time)

        current_hour = t1.hour
        print 'Prediction using KNN (0:do nothing,1: buy, 2:sell) = ', int(prediction[0])
        action = {0: "Nothing", 1: "Buy", 2: "Sell"}
        s_oper.join('pred = %s \n' % action[int(prediction[0])])
        probability_success = classifier.predict_proba(Xtest)
        print 'Probability of prediction = %0.1f percent' % (probability_success.max() * 100)
        logger.prediction(s_pred)

        if int(prediction[0]) != 0:
            logger.operation(s_oper)
            order = create_operation(int(prediction[0]), args[1], float(df.closeoutAsk[0]), args[3], args[4])
            ord_info = orders.OrderCreate(accountID, data=order)
            if number_of_tries < 10:
                try:
                   api.request(ord_info)
                except Exception:
                    number_of_tries += 1
                    logging.warn('Exception order request, trial number = %s') % number_of_tries
            else:
                logging.error('No connection, wait until next hour *** ')
                number_of_tries = 0
                sleep(sleep_time)

        sleep(sleep_time)




class TradesLogger(logging.Logger):
    """Class for logging proposes"""

    def __init__(self, name="logger", level=logging.NOTSET):
        self._count = 0
        self._countLock = threading.Lock()
        self.prediction_logger = setup_logger('prediction_logger', 'predictions.log')
        self.operation_logger = setup_logger('operation_logger', 'operations.log')

        return super(TradesLogger, self).__init__(name, level)

    @property
    def warn_count(self):
        return self._count

    def prediction(self, msg):
        """Method for logging predictions from the ML algo"""
        self._countLock.acquire()
        self._count += 1
        self._countLock.release()
        self.prediction_logger.info(msg)

    def operation(self, msg):
        """Method for logging operations from the bot"""
        self._countLock.acquire()
        self._count += 1
        self._countLock.release()
        self.operation_logger.info(msg)


class TechinicalIndicatorBuilder(object):
    """This class defines all the indicators used in the bot for the construction
        of new features"""

    def __init__(self, periods=14, n=60):
        """ Constructor, uses periods and total data to analyse
        per iteration"""
        # The number of periods you want for the indicator
        self.periods = periods
        # Data per request
        self.n = n

    def momentum(self, close_data):
        """ Momentum indicator construction"""
        return close_data[-1]/close_data[-1-self.periods]-1.0

    def sma(self, close_data):
        """ Custom standard moving average indicator construction"""
        return close_data[-1]/close_data[self.n-self.periods:self.n].mean()-1.0

    def bollinger_bands(self, close_data):
        """ Bollinger Bands indicator construction"""
        mean = close_data[-1]-close_data[self.n-self.periods:self.n].mean()
        return mean/(2*close_data[self.n-self.periods:self.n].std())


class OperationInvalidException(Exception):
    """Exception class for the module"""
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
