from __future__ import unicode_literals

from django.db import models
from django.urls import reverse
from django.contrib.auth.models import User
from yahoo_fin.stock_info import get_data
import pandas as pd
import numpy as np
import tushare as ts
import datetime
from functools import reduce
import scipy.optimize as solver
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Stocks(models.Model):
    symbol = models.CharField(max_length=200)
    name = models.CharField(max_length=200)
    buying_price = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    amount = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    coin = models.BooleanField(default=False)

    def __str__(self):
        return self.symbol

    def get_absolute_url(self):
        return reverse('portfolio-home')

    def save(self, *args, **kwargs):
        # convert all symbol to uppercase
        self.symbol = self.symbol.upper()
        return super().save(*args, **kwargs)




class generateOptimalPortfolio:
    ListofStocks = Stocks.objects.all()
    list = []
    for i in ListofStocks:
        list.append(i.symbol)
    def __init__(self):
        self.CC_List = self.list

    def getPricesFromYahoo(self, CC_list):
        hisPrices = {}
        for cc in CC_list:
            hisPrices[cc] = get_data(cc, '1/1/2019', datetime.date.today(), True, '1wk')
        Prices = pd.DataFrame(index=np.arange(hisPrices['BTC-USD'].shape[0]), columns=CC_list)
        Prices.set_index(hisPrices['BTC-USD'].index, inplace=True)
        for x in hisPrices:
            Prices[x] = hisPrices[x].adjclose
        return Prices

    def getReturnsFromPrices(self, Prices):
        returns = np.log(Prices / Prices.shift(1)).fillna(value=0)
        return returns

    def getLIBORFromTs(self, returns):
        ts.set_token('d22a3bc6484b9fd9618e97dfd6249e49d7146267e1167bdc375de7c1')
        pro = ts.pro_api()
        df = pro.libor(curr_type='USD', start_date='20190101', end_date='20201001')
        LIBOR = returns.copy(deep=True).drop(columns=self.CC_List)
        LIBOR['LIBOR'] = 0.0
        for i in LIBOR.index:
            d = str(i).split(" ")[0].split("-")
            date = d[0] + d[1] + d[2]
            LIBOR.loc[i]['LIBOR'] = self.findLIBOR(df, date)
        # change 0.0 to the adjacent value
        for i in range(1, len(LIBOR)):
            if LIBOR.iloc[i]['LIBOR'] == 0.0:
                LIBOR.iloc[i]['LIBOR'] = LIBOR.iloc[i - 1]['LIBOR']
        return LIBOR

    def findLIBOR(self, df, date):
        for i in df.index:
            if date == df.loc[i]['date']:
                return df.loc[i]['1w']
                break
        return 0

    def getMean(self, returns, start, end):
        return returns.iloc[start:end].mean()

    def getCov(self, returns, start, end):
        return returns.iloc[start:end].cov()

    def getRisk_free(self, LIBOR, start, end):
        return LIBOR.iloc[start:end]['LIBOR'].mean() / 5200

    def std(self, w, cov):
        return np.sqrt(reduce(np.dot, [w, cov, w.T]))

    def getOptimalPortfolio(self, returns, LIBOR, start, end):
        # general initialization
        mean = self.getMean(returns, start, end)
        cov = self.getCov(returns, start, end)
        risk_free = self.getRisk_free(LIBOR, start, end)
        max_sharpe = 0.0
        weights = []
        # Initialization of solver.minimize
        return_change_range = np.arange(min(mean), max(mean), (max(mean) - min(mean)) / 100)
        w0 = np.array([1 / len(self.CC_List) for x in range(len(self.CC_List))])  # equal weights
        # bounds = tuple((0,1) for x in range(len(Top10CC))) #boundary of each weight. (0,1)means no short selling
        for i in return_change_range:
            constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1.0},
                           {'type': 'eq', 'fun': lambda x: sum(x * mean) - i}]
            outcome = solver.minimize(self.std, args=cov, x0=w0, constraints=constraints)  # args is passed to std()
            if (i - risk_free) / outcome.fun > max_sharpe:
                max_sharpe = (i - risk_free) / outcome.fun
                weights.append(outcome.x)
        optimalReturn = sum(weights[-1] * mean)
        optimalWeights = weights[-1]
        return [max_sharpe, optimalReturn, self.CC_List, optimalWeights]

    def main(self):
        prices = self.getPricesFromYahoo(self.CC_List)
        returns = self.getReturnsFromPrices(prices)
        LIBOR = self.getLIBORFromTs(returns)
        return self.getOptimalPortfolio(returns, LIBOR, 0, len(returns))