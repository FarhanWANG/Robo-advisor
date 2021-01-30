from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, CreateView
from .models import Stocks
from .models import generateOptimalPortfolio
from django.http import HttpResponse
import yfinance as yf
import numpy as np

from .deep_model.predict import stock_pred
import json

class StocksListView(ListView):
    # stocks_list.html 
    model = Stocks

    def get_queryset(self):
        # only shows owner's stocks
        return self.model.objects.filter(owner=self.request.user)

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)
        tickers = self.model.objects.filter(owner=self.request.user).values("symbol")
        # boolen field:if stock then false
        digit_coins = self.model.objects.filter(owner=self.request.user).values("coin")

        prices = []
        intput_tensor=[]
        labels = []
        whole_lables = []
        # today's price of stock
        for ticker, coin in zip(tickers, digit_coins):
            # today's pirce
            symbol = ticker['symbol']
            ticker = yf.Ticker(symbol)
            price = format(ticker.history(period="1d")["Close"][0], '.2f')
            prices.append(price)
            whole_lables.append(symbol)


            if not coin['coin']:
                # 90days input data for deep model
                labels.append(symbol)
                prices_90 = yf.download(symbol, period ="6mo")[-90:]
                columnsTitles = ['Adj Close', 'Open', 'Close', 'High', 'Low', 'Volume']
                prices_90 = prices_90.reindex(columns=columnsTitles)
                prices_90 = prices_90.to_numpy()
                intput_tensor.append(prices_90)
            
        # append today's price to the original context
        

        # check if the user have created the portfolio or not
        if len(prices) == 0:
            context['exist'] = False
        else:
            context['exist'] = True
            res = generateOptimalPortfolio()
            result = res.main()
            maximalReturn = str(np.around(result[0], decimals=4))
            sharpeRatio = str(np.around(result[1], decimals=4))
            weights = np.around(result[3], decimals=4)

            # use wavenet to predict the stock price of the next 3 days
            pred_stock = stock_pred(intput_tensor)
            data = {'labels':labels, 'hist_prices':pred_stock}
            context['stockdata'] = json.dumps(data)
            wholedata = {'labels':whole_lables, 'weights':list(weights)}
            context['wholedata'] = json.dumps(wholedata)
    
            context['maximal'] = maximalReturn
            context['sharpe'] = sharpeRatio
            context['names'] = result[2]
            context['object_list'] = list(zip(prices, weights, context['object_list']))
            
                
        return context



class StocksCreateView(LoginRequiredMixin, CreateView):
    # stocks_form.html 
    model = Stocks
    fields = ['symbol', 'amount', 'coin']

    def form_valid(self, form):
        # https://aroussi.com/post/python-yahoo-finance
        # retrieve data from yahoo finance API
        ticker = yf.Ticker(form.instance.symbol)
        form.instance.name = ticker.info['shortName']
        day_price = ticker.history(period="1d")["Close"][0]
        form.instance.buying_price = format(day_price, '.2f')
        form.instance.owner = self.request.user
        return super().form_valid(form)

def risk(request):
    return render(request, 'portfolio/risk.html')

def welcome(request):
    return render(request, 'portfolio/welcome.html')






    