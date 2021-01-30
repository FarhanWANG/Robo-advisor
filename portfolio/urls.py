from django.urls import path
from . import views
from .views import StocksListView, StocksCreateView

urlpatterns = [
    path('create/', StocksCreateView.as_view(), name='portfolio-create'),
    path('', StocksListView.as_view(), name='portfolio-home'),
    path('risk/', views.risk, name='portfolio-risk'),
    path('welcome/', views.welcome, name='portfolio-welcome'),
]
