from django.urls import path
from .views import PredictDRView, PredictionListView

urlpatterns = [
    path('predict/', PredictDRView.as_view(), name='predict-dr'),
    path('predictions/', PredictionListView.as_view(), name='prediction-list'),
]