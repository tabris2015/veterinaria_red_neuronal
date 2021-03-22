from django.urls import path
from django.http import StreamingHttpResponse
from . import views

from . import services

# app_name = 'detection'
urlpatterns = [
    path('', views.index, name='Detection'),
    path('image/', views.receive_image, name='receiveImage'),
    path('feedback/', views.receive_feedback, name='receiveFeedback')
]
