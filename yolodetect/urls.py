from django.urls import path
from . import views

urlpatterns = [
    path('video_feed/', views.video_feed, name='video_feed'),
    path('', views.landing, name='landing'),
    path('contact/', views.contact, name='contact'),    # Contact form URL
]