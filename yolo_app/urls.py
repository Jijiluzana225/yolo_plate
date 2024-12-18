from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.detect_objects, name='detect_objects'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('stream/', views.stream_page, name='stream_page'),  # Page with the video feed
    path('compare_images/', views.compare_images, name='compare_images'),
  
]