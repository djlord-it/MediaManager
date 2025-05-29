from django.urls import path
from . import views

app_name = 'uploader'

urlpatterns = [
    path('', views.video_list, name='video_list'),
    path('upload/', views.video_upload, name='video_upload'),
    path('video/<int:pk>/', views.video_detail, name='video_detail'),
    path('video/<int:pk>/delete/', views.video_delete, name='video_delete'),
    path('video/<int:pk>/delete/ajax/', views.video_delete_ajax, name='video_delete_ajax'),
    path('category/<str:category>/', views.category_view, name='category_view'),
] 