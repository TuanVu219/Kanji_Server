from django.urls import path
from . import views

urlpatterns = [
    path('recognise/', views.recognize_strokes),
    path('draw/', views.draw_view, name='draw'),
    path('check_device/', views.check_device, name='check_device'),
]
