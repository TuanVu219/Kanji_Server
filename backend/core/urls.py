from django.urls import path
from . import views
from .views import gemini_translate
urlpatterns = [
    path('recognise/', views.recognize_strokes),
    path('draw/', views.draw_view, name='draw'),
    path('check_device/', views.check_device, name='check_device'),
    path("gemini/translate/", gemini_translate),

]
