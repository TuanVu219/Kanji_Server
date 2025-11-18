from django.urls import path
from django.contrib import admin
from core import views as core_views
from rest_framework import routers
from django.urls import path, include

router = routers.DefaultRouter()

urlpatterns = router.urls

urlpatterns += [
    path('admin/', admin.site.urls),
    path('api/', include('core.urls')),
    ]