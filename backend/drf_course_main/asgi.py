import os
import django
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from core import routing
from core.utils import get_effnet_model

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "drf_course_main.settings")

# === Load model 1 lần khi server start ===
# Giảm latency cho mọi kết nối WS
django.setup()
get_effnet_model()

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(routing.websocket_urlpatterns)
    ),
})
