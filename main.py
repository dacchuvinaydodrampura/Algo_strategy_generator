# Entry point redirect for Render's default launch command
from a2wsgi import ASGIMiddleware
from app.server import app as asgi_app

# Expose as WSGI app to support gunicorn's default sync worker
app = ASGIMiddleware(asgi_app)

