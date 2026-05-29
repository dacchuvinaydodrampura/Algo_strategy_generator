import os

# Gunicorn configuration file for Render deployment
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
