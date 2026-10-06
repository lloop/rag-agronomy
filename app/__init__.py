from flask import Flask

def create_app():
    app = Flask(__name__)

    # Register your existing api_bp and views_bp
    from app.routes.api import api_bp
    from app.routes.views import views_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(views_bp)

    return app