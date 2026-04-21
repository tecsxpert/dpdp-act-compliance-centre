#As of day 1 (from AI dev 1) , app.py is just a boilerplate which im going to use the variable later in those blueprints present in routes folder
#Other devs are requested to use the same varible name for their task while coding which i imported.

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Importing blueprints
from routes.describe import describe_bp
from routes.recommend import recommend_bp
from routes.categorise import categorise_bp
from routes.generate_report import generate_report_bp

def create_app():
    app = Flask(__name__)

    # Setup basic rate limiting
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["30 per minute"]
    )

    # Registering all Blueprints 
    app.register_blueprint(describe_bp)
    app.register_blueprint(recommend_bp)
    app.register_blueprint(categorise_bp)
    app.register_blueprint(generate_report_bp)

    @app.route('/health', methods=['GET'])
    def health_check():
        return {"status": "ok", "service": "ai-service"}, 200

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)