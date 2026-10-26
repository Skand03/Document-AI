import os
from flask import Flask, jsonify
from database.db import init_db
from routes.id_routes import id_blueprint


def create_app(config: dict = None) -> Flask:
    """Application factory for Document AI Flask microservice."""
    app = Flask(__name__)

    # Default configurations
    base_dir = os.path.dirname(os.path.abspath(__file__))
    storage_dir = os.path.join(base_dir, "storage")
    thumbnails_dir = os.path.join(storage_dir, "thumbnails")
    db_path = os.path.join(storage_dir, "records.db")

    os.makedirs(thumbnails_dir, exist_ok=True)

    app.config.update(
        DATABASE_PATH=os.environ.get("DATABASE_PATH", db_path),
        THUMBNAIL_DIR=os.environ.get("THUMBNAIL_DIR", thumbnails_dir),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB max upload
    )

    if config:
        app.config.update(config)

    # Initialize Database Schema
    init_db(app.config["DATABASE_PATH"])

    # Register Blueprints
    app.register_blueprint(id_blueprint)

    # Standard error handlers for clean JSON responses
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request"}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint or resource not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return jsonify({"error": "File size exceeds maximum allowed limit (16MB)"}), 413

    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify({"error": "An internal server error occurred"}), 500

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "healthy", "service": "document-ai-backend"}), 200

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
