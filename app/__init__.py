"""Flask application factory."""

from __future__ import annotations

from flask import Flask, jsonify

from src.email_spam.security import MAX_UPLOAD_BYTES

from .routes import api

MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + (64 * 1024)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    app.config["JSON_SORT_KEYS"] = False
    app.register_blueprint(api)

    @app.errorhandler(413)
    def request_too_large(_error):
        return (
            jsonify({"error": f"The request body exceeds the {MAX_REQUEST_BYTES // 1024} KB limit."}),
            413,
        )

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return jsonify({"error": "Method not allowed for this endpoint."}), 405

    return app
