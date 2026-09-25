"""Application entry point: ``python -m app.main``.

The Flask app is created here and served on 127.0.0.1:5000. The model is loaded
lazily on the first request that needs it, never at import time, and never
trained here.
"""

from __future__ import annotations

from . import create_app

app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
