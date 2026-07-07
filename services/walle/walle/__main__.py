"""walle entry point — the nightly lifecycle + reconciliation cleanup job.

Run as ``python -m walle`` by a Cloud Run job on a nightly Cloud Scheduler
trigger. Exits non-zero on failure so the job execution is marked failed and
retried.
"""

import json
import logging
import sys

from walle.cleanup import run


class StructuredLogHandler(logging.Handler):
    """Emit JSON log lines for Cloud Logging (severity + message)."""

    def emit(self, record: logging.LogRecord) -> None:
        entry = {"severity": record.levelname, "message": record.getMessage()}
        sys.stdout.write(json.dumps(entry) + "\n")


def _configure_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(StructuredLogHandler())
    root.setLevel(logging.INFO)


def main() -> None:
    _configure_logging()
    try:
        run()
    except Exception:
        logging.getLogger(__name__).exception("walle cleanup job failed")
        raise


if __name__ == "__main__":
    main()
