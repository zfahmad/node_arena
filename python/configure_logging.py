import logging
import sys


def configure_logging(log_file: str = "app.log") -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove any existing handlers (important for multiprocessing)
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(processName)s %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    # root.addHandler(console_handler)
