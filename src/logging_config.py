import logging
from logging.config import dictConfig

from src.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = settings.log_level.upper()

    if settings.log_json:
        format_string = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        format_string = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": format_string,
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": level,
                }
            },
            "root": {
                "handlers": ["default"],
                "level": level,
            },
        }
    )


logger = logging.getLogger(__name__)
