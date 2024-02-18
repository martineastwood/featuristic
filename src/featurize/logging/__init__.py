import logging
import logging.config

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(name)s.%(module)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        # 'file_handler': {
        #     'level': 'INFO',
        #     'filename': '/tmp/mylogfile.log',
        #     'class': 'logging.FileHandler',
        #     'formatter': 'standard'
        # }
    },
    # 'loggers': {
    #     '': {
    #         'handlers': ['file_handler'],
    #         'level': 'INFO',
    #         'propagate': True
    #     },
    # }
}

# logging.config.dictConfig(LOGGING_CONFIG)

logging.basicConfig()
