import os
import json
import logging.config

# Set default logging directly
def setup_logger(path = 'logging.json'):
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)