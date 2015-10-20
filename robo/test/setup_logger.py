import os
import json
import logging.config

# Set default logging directly
if 'loggingInitialized' not in locals():
    loggingInitialized = True

    path = 'logging.json'
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)