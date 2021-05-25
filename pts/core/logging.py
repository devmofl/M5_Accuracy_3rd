import os
import socket
from datetime import datetime
import logging
from pathlib import Path

def get_log_path(log_dir, log_comment='temp', trial='t0', mkdir=True):    
    if log_comment=='':
        log_comment='temp'
    
    base_path = os.path.join('logs', log_dir, log_comment)
    trial_path = trial

    full_log_path = f"{base_path}/{trial_path}"

    path = Path(full_log_path)    
    if mkdir==True:
        if not path.exists():
            path.mkdir(parents=True)

    return full_log_path, base_path, trial_path

def set_logger(text_log_path, text_log_file = 'log.txt', level = logging.INFO):
    logger = logging.getLogger("mofl")
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s%(name)18s%(levelname)10s\t%(message)s')

    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    logger.addHandler(stream_hander)

    log_file = f"{text_log_path}/{text_log_file}"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    logger.info(f"Logging to {log_file}...")