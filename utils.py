import logging
import json
import argparse
import os
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--configs_path', dest='configs_path', type=str,
                        help='configureations file path')
    args = parser.parse_args()

    return args


def load_json(path):
    with open(path) as json_file:
        json_obj = json.load(json_file)

    return json_obj


def make_dir_if_not_exist(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_filelines(path: Path) -> int:
    return sum([1 for _ in open(path)])


def load_model(path: Path, model: torch.nn.Module) -> torch.nn.Module:
    model.load_state_dict(torch.load(path))
    return model


def set_logging_config(log_path):
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging_handlers = [stdout_handler]
    logging_level = logging.INFO

    log_path = os.path.join(
        log_path, f"train.log"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    logging_handlers.append(file_handler)

    logging.basicConfig(
        format="%(asctime)s (%(filename)s:%(lineno)d): [%(levelname)s] - %(message)s",
        handlers=logging_handlers,
        level=logging_level,
    )
