# Copyright (c) 2013-2018 Hanson Robotics, Ltd, all rights reserved 
import yaml
import json
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

def parse(filename):
    with open(os.path.join(current_dir, filename + '.yaml'), 'r') as stream:
        try:
            return json.dumps(yaml.load(stream))
        except yaml.YAMLError as exc:
            return False

def load(filename):
    with open(os.path.join(current_dir, filename + '.yaml'), 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            return False
