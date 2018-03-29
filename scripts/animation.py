#!/usr/bin/env python

import time
import random
import json
import numpy as np

import rospy
from blender_api_msgs.msg import EmotionState, SetGesture
from r2_behavior.cfg import AnimationConfig
from dynamic_reconfigure.server import Server


class Utils:
    def __init__(self):
        pass

    @staticmethod
    # Picks random expressions gesture from multidimensional array which has probabilities in first column.
    # Returns row of array
    def random(entries):
        probabilities_sum = sum([e['probability'] for e in entries])
        probabilities = [e['probability'] / probabilities_sum for e in entries]
        i = np.random.choice(len(probabilities), p=probabilities)
        return entries[i]


class Expressions:
    def __init__(self, config):
        self.config = config
        self.next_expression_time = 0
        self.smile = 0
        # Future time
        self.smile_started = time.time()*2

    def show_expression(self):
        t = time.time()

        if t > self.next_expression_time:
            self.smile_started = time.time()*2
            self.next_expression_time = t + random.uniform(self.config['time_between_expressions_min'],
                                                           self.config['time_between_expressions_max'])
            try:
                e = Utils.random(json.loads(self.config['expressions']))
            except:
                return
            expression = EmotionState()
            expression.name = e['name']
            expression.magnitude = max(0, min(1, random.uniform(float(e['magnitude_min']), float(e['magnitude_max'])) *
                                              self.config['expression_magnitude']))
            expression.duration = rospy.Duration(
                max(1, random.uniform(float(e['duration_min']), float(e['duration_max'])) *
                    self.config['expression_duration']))
            return expression
        return None


class Gestures:
    def __init__(self, config):
        self.config = config
        self.next_gesture_time = 0

    def show_gesture(self):
        t = time.time()
        if t > self.next_gesture_time:
            self.next_gesture_time = t + random.uniform(self.config['time_between_gestures_min'],
                                                       self.config['time_between_gestures_max'])
            try:
                g = Utils.random(json.loads(self.config['gestures']))
            except:
                return
            gesture = SetGesture()
            gesture.name = g['name']
            gesture.magnitude = max(0, min(3, random.uniform(float(g['magnitude_min']), float(g['magnitude_max'])) *
                                           self.config['gesture_magnitude']))
            gesture.speed = min(3, random.uniform(float(g['speed_min']), float(g['speed_max'])) *
                                self.config['gesture_speed'])
            return gesture
        return None

class Animations:
    def __init__(self):
        self.config = {}
        self.init = True
        self.gestures = None
        self.expressions = None
        self.config = None
        self.animations = Server(AnimationConfig, self.config_callback)

    def config_callback(self, config, level ):
        self.config = config
        if self.init:
            self.gestures = Gestures(config)
            self.expressions = Expressions(config)
            self.init = False
        else:
            self.gestures.config = config
            self.expressions.config = config

        return config

    def timer(self):
        if self.init:
            return
        if self.config.enable_flag:
            return
        # Shows gestures based on timings setr in configs
        self.expressions.show_expression()
        self.gestures.show_gesture()


if __name__ == "__main__":
    rospy.init_node("animations")
    rate = rospy.Rate(10)
    animations = Animations()
    while not rospy.is_shutdown():
        animations.timer()
        rate.sleep()

