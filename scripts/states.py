#!/usr/bin/env python
import logging
from transitions.extensions import HierarchicalMachine
from transitions.extensions.nesting import NestedState
import rospy
import rospkg
import os
import re
import random
from std_msgs.msg import String, Bool
from blender_api_msgs.msg import Target, SomaState
from blender_api_msgs.srv import SetParam
import time
import performances.srv as srv
from performances.msg import Event
import subprocess
import threading
import dynamic_reconfigure.client
from hr_msgs.msg import ChatMessage
from dynamic_reconfigure.server import Server
from performances.nodes import pause
from r2_behavior.cfg import AttentionConfig, AnimationConfig

logger = logging.getLogger('hr.performance.wholeshow')

# High level hierarchical state machine
STATES = [
    'idle',
    {'name': 'interacting', 'initial': 'bored', 'children':[
        'bored',
        'listening',
        'speaking',
        'thinking'
    ]},
    {'name': 'presenting', 'initial': 'listening', 'children':[
        'performing',
        'waiting',
    ]},
    'analysis'
]


class InteractiveState(NestedState):
    def __init__(self, name, on_enter=None, on_exit=None, ignore_invalid_triggers=None, parent=None, initial=None):
        NestedState.__init__(self, name, on_enter, on_exit, ignore_invalid_triggers, parent, initial)
        # create dynamic reconfigure server
        self.attention_config = {}
        self.animation_config = {}
        self.state_config = {}
        node_name = name if parent is None else "{}_{}".format(parent.name, name)
        self.attention_server = Server(AttentionConfig, self.attention_callback, namespace="{}_atteantion".format(node_name))
        self.animation_server = Server(AnimationConfig, self.animation_callback,
                                         namespace="{}_animkation".format(node_name))
    def attention_callback(self, config, level):
        self.attention_config = config
        # TODO make sure we update actual behaviors
        return config

    def animation_callback(self, config, level):
        self.animation_config = config
        # TODO make sure we update actual behaviors
        return config


class Robot(HierarchicalMachine):
    state_cls = InteractiveState
    def __init__(self):
        HierarchicalMachine.__init__(self, states=STATES, initial='sleep')




if __name__ == "__main__":
    rospy.init_node("hrx")
    robot = Robot()
    rospy.spin()
