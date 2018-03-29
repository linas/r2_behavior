#!/usr/bin/env python
import logging

from transitions.extensions import HierarchicalMachine
from transitions.extensions.nesting import NestedState

import rospy
import os
import re
import random
from std_msgs.msg import String, Bool, Float32
from blender_api_msgs.msg import Target, SomaState
from blender_api_msgs.srv import SetParam
import time
import performances.srv as srv
from performances.msg import Event as PerformanceEvent
import subprocess
import threading
import dynamic_reconfigure.client
from hr_msgs.msg import ChatMessage
from dynamic_reconfigure.server import Server
from performances.nodes import pause
from r2_behavior.cfg import AttentionConfig, AnimationConfig, StatesConfig, StateConfig

logger = logging.getLogger('hr.behavior')

# High level hierarchical state machine
STATES = [
    'idle',
    {'name': 'interacting', 'initial': 'interested', 'children': [
        'bored',
        'interested',
        'listening',
        'speaking',
        'thinking'
    ]},
    {'name': 'presenting', 'initial': 'waiting', 'children': [
        'performing',
        'waiting',
    ]},
    'analysis'
]
# Otional state params:
# States to serve separate attention settings
ATTENTION = ['interacting_bored', 'interacting_interested', 'interacting_listening', 'interacting_speaking',
             'interacting_thinking', 'presenting_waiting']
# States to serve separate animation settings
ANIMATIONS = ATTENTION

# Defines transitions:
# name, from states, to_states, [condition, unless]

TRANSITIONS = [
    ['start_interacting', '*', 'interacting'],
    ['start_presentation', '*', 'presenting'],
    ['run_timeline', 'presenting_waiting', 'presenting_performing'],
    ['timeline_paused', 'presenting_performing', 'presenting_waiting'],
    ['timeline_finished', 'presenting_performing', 'presenting_waiting'],  # Need to consider if should sswitch back to interacting
    ['start_analysis', 'interacting', 'analysis'],
    ['stop_analysis', 'analysis', 'interacting'],
    ['boring', 'interacting_interested', 'interacting_bored'],
    ['someone_come_in', 'interacting_bored', 'interacting_interested'], # Visual input, robot wakes up
    ['speech_start', ['interacting_interested', 'interacting_speaking'], 'interacting_listening'],  # If speech detected but not yet transcribed
    ['speech_finished', ['interacting_bored', 'interacting_interested', 'interacting_listening'],
        'interacting_thinking', 'need_to_think'], # If thinking is needed return true
    ['speech_finished', ['interacting_bored', 'interacting_interested', 'interacting_listening'],
     'interacting_speaking', None,  'need_to_think'],  # Straight answer without thinking (using unless)
    ['finish_talking', 'interacting_speaking', 'interacting_listening'],
    ['finish_listening', 'interacting_listening', 'interacting_interested'],
]



class InteractiveState(NestedState):
    def __init__(self, name, on_enter=None, on_exit=None, ignore_invalid_triggers=None, parent=None, initial=None):
        NestedState.__init__(self, name, on_enter, on_exit, ignore_invalid_triggers, parent, initial)
        # create dynamic reconfigure server
        self.attention_config = {}
        self.animations_config = {}
        self.state_config = {}
        node_name = name if parent is None else "{}_{}".format(parent.name, name)
        if node_name in ATTENTION:
            self.attention_server = Server(AttentionConfig, self.attention_callback,
                                           namespace="{}_attention".format(node_name))
        if node_name in ANIMATIONS:
            self.animation_server = Server(AnimationConfig, self.animations_callback,
                                         namespace="{}_animation".format(node_name))
        self.state_server = Server(StateConfig, self.config_callback, namespace='{}_settings'.format(node_name))

    def attention_callback(self, config, level):
        self.attention_config = config
        return config

    def animations_callback(self, config, level):
        self.animations_config = config
        return config

    def config_callback(self, config, level):
        self.state_config = config
        return config



class Robot(HierarchicalMachine):

    state_cls = InteractiveState

    def __init__(self):
        # Wait for service to set initial params
        rospy.wait_for_service('attention/set_parameters')
        rospy.wait_for_service('animation/set_parameters')
        print('Wait finished')
        self.clients = {
            'attention': dynamic_reconfigure.client.Client('attention', timeout=0.1),
            'animation': dynamic_reconfigure.client.Client('animation', timeout=0.1),
        }
        # Current state config
        self.state_config = None
        self.config = None
        self.starting = True
        # ROS Topics and services
        self.robot_name = rospy.get_param('/robot_name')
        # ROS publishers
        self.topics = {
            'running_performance': rospy.Publisher('/running_performance', String, queue_size=1),
            'states_pub': rospy.Publisher("/state", String, queue_size=5),
            'chatbot_speech': rospy.Publisher('/{}/chatbot_speech'.format(self.robot_name), ChatMessage, queue_size=10),
            'soma_pub': rospy.Publisher('/blender_api/set_soma_state', SomaState, queue_size=10),
        }
        # ROS Subscribers
        self.subscribers = {
            'speech': rospy.Subscriber('/{}/speech'.format(self.robot_name), ChatMessage, self.speech_cb),
            'running_performance': rospy.Subscriber('/performances/running_performance', String, self.performance_cb),
            'performance_events': rospy.Subscriber('/performances/events', PerformanceEvent,
                                                        self.performance_events_cb),
            'speech_events': rospy.Subscriber('/{}/speech_events'.format(self.robot_name), String,
                                                   self.speech_events_cb),
            'chat_events': rospy.Subscriber('/{}/chat_events'.format(self.robot_name), String,
                                              self.chat_events_cb),
        }
        # ROS Services
        # Wait for all services to become available
        rospy.wait_for_service('/blender_api/set_param')
        rospy.wait_for_service('/performances/current')
        # All services required.
        self.services = {
            'performance_runner': rospy.ServiceProxy('/performances/run_full_performance', srv.RunByName),
            'blender_param': rospy.ServiceProxy('/blender_api/set_param', SetParam),
        }
        # Configure clients
        self.clients = {
            'attention': dynamic_reconfigure.client.Client('attention', timeout=0.1),
            'animation': dynamic_reconfigure.client.Client('animation', timeout=0.1),
        }
        # robot properties
        self.props = {
            'disable_attention': None,
            'disable_animations': None,
            'disable_blinking': None,
            'disable_saccades': None,
            'disable_keepalive': None,
        }

        self._before_presentation = ''
        self._current_performance = None
        # State server mostly used for checking current states settings
        self.state_server = Server(StateConfig, self.state_config_callback, namespace='current_state_settings')
        # Machine starts
        HierarchicalMachine.__init__(self, states=STATES, transitions=TRANSITIONS, initial='idle',
                                          ignore_invalid_triggers=True, after_state_change=self.state_changed)
        # Main param server
        self.server = Server(StatesConfig, self.config_callback)
        self._listen_timer = None

    # Calls after each state change to apply new configs
    def state_changed(self):
        rospy.set_param('/current_state', self.state)
        # State object
        state = self.get_state(self.state)
        print(self.state)
        if state.attention_config:
            self.clients['attention'].update_configuration(state.attention_config)
        if state.animations_config:
            self.clients['animation'].update_configuration(state.animations_config)
        # Ap
        # Aply general behavior for states
        if state.state_config:
            self.state_server.update_configuration(state.state_config)

    def state_config_callback(self, config, level):
        # Apply properties
        self.disable_animations = config.disable_animations
        self.disable_attention = config.disable_attention
        self.disable_blinking = config.disable_blinking
        self.disable_keepalive = config.disable_keepalive
        self.disable_saccades = config.disable_saccades
        self.state_config = config
        return config


    #  ROS Callback methods
    def config_callback(self, config, level):
        self.config = config
        # Set correct init state on loading. Will set the parameters as well
        if self.starting:
            self.starting = False
            if self.config['init_state'] == 'interacting':
                self.start_interacting()
            if self.config['init_state'] == 'presentation':
                self.start_presentation()
        return config

    # Handles all speech inputs
    def speech_cb(self, msg):
        try:
            speech = str(msg.utterance).lower()
            # Check if performance is not waiting for same keyword to continue in timeline
            if self.is_presenting(allow_substates=True) == 'presenting' and self.config['chat_during_performance']:
                keywords = rospy.get_param('/performances/keywords_listening', False)
                # Don't pass the keywords if pause node waits for same keyword (i.e resume performance).
                if keywords and pause.event_matched(keywords, msg.utterance):
                    return
            # Allow trigger performances by keywords
            if self.state_config.performances_by_keyword:
                performances = self.find_performance_by_speech(speech)
                # Split between performances for general modes and analysis
                analysis_performances = [p for p in performances if ('shared/analysis' in p or 'robot/analysis' in p)]
                for a in analysis_performances:
                    performances.remove(a)
                if performances and self.state != 'analysis':
                    self.services['performance_runner'](random.choice(performances))
                elif analysis_performances:
                    self.services['performance_runner'](random.choice(analysis_performances))

            # If chat is not enabled for specific state ignore it
            if not self.state_config.chat_enabled:
                return
            # Need to respond to speech
            # Check if state allows chat
            self.speech_finished()
            self.topics['chatbot_speech'].publish(msg)
        except Exception as e:
            logger.error(e)
            self.topics['chatbot_speech'].publish(msg)

    def speech_events_cb(self, msg):
        if msg.data == 'start':
            # Speech finished, Robot starts talklign after
            self.speech_finished()
        if msg.data == 'stop':
            # Talking finished, robot starts listening
            self.finish_talking()

    def chat_events_cb(self, msg):
        if msg.data == 'speech_start':
            # Speech finished, Robot starts talklign after
            self.speech_start()

    def find_performance_by_speech(self, speech):
        """ Finds performances which one of keyword matches"""
        performances = []
        for performance, keywords in self.get_keywords().items():
            if self.performance_keyword_match(keywords, speech):
                performances.append(performance)
        return performances

    @staticmethod
    def performance_keyword_match(keywords, input):
        for keyword in keywords:
            if not keyword:
                continue
            # Currently only simple matching
            if re.search(r"\b{}\b".format(keyword), input, flags=re.IGNORECASE):
                return True
        return False

    def get_keywords(self, performances=None, keywords=None, path='.'):
        if performances is None:
            performances = rospy.get_param(os.path.join('/', self.robot_name, 'webui/performances'))
            keywords = {}

        if 'properties' in performances and 'keywords' in performances['properties']:
            keywords[path] = performances['properties']['keywords']

        for key, value in performances.items():
            if key != 'properties':
                self.get_keywords(performances[key], keywords, os.path.join(path, key).strip('./'))

        return keywords

    def performance_cb(self, msg):
        try:
            # Track current performance
            self._current_performance = msg.data
            if msg.data == "null":
                if self._before_presentation:
                    if self._before_presentation.startswith("interacting"):
                        self.start_interacting()
                    # Might be usefull if we do idle out
                    elif self._before_presentation.startswith("idle"):
                        self.to_idle()
                    else:
                        self.timeline_finished()
                else:
                    # Stay in presentation
                    self.timeline_finished()
            else:
                # New performance loaded
                print("Starting presentation from {}".format(self.state))
                self._before_presentation = self.state
                self.start_presentation()
        except Exception as e:
            logger.error(e)

    def performance_events_cb(self, msg):
        if msg.event in ['resume', 'running']:
            self.run_timeline()
        elif msg.event in ['paused']:
            self.timeline_paused()
        else:
            self.timeline_finished()

    def need_to_think(self):
        # Currently no thinking needed to respond
        return False

    def on_enter_interacting_listening(self):

        self._listen_timer = threading.Timer(self.config.listening_time, self.finish_listening).start()

    def on_exit_interacting_listening(self):
        if self._listen_timer:
            try:
                self._listen_timer.cancel()
                self._listen_timer = False
            except:
                pass
    @property
    def disable_attention(self):
        return self.props['disable_attention']

    @disable_attention.setter
    def disable_attention(self, val):
        if self.props['disable_attention'] != val:
            self.props['disable_attention'] = val
            try:
                self.clients['attention'].update_configuration({'enable_flag': val})
            except Exception as e:
                logger.errot(e)

    @property
    def disable_animations(self):
        return self.props['disable_animations']

    @disable_animations.setter
    def disable_animations(self, val):
        if self.props['disable_animations'] != val:
            self.props['disable_animations'] = val
            try:
                self.clients['animation'].update_configuration({'enable_flag': val})
            except Exception as e:
                logger.error(e)
    @property
    def disable_blinking(self):
        return self.props['disable_blinking']

    @disable_blinking.setter
    def disable_blinking(self, val):
        if self.props['disable_blinking'] != val:
            try:
                self.services['blender_param']("bpy.data.scenes[\"Scene\"].actuators.ACT_blink_randomly.HEAD_PARAM_enabled",
                                   str(not val))
            except Exception as e:
                logger.error(e)

    @property
    def disable_saccades(self):
        return self.props['disable_saccades']

    @disable_saccades.setter
    def disable_saccades(self, val):
        if self.props['disable_saccades'] != val:
            try:
                self.services['blender_param']("bpy.data.scenes[\"Scene\"].actuators.ACT_saccade.HEAD_PARAM_enabled",
                                   str(not val))
            except Exception as e:
                logger.error(e)

    @property
    def disable_keepalive(self):
        return self.props['disable_keepalive']

    @disable_keepalive.setter
    def disable_keepalive(self, val):
        if self.props['disable_keepalive'] != val:
            self.props['disable_keepalive'] = val
            try:
                magnitude = 0.0 if val else 1.0
                self.topics['soma_pub'].publish(self._get_soma('normal', magnitude))
                self.topics['soma_pub'].publish(self._get_soma('breathing', magnitude))
                self.topics['soma_pub'].publish(self._get_soma('normal-saccades', magnitude))
            except Exception as e:
                logger.error(e)

    @staticmethod
    def _get_soma(name, magnitude):
        """ Speech"""
        s = SomaState()
        s.name = name
        s.ease_in.secs = 0
        s.ease_in.nsecs = 0.1 * 1000000000
        s.magnitude = magnitude
        s.rate = 1
        return s


if __name__ == "__main__":
    rospy.init_node("hrx_graph")
    robot = Robot()
    rospy.spin()