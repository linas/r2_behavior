#!/usr/bin/env python
import rospy
import tf
import time
import threading
import math
import operator
import random
import numpy as np
import json
import os
import yaml
import random
from dynamic_reconfigure.server import Server
import dynamic_reconfigure.client
from r2_behavior.cfg import BehaviorConfig
from blender_api_msgs.msg import Target, EmotionState, SetGesture
from std_msgs.msg import String, Float64
from r2_perception.msg import Float32XYZ, CandidateFace, CandidateHand, CandidateSaliency, AudioDirection, MotionVector
from enum import Enum
from hr_msgs.msg import TTS


# in interactive settings with people, the EyeContact machine is used to define specific states for eye contact
# this is purely mechanical, so it follows a very strict control logic; the lookat, awareness and overall state machines control which eyecontact mode is actually used by switching the eyecontact state
class EyeContact(Enum):
    IDLE      = 0  # don't make eye contact
    LEFT_EYE  = 1  # look at left eye
    RIGHT_EYE = 2  # look at right eye
    BOTH_EYES = 3  # switch between both eyes
    TRIANGLE  = 4  # switch between eyes and mouth


# the lookat machine is the lowest level and has the robot look at specific things: saliency, hands, faces
# this is purely mechanical, so it follows a very strict control logic; the awareness and overall state machines control where the robot looks at by switching the lookat state
class LookAt(Enum):
    IDLE      = 0  # look at nothing in particular
    AVOID     = 1  # actively avoid looking at face, hand or saliency
    SALIENCY  = 2  # look at saliency and switch
    HAND      = 3  # look at hand
    ONE_FACE  = 4  # look at single face and make eye contact
    ALL_FACES = 5  # look at all faces, make eye contact and switch
    AUDIENCE  = 6  # look at the audience and switch
    SPEAKER   = 7  # look at the speaker
# param: current face


# awareness: saliency, hands, faces, motion sensors


# the overall state machine controls awareness, lookat and eyecontact and renders different general states the robot is in
# this is what we want to control from the user interface and chatscript and wholeshow and all that, these states are subjective/idealized behavior patterns; the overall state machine "plays the lookat and eyecontact instruments", taking awareness into account
class State(Enum):
    SLEEPING   = 0  # the robot sleeps
    IDLE       = 1  # the robot is idle
    INTERESTED = 2  # the robot is actively idle
    FOCUSED    = 3  # the robot is very interested at something specific
    SPEAKING   = 4  # the robot is speaking to one or more persons or the speaker
    LISTENING  = 5  # the robot is listening to whoever is speaking
    PRESENTING = 6  # the robot is presenting at an audience

    # speaking/listening behavior as per rough video analysis early december 2017


class Behavior:

    def __init__(self):

        self.robot_name = rospy.get_param("/robot_name")
        self.lock = threading.Lock()

        # setup face, hand and saliency structures
        self.faces = {}  # index = cface_id, which should be relatively steady from vision_pipeline
        self.current_face_id = 0  # cface_id of current face
        self.last_face_id = 0  # most recent cface_id of added face
        self.hand = None  # current hand
        self.saliencies = {}  # index = ts, and old saliency vectors will be removed after time
        self.current_saliency_ts = 0  # ts of current saliency vector
        self.current_eye = 0  # current eye (0 = left, 1 = right, 2 = mouth)

        # dynamic parameters
        self.enable_flag = rospy.get_param("enable_flag")
        self.synthesizer_rate = rospy.get_param("synthesizer_rate")

        self.eyecontact = rospy.get_param("eyecontact_state")
        self.lookat = rospy.get_param("lookat_state")
        self.state = rospy.get_param("state")

        self.eyes_counter_min = rospy.get_param("eyes_counter_min")
        self.eyes_counter_max = rospy.get_param("eyes_counter_max")
        self.saliency_counter_min = rospy.get_param("saliency_counter_min")
        self.saliency_counter_max = rospy.get_param("saliency_counter_max")
        self.faces_counter_min = rospy.get_param("faces_counter_min")
        self.faces_counter_max = rospy.get_param("faces_counter_max")
        self.audience_counter_min = rospy.get_param("audience_counter_min")
        self.audience_counter_max = rospy.get_param("audience_counter_max")
        self.keep_time = rospy.get_param("keep_time")

        # counters
        self.eyes_counter = random.randint(self.eyes_counter_min,self.eyes_counter_max)
        self.saliency_counter = random.randint(self.saliency_counter_min,self.saliency_counter_max)
        self.faces_counter = random.randint(self.faces_counter_min,self.faces_counter_max)
        self.audience_counter = random.randint(self.audience_counter_min,self.audience_counter_max)

        # take candidate streams exactly like RealSense Tracker until fusion is better defined and we can rely on combined camera stuff
        rospy.Subscriber('/{}/perception/realsense/cface'.format(self.robot_name), CandidateFace, self.HandleFace)
        rospy.Subscriber('/{}/perception/realsense/chand'.format(self.robot_name), CandidateHand, self.HandleHand)
        rospy.Subscriber('/{}/perception/wideangle/csaliency'.format(self.robot_name), CandidateSaliency, self.HandleSaliency)
        rospy.Subscriber('/{}/perception/acousticmagic/raw_audiodir'.format(self.robot_name), AudioDirection, self.HandleAudioDirection)
        rospy.Subscriber('/{}/perception/motion/raw_motion'.format(self.robot_name), MotionVector, self.HandleMotion)

        rospy.Subscriber('/{}/chat_events'.format(self.robot_name), String, self.HandleChatEvents)
        rospy.Subscriber('/{}/speech_events'.format(self.robot_name), String, self.HandleSpeechEvents)

        self.head_focus_pub = rospy.Publisher('/blender_api/set_face_target', Target, queue_size=1)
        self.gaze_focus_pub = rospy.Publisher('/blender_api/set_gaze_target', Target, queue_size=1)
        self.expressions_pub = rospy.Publisher('/blender_api/set_emotion_state', EmotionState, queue_size=1)
        self.gestures_pub = rospy.Publisher('/blender_api/set_gesture', SetGesture, queue_size=1)

        self.tts_pub = rospy.Publisher('/{}/tts'.format(self.robot_name), TTS, queue_size=1) # for debug messages

        self.hand_events_pub = rospy.Publisher('/hand_events', String, queue_size=1)

        self.tf_listener = tf.TransformListener(False, rospy.Duration(1))

        # start the timer
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.synthesizer_rate),self.HandleTimer)

        # and start the dynamic reconfigure servers
        self.behavior_srv = Server(BehaviorConfig, self.HandleBehaviorConfig)


    def Say(self,text):
        # publish TTS message
        msg = TTS()
        msg.text = text
        msg.lang = 'en-US'
        self.tts_pub.publish(msg)


    def SetHeadFocus(self,pos):
        # publish head focus message
        msg = Target()
        msg.x = pos.x
        msg.y = pos.y
        msg.z = pos.z
        msg.speed = 5.0
        self.head_focus_pub.publish(msg)


    def SelectNextFace(self):
        # switch to the next (or first) face
        if len(self.faces) == 0:
            # there are no faces, so select none
            self.current_face_id = 0
            return
        if self.current_face_id == 0:
            self.current_face_id = self.faces.keys()[0]
        else:
            if self.current_face_id in self.faces:
                next = self.faces.keys().index(self.current_face_id) + 1
                if next >= len(self.faces.keys()):
                    next = 0
            else:
                next = 0
            self.current_face_id = self.faces.keys()[next]


    def SelectNextSaliency(self):
        # switch to the next (or first) saliency vector
        if len(self.saliencies) == 0:
            # there are no saliency vectors, so select none
            self.current_saliency_ts = 0
            return
        if self.current_saliency_ts == 0:
            self.current_saliency_ts = self.saliencies.keys()[0]
        else:
            if self.current_saliency_ts in self.saliencies:
                next = self.saliencies.keys().index(self.current_saliency_ts) + 1
                if next >= len(self.saliencies):
                    next = 0
            else:
                next = 0
            self.current_saliency_ts = self.saliencies.keys()[next]


    def SelectNextAudience(self):
        # TODO: switch to next audience (according to audience ROI)


    def HandleTimer(self,data):

        # this is the heart of the synthesizer, here the lookat and eyecontact state machines take care of where the robot is looking, and random expressions and gestures are triggered to look more alive (like RealSense Tracker)

        ts = data.current_expected

        # ==== handle lookat
        if self.lookat == LookAt.IDLE:
            # no specific target, let Blender do it's soma cycle thing
            ()

        elif self.lookat == LookAt.AVOID:
            # TODO: find out where there is no saliency, hand or face
            # TODO: head_focus_pub
            ()

        elif self.lookat == LookAt.SALIENCY:
            self.saliency_counter -= 1
            if self.saliency_counter == 0:
                self.saliency_counter = random.randint(self.saliency_counter_min,self.saliency_counter_max)
                self.SelectNextSaliency()
            if self.current_saliency_ts != 0:
                cursaliency = self.saliencies[self.current_saliency_ts]
                self.SetHeadFocus(self.saliencies[self.current_saliency_ts].direction)

        elif self.lookat == LookAt.HAND:
            # stare at hand
            if self.hand != None:
                self.SetHeadFocus(self.hand.position)

        elif self.lookat == LookAt.AUDIENCE:
            self.audience_counter -= 1
            if self.audience_counter == 0:
                self.audience_counter = random.randint(self.audience_counter_min,self.audience_counter_max)
                self.SelectNextAudience()
                # TODO: self.SetHeadFocus()

        elif self.lookat == LookAt.SPEAKER:
            ()
            # TODO: look at the speaker, according to speaker ROI

        else:
            if self.lookat == LookAt.ALL_FACES:
                self.faces_counter -= 1
                if self.faces_counter == 0:
                    self.faces_counter = random.randint(self.faces_counter_min,self.faces_counter_max)
                    self.SelectNextFace()

            # take the current face
            if self.current_face_id != 0:
                curface = self.faces[self.current_face_id]
                face_pos = curface.position

                # ==== handle eyecontact (only for LookAt.ONE_FACE and LookAt.ALL_FACES)

                # calculate where left eye, right eye and mouth are on the current face
                left_eye_pos = Float32XYZ()
                right_eye_pos = Float32XYZ()
                mouth_pos = Float32XYZ()

                # all are 5cm in front of the center of the face
                left_eye_pos.x = face_pos.x - 0.05
                right_eye_pos.x = face_pos.x - 0.05
                mouth_pos.x = face_pos.x - 0.05

                left_eye_pos.y = face_pos.y + 0.03  # left eye is 3cm to the left of the center
                right_eye_pos.y = face_pos.y - 0.03  # right eye is 3cm to the right of the center
                mouth_pos.y = face_pos.y  # mouth is dead center

                left_eye_pos.z = face_pos.z + 0.06  # left eye is 6cm above the center
                right_eye_pos.z = face_pos.z + 0.06  # right eye is 6cm above the center
                mouth_pos.z = face_pos.z - 0.04  # mouth is 4cm below the center

                if self.eyecontact == EyeContact.IDLE:
                    # look at center of the head
                    self.SetHeadFocus(face_pos)

                elif self.eyecontact == EyeContact.LEFT_EYE:
                    # look at left eye
                    self.SetHeadFocus(left_eye_pos)

                elif self.eyecontact == EyeContact.RIGHT_EYE:
                    # look at right eye
                    self.SetHeadFocus(right_eye_pos)

                elif self.eyecontact == EyeContact.BOTH_EYES:
                    # switch between eyes back and forth
                    self.eyes_counter -= 1
                    if self.eyes_counter == 0:
                        self.eyes_counter = random.randint(self.eyes_counter_min,self.eyes_counter_max)
                        if self.current_eye == 1:
                            self.current_eye = 0
                        else:
                            self.current_eye = 1
                    # look at that eye
                    if self.current_eye == 0:
                        cur_eye_pos = left_eye_pos
                    else:
                        cur_eye_pos = right_eye_pos
                    self.SetHeadFocus(cur_eye_pos)

                elif self.eyecontact == EyeContact.TRIANGLE:
                    # cycle between eyes and mouth
                    self.eyes_counter -= 1
                    if self.eyes_counter == 0:
                        self.eyes_counter = random.randint(self.eyes_counter_min,self.eyes_counter_max)
                        if self.current_eye == 2:
                            self.current_eye = 0
                        else:
                            self.current_eye += 1
                    # look at that eye
                    if self.current_eye == 0: 
                        cur_eye_pos = left_eye_pos
                    elif self.current_eye == 1:
                        cur_eye_pos = right_eye_pos
                    elif self.current_eye == 2:
                        cur_eye_pos = mouth_pos
                    self.SetHeadFocus(cur_eye_pos)

        # TODO: start random expressions like RealSense Tracker

        # TODO: start random gestures like RealSense Tracker

        prune_before_time = ts - rospy.Duration.from_sec(self.keep_time)

        # flush faces dictionary, update current face accordingly
        to_be_removed = []
        for face in self.faces.values():
            if face.ts < prune_before_time:
                to_be_removed.append(face.cface_id)
        # remove the elements
        for key in to_be_removed:
            del self.faces[key]
            # make sure the selected face is always valid
            if self.current_face_id == key:
                self.SelectNextFace()
                
        # remove hand if it is too old
        if self.hand != None:
            if self.hand.ts < prune_before_time:
                self.hand = None

        # flush saliency dictionary
        to_be_removed = []
        for key in self.saliencies.keys():
            if key < prune_before_time:
                to_be_removed.append(key)
        # remove the elements
        for key in to_be_removed:
            del self.saliencies[key]
            # make sure the selected saliency is always valid
            if self.current_saliency_ts == key:
                self.SelectNextSaliency()


    def SetEyeContact(self, neweyecontact):

        if neweyecontact == self.eyecontact:
            return

        self.eyecontact = neweyecontact

        # initialize new eyecontact
        if self.eyecontact == EyeContact.IDLE:
            ()

        elif self.eyecontact == EyeContact.LEFT_EYE:
            ()

        elif self.eyecontact == EyeContact.RIGHT_EYE:
            ()

        elif self.eyecontact == EyeContact.BOTH_EYES:
            self.eyes_counter = random.randint(self.eyes_counter_min,self.eyes_counter_max)

        elif self.eyecontact == EyeContact.TRIANGLE:
            self.eyes_counter = random.randint(self.eyes_counter_min,self.eyes_counter_max)


    def SetLookAt(self, newlookat):

        if newlookat == self.lookat:
            return

        self.lookat = newlookat

        # initialize new lookat
        if self.lookat == LookAt.IDLE:
            ()

        elif self.lookat == LookAt.AVOID:
            ()

        elif self.lookat == LookAt.SALIENCY:
            # reset saliency switch counter
            self.saliency_counter = random.randint(self.saliency_counter_min,self.saliency_counter_max)

        elif self.lookat == LookAt.HAND:
            ()

        elif self.lookat == LookAt.ONE_FACE:
            # reset eye switch counter
            self.eyes_counter = random.randint(self.eyes_counter_min,self.eyes_counter_max)

        elif self.lookat == LookAt.ALL_FACES:
            # reset eye and face switch counters
            self.faces_counter = random.randint(self.faces_counter_min,self.faces_counter_max)
            self.eyes_counter = random.randint(self.eyes_counter_min,self.eyes_counter_max)

        elif self.lookat == LookAt.AUDIENCE:
            # reset audience switch counter
            self.audience_counter = random.randint(self.audience_counter_min,self.audience_counter_max)

        elif self.lookat == LookAt.SPEAKER:
            ()


    # ==== MAIN STATE MACHINE

    def SetState(self, newstate):

        # this is where the new main state is initialized, it sets up lookat and eyecontact states appropriately, manage perception system refresh rates and load random gesture and expression probabilities to be processed by HandleTimer

        if newstate == self.state:
            return

        self.state = newstate

        # initialize new state
        if self.state == State.SLEEPING:
            # the robot sleeps
            ()

        elif self.state == State.IDLE:
            # the robot is idle
            ()

        elif self.state == State.INTERESTED:
            # the robot is actively idle
            ()

        elif self.state == State.FOCUSED:
            # the robot is very interested at something specific
            ()

        elif self.state == State.SPEAKING:
            # the robot is speaking (directly/intimately) to people
            ()

        elif self.state == State.LISTENING:
            # the robot is listening to whoever is speaking
            ()

        elif self.state == State.PRESENTING:
            # the robot is presenting to the audience
            ()


    def HandleBehaviorConfig(self, config, level):

        if config.enable_flag != self.enable_flag:
            self.enable_flag = new_enable_flag
            # TODO: enable or disable the behaviors

        if config.synthesizer_rate != self.synthesizer_rate:
            self.synthesizer_rate = new_synthesizer_rate
            self.timer.shutdown()
            self.timer = rospy.Timer(rospy.Duration(1.0 / self.synthesizer_rate),self.HandleTimer)

        # update the counter ranges (and counters if the ranges changed)
        if config.saliency_counter_min != self.saliency_counter_min or config.saliency_counter_max != self.saliency_counter_max:
            self.saliency_counter_min = config.saliency_counter_min
            self.saliency_counter_max = config.saliency_counter_max
            self.saliency_counter = random.randint(self.saliency_counter_min,self.saliency_counter_max)

        if config.faces_counter_min != self.faces_counter_min or config.faces_counter_max != self.faces_counter_max:
            self.faces_counter_min = config.faces_counter_min
            self.faces_counter_max = config.faces_counter_max
            self.faces_counter = random.randint(self.faces_counter_min,self.faces_counter_max)

        if config.eyes_counter_min != self.eyes_counter_min or config.eyes_counter_max != self.eyes_counter_max:
            self.eyes_counter_min = config.eyes_counter_min
            self.eyes_counter_max = config.eyes_counter_max
            self.eyes_counter = random.randint(self.eyes_counter_min,self.eyes_counter_max)
        
        if config.audience_counter_min != self.audience_counter_min or config.audience_counter_max != self.audience_counter_max:
            self.audience_counter_min = config.audience_counter_min
            self.audience_counter_max = config.audience_counter_max
            self.audience_counter = random.randint(self.audience_counter_min,self.audience_counter_max)
        
        # keep time
        self.keep_time = config.keep_time

        # and set the states for each state machine
        self.SetLookAt(config.lookat_state)

        self.SetEyeContact(config.eyecontact_state)

        # and finally the overall state
        self.SetState(config.state)

        return config


    def HandleFace(self, msg):

        self.faces[msg.cface_id] = msg
        self.last_face = msg.cface_id

        # TEMP: if there is no current face, make this the current face
        if self.current_face_id == 0:
            self.current_face_id = msg.cface_id


    def HandleHand(self, msg):

        self.hand = msg


    def HandleSaliency(self, msg):

        self.saliencies[msg.ts] = msg

        # TEMP: if there is no current saliency vector, make this the current saliency vector
        if self.current_saliency_ts == 0:
            self.current_saliency_ts = msg.ts


    def HandleChatEvents(self, msg):
        # triggered when someone starts talking to the robot

        # go to listening state if robot is not talking
        #if self.state != State.USERS_SPEAKING:
        #    self.SetState(State.USERS_LISTENING)


    def HandleSpeechEvents(self, msg):
        print("{}".format(msg))
        # triggered when the robot starts or stops talking
        #if msg.data == "start":
            # robot starts talking
            #self.SetState(State.USERS_SPEAKING)
        #elif msg.data == "stop":
            # robot stops talking
            #self.SetState(State.USERS_IDLE)


    def HandleAudioDirection(self, msg):
        # use to correlate with person speaking to select correct current face
        ()


    def HandleMotion(self, msg):
        # use to trigger awareness of people even without seeing them
        ()


if __name__ == "__main__":
    rospy.init_node('behavior')
    node = Behavior()
    rospy.spin()
