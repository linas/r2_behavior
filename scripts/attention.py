#!/usr/bin/env python
import rospy
import time
import threading
import operator
import random
import logging
import math
# Attention regions
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Point
from geometry_msgs.msg import Point,PointStamped
from hr_msgs.msg import APILookAt
from r2_perception.msg import State, Face, SalientPoint
from hr_msgs.msg import Target, EmotionState, SetGesture
from hr_msgs.msg import pau
from performances.nodes import attention as AttentionRegion
from r2_behavior.cfg import AttentionConfig
from std_msgs.msg import String, Float64, UInt8, Int64
import dynamic_reconfigure.client
import tf

logger = logging.getLogger('hr.r2_behavior.attention')

# in interactive settings with people, the EyeContact machine is used to define specific states for eye contact
# this is purely mechanical, so it follows a very strict control logic; the overall state machines controls which
# eyecontact mode is actually used by switching the eyecontact state
class EyeContact:
    IDLE = 0  # don't make eye contact
    LEFT_EYE = 1  # look at left eye
    RIGHT_EYE = 2  # look at right eye
    BOTH_EYES = 3  # switch between both eyes
    TRIANGLE = 4  # switch between eyes and mouth


# the lookat machine is the lowest level and has the robot look at specific things: saliency, hands, faces
# this is purely mechanical, so it follows a very strict control logic; the overall state machines controls
# where the robot looks at by switching the lookat state
class LookAt:
    IDLE = 0  # look at nothing in particular
    AVOID = 1  # actively avoid looking at face, hand or saliency
    SALIENCY = 2  # look at saliency and switch
    ONE_FACE = 3  # look at single face and make eye contact
    ALL_FACES = 4  # look at all faces, make eye contact and switch
    REGION = 5  # look at the region and switch
    HOLD = 6 # Do not move the head while in  this state
    NEAREST_FACE = 7  # only look at face closest to the robot. This ensures robot do not get distracted by other people around
    POSES = 8

# params: current face


# the mirroring machine is the lowest level and has the robot mirror the face it is currently looking at
# this is purely mechanical, so it follows a very strict control logic; the overall state machine controls which mirroring more is actually used by switching the mirroring state
# class Mirroring:
#     IDLE = 0  # no mirroring
#     EYEBROWS = 1  # mirror the eyebrows only
#     EYELIDS = 2  # mirror the blinking only
#     EYES = 3  # mirror eyebrows and eyelids
#     MOUTH = 4  # mirror mouth opening
#     MOUTH_EYEBROWS = 5  # mirror mouth and eyebrows
#     MOUTH_EYELIDS = 6  # mirror mouth and eyelids
#     ALL = 7  # mirror everything


# params: eyebrows magnitude, eyelid magnitude, mouth magnitude


# the gaze machine is the lowest level and defines the robot head+gaze behavior
# this is purely mechanical, so it follows a very strict control logic; the overall state machine controls which gaze mode is actually used by switching the gaze state
class Gaze:
    GAZE_ONLY = 0  # only gaze
    HEAD_ONLY = 1  # only head
    GAZE_AND_HEAD = 2  # gaze and head at the same time
    GAZE_LEADS_HEAD = 3  # gaze first, and after some time have head follow
    HEAD_LEADS_GAZE = 4  # head first, and after some time have gaze follow


# params: gaze delay, gaze speed
REGIONS = {
    0: 'audience',  # Audience region selected
    1: 'main',  # Presenter (speaker) region
    2: 'specific'   # Co-presenter or some other region for specific setting
}

# awareness: saliency, hands, faces, motion sensors

class Attention:


    def InitCounter(self, counter, minmax):
        try:
            min_v = min(int(getattr(self, "{}_min".format(minmax))), int(getattr(self, "{}_max".format(minmax))))
            max_v = max(int(getattr(self, "{}_min".format(minmax))), int(getattr(self, "{}_max".format(minmax))))
            val = random.randint(int(min_v * self.synthesizer_rate),
                                 int(max_v * self.synthesizer_rate))
        except:
            val = 1
        setattr(self, "{}_counter".format(counter), val)

    def __getattr__(self, item):
        # Allow configuration attributes to be accessed directly
        if self.config is None:
            raise AttributeError
        return self.config.__getattr__(item)

    def __init__(self):

        # create lock
        self.lock = threading.Lock()

        self.robot_name = rospy.get_param("/robot_name")
        self.eyecontact = 0
        self.lookat = 0
        self.mirroring = 0
        # By default look with head and eyes
        self.gaze = 2
        # setup face, hand and saliency structures
        self.state = State()
        self.current_face_index = -1  # index to current face
        self.wanted_face_id = 0  # ID for wanted face
        self.current_saliency_index = -1  # index of current saliency vector
        self.current_eye = 0  # current eye (0 = left, 1 = right, 2 = mouth)
        self.interrupted_state = LookAt.IDLE  # which state was interrupted to look at all faces
        self.interrupting = False  # LookAt state is currently interrupted to look at all faces

        self.gaze_delay_counter = 0  # delay counter after with gaze or head follows head or gaze
        self.gaze_pos = None  # current gaze position
        # counter to run if face not visible
        self.no_face_counter = 0
        self.no_switch_counter = 0
        self.last_pose_counter = 0
        self.last_pose = None
        self.eyecontact_state = EyeContact.TRIANGLE
        self.tf_listener = tf.TransformListener(False, rospy.Duration.from_sec(1))

        # tracks last face by fsdk_id, if changes informs eye tracking to pause
        self.last_target = -2
        rospy.Subscriber('/{}/perception/state'.format(self.robot_name), State, self.HandleState)

        self.head_focus_pub = rospy.Publisher('/blender_api/set_face_target', Target, queue_size=1)
        self.gaze_focus_pub = rospy.Publisher('/blender_api/set_gaze_target', Target, queue_size=1)
        self.expressions_pub = rospy.Publisher('/blender_api/set_emotion_state', EmotionState, queue_size=1)
        self.gestures_pub = rospy.Publisher('/blender_api/set_gesture', SetGesture, queue_size=1)
        self.animationmode_pub = rospy.Publisher('/blender_api/set_animation_mode', UInt8, queue_size=1)
        self.setpau_pub = rospy.Publisher('/blender_api/set_pau', pau, queue_size=1)
        # Topic to publish target changes
        self.current_target_pub = rospy.Publisher('/behavior/current_target', Int64, queue_size=5, latch=True)

        self.synthesizer_rate = 30

        self.hand_events_pub = rospy.Publisher('/hand_events', String, queue_size=1)

        # Timer, started by config server
        self.timer = None

        # start dynamic reconfigure server
        self.configs_init = False
        self.config_server = Server(AttentionConfig, self.HandleConfig, namespace='/current/attention')


        # API calls (when the overall state is not automatically setting these values via dynamic reconfigure)
        self.eyecontact_sub = rospy.Subscriber('/behavior/attention/api/eyecontact',UInt8, self.HandleEyeContact)
        self.lookat_sub = rospy.Subscriber('/behavior/attention/api/lookat',APILookAt, self.HandleLookAt)
        self.mirroring_sub = rospy.Subscriber('/behavior/attention/api/mirroring',UInt8, self.HandleMirroring)
        self.gaze_sub = rospy.Subscriber('/behavior/attention/api/gaze',UInt8, self.HandleGaze)



    def ChangeTarget(self, id=None):
        if id is None:
            id = int(time.time()*10)
        if id <> self.last_target:
            self.last_target = id
            self.current_target_pub.publish(id)


    def UpdateStateDisplay(self):
        self.config_server.update_configuration({
            "eyecontact_state": self.eyecontact,
            "lookat_state": self.lookat,
            "mirroring_state": self.mirroring,
            "gaze_state": self.gaze,
        })


    def HandleConfig(self, config, level):
        with self.lock:
            self.config = config
            # and set the states for each state machine
            #self.SetEyeContact(config.eyecontact_state)
            self.SetLookAt(config.lookat_state)
            # self.SetMirroring(config.mirroring_state)
            # self.SetGaze(config.gaze_state)
            # Counters
            if not self.configs_init:
                self.timer = rospy.Timer(rospy.Duration.from_sec(1.0 / self.synthesizer_rate), self.HandleTimer)
                # self.InitCounter("saliency","saliency_time")
                self.InitCounter("faces", "faces_time")
                self.InitCounter("region", "region_time")
                # self.InitCounter("rest", "rest_time")

            self.configs_init = True
        return config


    def getBlenderPos(self, pos, ts, frame_id):
        if frame_id == 'blender':
            return pos
        else:
            ps = PointStamped()
            ps.header.seq = 0
            ps.header.stamp = ts
            ps.header.frame_id = frame_id
            ps.point.x = pos.x
            ps.point.y = pos.y
            ps.point.z = pos.z
            if self.tf_listener.canTransform("blender", frame_id, ts):
                pst = self.tf_listener.transformPoint("blender", ps)
                return pst.point
            else:
                raise Exception("tf from robot to blender did not work")


    def SetGazeFocus(self, pos, speed, ts, frame_id='robot'):
        try:
            pos = self.getBlenderPos(pos, ts, frame_id)
            msg = Target()
            msg.x = max(0.3, pos.x)
            msg.y = pos.y if not math.isnan(pos.y) else 0
            msg.z = pos.z if not math.isnan(pos.z) else 0
            msg.z = max(-0.3, min(0.3, msg.z))
            msg.speed = speed
            self.gaze_focus_pub.publish(msg)
        except Exception as e:
            logger.warn("Gaze focus exception: {}".format(e))

    def SetHeadFocus(self, pos, speed, ts, frame_id='robot'):
        try:
            pos = self.getBlenderPos(pos, ts, frame_id)
            msg = Target()
            msg.x = max(0.3, pos.x)
            msg.y = pos.y if not math.isnan(pos.y) else 0
            msg.z = pos.z if not math.isnan(pos.z) else 0
            msg.z = max(-0.3, min(0.3, msg.z))
            msg.speed = speed
            self.head_focus_pub.publish(msg)
        except Exception as e:
            logger.warn("Head focus exception: {}".format(e))


    def UpdateGaze(self, pos, ts, frame_id="robot"):

        self.gaze_pos = pos

        if self.gaze == Gaze.GAZE_ONLY:
            self.SetGazeFocus(pos, 5.0, ts, frame_id)

        elif self.gaze == Gaze.HEAD_ONLY:
            self.SetHeadFocus(pos, self.head_speed, ts, frame_id)

        elif self.gaze == Gaze.GAZE_AND_HEAD:
            self.SetGazeFocus(pos, 5.0, ts, frame_id)
            self.SetHeadFocus(pos, self.head_speed, ts, frame_id)

        elif self.gaze == Gaze.GAZE_LEADS_HEAD:
            self.SetGazeFocus(pos, 5.0, ts, frame_id)

        elif self.gaze == Gaze.HEAD_LEADS_GAZE:
            self.SetHeadFocus(pos, self.head_speed, ts, frame_id)


    def SelectNextFace(self):
        # switch to the next (or first) face
        if self.state is None or len(self.state.faces) == 0:
            # there are no faces, so select none
            self.current_face_index = -1
            return

        if self.lookat  == LookAt.NEAREST_FACE:
            self.current_face_index, f = min(enumerate(self.state.faces),
                                             key=lambda f: (f[1].position.x**2+f[1].position.y**2))
        else:
            # Pick random or any other face from list
            if self.current_face_index == -1:
                self.current_face_index = 0
            else:
                self.current_face_index += 1
                if self.current_face_index >= len(self.state.faces):
                    self.current_face_index = 0

    def SelectNextPose(self):
        # switch to the next (or first) face
        if self.state is None or len(self.state.poses) == 0:
            # there are no faces, so select none
            self.last_pose_counter = 0
            return False
        # Select nearest pose to the robot, otherwise select nearest pose to previous
        if self.last_pose is None or self.last_pose_counter <= 0:
            i, p = min(enumerate(self.state.poses),
                                         key=lambda f: (f[1].position.x**2+f[1].position.y**2))
            self.last_pose_counter = self.synthesizer_rate * self.min_time_between_targets
        else:
            i, p = min(enumerate(self.state.poses),
                                         key=lambda f: ((self.last_pose.position.x - f[1].position.x)**2+
                                                        (self.last_pose.position.y - f[1].position.y)**2+
                                                        (self.last_pose.position.z - f[1].position.z)**2))
            self.last_pose_counter -= 1
        self.last_pose = p
        return p


    def SelectNextSalientPoint(self):

        # switch to the next (or first) saliency vector
        if (self.state == None):
            self.current_saliency_index = -1
        if len(self.state.salientpoints) == 0:
            # there are no saliency vectors, so select none
            self.current_saliency_index = -1
            return
        if self.current_saliency_index == -1:
            self.current_saliency_index = 0
        else:
            self.current_saliency_index += 1
            if self.current_saliency_index >= len(self.state.salientpoints):
                self.current_saliency_index = 0


    def SelectNextRegion(self):
        # Check if setperformance has set regions
        regions = rospy.get_param("/{}/performance_regions".format(self.robot_name), {})
        if len(regions) == 0:
            regions = rospy.get_param("/{}/regions".format(self.robot_name), {})
        point = AttentionRegion.get_point_from_regions(regions, REGIONS[self.attention_region])
        return Point(x=point['x'], y=point['y'], z=point['z'])


    def StepLookAtFace(self, ts):

        if self.current_face_index == -1:
            raise Exception("No face available")

        try:
            curface = self.state.faces[self.current_face_index]
        except:
            raise Exception("No face available")

        self.ChangeTarget(curface.fsdk_id)
        face_pos = curface.position

        # ==== handle eyecontact (only for LookAt.ONE_FACE and LookAt.ALL_FACES)

        # calculate where left eye, right eye and mouth are on the current face
        left_eye_pos = Point()
        right_eye_pos = Point()
        mouth_pos = Point()

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
            self.UpdateGaze(face_pos, ts)

        elif self.eyecontact == EyeContact.LEFT_EYE:
            # look at left eye
            self.UpdateGaze(left_eye_pos, ts)

        elif self.eyecontact == EyeContact.RIGHT_EYE:
            # look at right eye
            self.UpdateGaze(right_eye_pos, ts)

        elif self.eyecontact == EyeContact.BOTH_EYES:
            # switch between eyes back and forth
            self.eyes_counter -= 1
            if self.eyes_counter == 0:
                self.InitCounter("eyes", "eyes_time")
                if self.current_eye == 1:
                    self.current_eye = 0
                else:
                    self.current_eye = 1
            # look at that eye
            if self.current_eye == 0:
                cur_eye_pos = left_eye_pos
            else:
                cur_eye_pos = right_eye_pos
            self.UpdateGaze(cur_eye_pos, ts)

        elif self.eyecontact == EyeContact.TRIANGLE:
            # cycle between eyes and mouth
            self.eyes_counter -= 1
            if self.eyes_counter == 0:
                self.InitCounter("eyes", "eyes_time")
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
            self.UpdateGaze(cur_eye_pos, ts)

        # mirroring
        # msg = pau()
        # msg.m_coeffs = []
        # msg.m_shapekeys = []

        # if self.mirroring == Mirroring.EYEBROWS or self.mirroring == Mirroring.EYES or self.mirroring == Mirroring.MOUTH_EYEBROWS or self.mirroring == Mirroring.ALL:
        #    # mirror eyebrows
        #    left_brow = curface.left_brow
        #    right_brow = curface.right_brow
        #    msg.m_coeffs.append("brow_outer_UP.L")
        #    msg.m_shapekeys.append(left_brow)
        #    msg.m_coeffs.append("brow_inner_UP.L")
        #    msg.m_shapekeys.append(left_brow * 0.8)
        #    msg.m_coeffs.append("brow_outer_DN.L")
        #    msg.m_shapekeys.append(1.0 - left_brow)
        #    msg.m_coeffs.append("brow_outer_up.R")
        #    msg.m_shapekeys.append(right_brow)
        #    msg.m_coeffs.append("brow_inner_UP.R")
        #    msg.m_shapekeys.append(right_brow * 0.8)
        #    msg.m_coeffs.append("brow_outer_DN.R")
        #    msg.m_shapekeys.append(1.0 - right_brow)

        # if self.mirroring == Mirroring.EYELIDS or self.mirroring == Mirroring.EYES or self.mirroring == Mirroring.MOUTH_EYELIDS or self.mirroring == Mirroring.ALL:
        #    # mirror eyelids
        #    eyes_closed = ((1.0 - curface.left_eyelid) + (1.0 - curface.right_eyelid)) / 2.0
        #    msg.m_coeffs.append("eye-blink.UP.R")
        #    msg.m_shapekeys.append(eyes_closed)
        #    msg.m_coeffs.append("eye-blink.UP.L")
        #    msg.m_shapekeys.append(eyes_closed)
        #    msg.m_coeffs.append("eye-blink.LO.R")
        #    msg.m_shapekeys.append(eyes_closed)
        #    msg.m_coeffs.append("eye-blink.LO.L")
        #    msg.m_shapekeys.append(eyes_closed)

        # if self.mirroring == Mirroring.MOUTH or self.mirroring == Mirroring.MOUTH_EYEBROWS or self.mirroring == Mirroring.MOUTH_EYELIDS:
        #    # mirror mouth
        #    mouth_open = curface.mouth_open
        #    msg.m_coeffs.append("lip-JAW.DN")
        #    msg.m_shapekeys.append(mouth_open)

        # if self.mirroring != Mirroring.IDLE:
        #    self.StartPauMode()
        #    self.setpau_pub.publish(msg)


    def HandleTimer(self, data):
        looking_at_face = False
        with self.lock:
            if not self.configs_init:
                return False
            if not self.enable_flag:

                self.ChangeTarget(-1)
                return False

            # this is the heart of the synthesizer, here the lookat and eyecontact state machines take care of where the robot is looking, and random expressions and gestures are triggered to look more alive (like RealSense Tracker)
            ts = data.current_expected
            #If nowhere to look, straighten the head
            idle_point = Point(x = 1, y=0, z=0)
            # Fallback to look at region
            region = False
            # Fallback o rest
            idle =False
            # ==== handle lookat
            if self.lookat == LookAt.AVOID:
                # TODO: find out where there is no saliency, hand or face
                # TODO: head_focus_pub
                ()

            if self.lookat == LookAt.HOLD or self.lookat == LookAt.IDLE:
                # Do nothing
                pass

            # if self.lookat == LookAt.SALIENCY:
            #     self.saliency_counter -= 1
            #     if self.saliency_counter == 0:
            #         self.SelectNextSalientPoint()
            #         # Reset head position if nothing happens
            #         if self.current_saliency_index == -1:
            #             self.UpdateGaze(idle_point, ts, frame_id='blender')
            #         else:
            #             # Init counter only if any salient point found
            #             self.InitCounter("saliency","saliency_time")
            #
            #     if self.current_saliency_index != -1:
            #         cursaliency = self.state.salientpoints[self.current_saliency_index]
            #         self.UpdateGaze(cursaliency.position, ts)

            elif self.lookat == LookAt.REGION:
                self.region_counter -= 1
                if self.region_counter == 0:
                    self.InitCounter("region", "region_time")
                    # SelectNextRegion returns idle point if no region set
                    point = self.SelectNextRegion()
                    # Attention points are calculated in blender frame
                    self.UpdateGaze(point, ts, frame_id='blender')
            # elif self.lookat == LookAt.POSES:
            #     pose = self.SelectNextPose()
            #     if pose:
            #         self.UpdateGaze(pose.position, ts)
            #         self.no_switch_counter = self.synthesizer_rate * self.min_time_between_targets
            #     else:
            #         self.no_switch_counter -= 1
            #         if self.no_switch_counter < 0:
            #             point = self.SelectNextRegion()
            #             self.UpdateGaze(point, ts, frame_id='blender')
            #             self.no_switch_counter = self.synthesizer_rate * self.min_time_between_targets
            else:
                if self.lookat == LookAt.ALL_FACES or self.lookat == LookAt.NEAREST_FACE or self.lookat == LookAt.POSES:
                    self.faces_counter -= 1
                    if self.faces_counter == 0:
                        self.SelectNextFace()
                        self.no_switch_counter = self.synthesizer_rate * self.min_time_between_targets
                        self.InitCounter("faces", "faces_time")
                    try:
                        # This will make sure robot will look somewhere so eye contact only should be paused
                        # It should fallback to attention region if defined or idle point
                        looking_at_face = True
                        # only look at faces after switch is allowed:
                        self.StepLookAtFace(ts)
                        # Reset after face is found
                        self.no_face_counter = 0
                    except:
                        # Look at poses if no faces are visible, and then look at region otherwise
                        pose = self.SelectNextPose()
                        self.no_face_counter -= 1
                        if pose:
                            self.UpdateGaze(pose.position, ts)
                            self.no_switch_counter = self.synthesizer_rate * self.min_time_between_targets
                        else:
                            self.no_switch_counter -= 1
                            if self.no_switch_counter <= 0:
                                if self.no_face_counter <= 0:
                                    point = self.SelectNextRegion()
                                    self.InitCounter('no_face', 'region_time')
                                    self.UpdateGaze(point, ts, frame_id='blender')
                                    self.no_switch_counter = self.synthesizer_rate * self.min_time_between_targets

            if self.lookat == LookAt.REGION or region:
                self.region_counter -= 1
                if self.region_counter == 0:
                    # Eye contact enabled when looking at region
                    if self.eyecontact > 0:
                        looking_at_face = True
                    self.InitCounter("region", "region_time")
                    # SelectNextRegion returns idle point if no region set
                    try:
                        point = self.SelectNextRegion()
                        # Attention points are calculated in blender frqame
                        self.UpdateGaze(point, ts, frame_id='blender')
                        # Target have been changed

                    except:
                        idle = True

            # if self.lookat == LookAt.IDLE or idle:
            #     self.rest_counter -= 1
            #     if self.rest_counter == 0:
            #         self.InitCounter("rest", "rest_time")
            #         if self.eyecontact > 0:
            #             looking_at_face = True
            #         idle_point = Point(x=1, y=random.uniform(-self.rest_range_x, self.rest_range_y)
            #                            , z=random.uniform(-self.rest_range_y, self.rest_range_y))
            #         # Attention points are calculated in blender frame
            #         self.UpdateGaze(idle_point, ts, frame_id='blender')
            #         # Target have been changed
            #         self.ChangeTarget()

            # have gaze or head follow head or gaze after a while
            if self.gaze_delay_counter > 0 and self.gaze_pos != None:

                self.gaze_delay_counter -= 1
                if self.gaze_delay_counter == 0:

                    if self.gaze == Gaze.GAZE_LEADS_HEAD:
                        self.SetHeadFocus(self.gaze_pos, self.gaze_speed,ts)
                        self.gaze_delay_counter = int(self.gaze_delay * self.synthesizer_rate)

                    elif self.gaze == Gaze.HEAD_LEADS_GAZE:
                        self.SetGazeFocus(self.gaze_pos, self.gaze_speed,ts)
                        self.gaze_delay_counter = int(self.gaze_delay * self.synthesizer_rate)

            # when speaking, sometimes look at all faces
            # if self.interrupt_to_all_faces:
            #
            #     if self.interrupting:
            #         self.all_faces_duration_counter -= 1
            #         if self.all_faces_duration_counter <= 0:
            #             self.interrupting = False
            #             self.InitCounter("all_faces_duration", "all_faces_duration")
            #             self.SetLookAt(self.interrupted_state)
            #             self.UpdateStateDisplay()
            #     else:
            #         self.all_faces_start_counter -= 1
            #         if self.all_faces_start_counter <= 0:
            #             self.interrupting = True
            #             self.InitCounter("all_faces_start", "all_faces_start_time")
            #             self.interrupted_state = self.lookat
            #             self.SetLookAt(LookAt.ALL_FACES)
            #             self.UpdateStateDisplay()

        if not looking_at_face:
            self.ChangeTarget(-1)

    def SetEyeContact(self, neweyecontact):

        if neweyecontact == self.eyecontact:
            return

        self.eyecontact = neweyecontact

        # if self.eyecontact == EyeContact.BOTH_EYES or self.eyecontact == EyeContact.TRIANGLE:
        #     self.InitCounter("eyes", "eyes_time")


    def SetLookAt(self, newlookat):

        if newlookat == self.lookat:
            return

        self.lookat = newlookat

        if self.lookat == LookAt.SALIENCY:
            self.InitCounter("saliency","saliency_time")

        # elif self.lookat == LookAt.ONE_FACE:
        #     self.InitCounter("eyes", "eyes_time")

        elif self.lookat == LookAt.ALL_FACES:
            self.InitCounter("faces", "faces_time")
            # self.InitCounter("eyes", "eyes_time")

        elif self.lookat == LookAt.REGION:
            self.InitCounter("region", "region_time")

        # elif self.lookat == LookAt.IDLE:
        #     self.InitCounter("rest", "rest_time")

    def StartPauMode(self):

        mode = UInt8()
        mode.data = 148
        self.animationmode_pub.publish(mode)


    def StopPauMode(self):

        mode = UInt8()
        mode.data = 0
        self.animationmode_pub.publish(mode)


    def SetMirroring(self, newmirroring):
        pass
        # logger.warn("SetMirroring {}".format(newmirroring))
        # if newmirroring == self.mirroring: return
        #
        # self.mirroring = newmirroring
        #
        # if self.mirroring == Mirroring.IDLE:
        #     self.StopPauMode()
        # else:
        #     self.StartPauMode()


    def SetGaze(self, newgaze):

        if newgaze == self.gaze:
            return

        self.gaze = newgaze

        if self.gaze == Gaze.GAZE_LEADS_HEAD or self.gaze == Gaze.HEAD_LEADS_GAZE:
            self.gaze_delay_counter = int(self.gaze_delay * self.synthesizer_rate)


    def HandleState(self, data):
        with self.lock:

            self.state = data

            # if there is a wanted face, try to find it and use that
            if self.wanted_face_id != 0:
                index = 0
                self.current_face_index = -1
                for face in self.state.faces:
                    if face.id == self.wanted_face_id:
                        self.current_face_index = index
                        break
                    index += 1

            # otherwise, just make sure current_face_index is valid
            elif (self.current_face_index >= len(self.state.faces)) or (self.current_face_index == -1):
                # Only try to look at new faces once there are no
                if self.no_switch_counter <= 0:
                    self.SelectNextFace()
            # TODO: it's better to have the robot look at the same ID, regardless of which index in the stateface list

            # # if there is no current saliency or the current saliency is out of range, select a new current saliency
            # if (self.current_saliency_index >= len(self.state.salientpoints)) or (self.current_saliency_index == -1):
            #     self.SelectNextSaliency()


    def HandleEyeContact(self,data):

        with self.lock:

            self.SetEyeContact(data)

        self.UpdateStateDisplay()


    def HandleLookAt(self,data):

        with self.lock:

            if data.mode == LookAt.ONE_FACE: # if client wants to the robot to look at one specific face (even if the face temporarily disappears from the state)
                self.wanted_face_id = data.id
            elif data.mode == LookAt.REGION:
                self.attention_region = data.id
            else:
                self.wanted_face_id = 0
            self.SetLookAt(self,data.mode)

        self.UpdateStateDisplay()


    def HandleMirroring(self,data):

        with self.lock:

            self.SetMirroring(data)

        self.UpdateStateDisplay()


    def HandleGaze(self,data):

        with self.lock:

            self.SetGaze(data)

        self.UpdateStateDisplay()


if __name__ == "__main__":
    rospy.init_node('attention')
    node = Attention()
    rospy.spin()
