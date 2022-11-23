"""
    Controller of Nao robot for navigation using ddpg algorithm.
    This python file defines agent and environment classes
    Set up logic flow for reinforcement learning
"""

from controller import Robot, Keyboard, Motion
from controller import Supervisor, Node
from itertools import count
from datetime import datetime
import time
import math
import sys
import random
import numpy as np

from ddpg import DDPG
from federation import federation

STATE_DIM = 4
ACTION_DIM = 3
EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99
MODE = 'train'
LOAD = False
NUM_ROBOT = 3
max_step = 200
max_episode = 50000
train_start_step = 1000
log_interval = 10

fed_episode = 25
# fed_episode = 100000000

max_length_of_trajectory = 100
test_iteration = 1000

"""
    Define some useful APIs for the Nao agent in the environment.
    Could get sonar data, translation and rotation infomation.
    Provide methods to instruct the robot to move.
"""


class Nao_agent (Supervisor):
    PHALANX_MAX = 8

    # load motion files
    def __loadMotionFiles(self):
        self.handWave = Motion('../../motions/HandWave.motion')
        self.forwards = Motion('../../motions/Forwards.motion')
        self.backwards = Motion('../../motions/Backwards.motion')
        self.sideStepLeft = Motion('../../motions/SideStepLeft.motion')
        self.sideStepRight = Motion('../../motions/SideStepRight.motion')
        self.turnLeft60 = Motion('../../motions/TurnLeft60.motion')
        self.turnRight60 = Motion('../../motions/TurnRight60.motion')
        self.turnLeft40 = Motion('../../motions/TurnLeft40.motion')
        self.turnRight40 = Motion('../../motions/TurnRight40.motion')

    def fallCheck(self):
        translation = self.getTranslation()
        if translation[1] < 0.05:
            return True
        else:
            return False

    def findAndEnableDevices(self):
        # get the time step of the current world.
        self.timeStep = int(self.getBasicTimeStep())

        # camera
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)

        # accelerometer
        self.accelerometer = self.getDevice('accelerometer')
        self.accelerometer.enable(4 * self.timeStep)

        # gyro
        self.gyro = self.getDevice('gyro')
        self.gyro.enable(4 * self.timeStep)

        # gps
        self.gps = self.getDevice('gps')
        self.gps.enable(4 * self.timeStep)

        # inertial unit
        self.inertialUnit = self.getDevice('inertial unit')
        self.inertialUnit.enable(self.timeStep)

        # ultrasound sensors
        self.us = []
        usNames = ['Sonar/Left', 'Sonar/Right']
        for i in range(0, len(usNames)):
            self.us.append(self.getDevice(usNames[i]))
            self.us[i].enable(self.timeStep)

        # foot sensors
        self.fsr = []
        fsrNames = ['LFsr', 'RFsr']
        for i in range(0, len(fsrNames)):
            self.fsr.append(self.getDevice(fsrNames[i]))
            self.fsr[i].enable(self.timeStep)

        # foot bumpers
        self.lfootlbumper = self.getDevice('LFoot/Bumper/Left')
        self.lfootrbumper = self.getDevice('LFoot/Bumper/Right')
        self.rfootlbumper = self.getDevice('RFoot/Bumper/Left')
        self.rfootrbumper = self.getDevice('RFoot/Bumper/Right')
        self.lfootlbumper.enable(self.timeStep)
        self.lfootrbumper.enable(self.timeStep)
        self.rfootlbumper.enable(self.timeStep)
        self.rfootrbumper.enable(self.timeStep)

        # there are 7 controlable LED groups in Webots
        self.leds = []
        self.leds.append(self.getDevice('ChestBoard/Led'))
        self.leds.append(self.getDevice('RFoot/Led'))
        self.leds.append(self.getDevice('LFoot/Led'))
        self.leds.append(self.getDevice('Face/Led/Right'))
        self.leds.append(self.getDevice('Face/Led/Left'))
        self.leds.append(self.getDevice('Ears/Led/Right'))
        self.leds.append(self.getDevice('Ears/Led/Left'))

        # get phalanx motor tags
        # the real Nao has only 2 motors for RHand/LHand
        # but in Webots we must implement RHand/LHand with 2x8 motors
        self.lphalanx = []
        self.rphalanx = []
        self.maxPhalanxMotorPosition = []
        self.minPhalanxMotorPosition = []
        for i in range(0, self.PHALANX_MAX):
            self.lphalanx.append(self.getDevice("LPhalanx%d" % (i + 1)))
            self.rphalanx.append(self.getDevice("RPhalanx%d" % (i + 1)))

            # assume right and left hands have the same motor position bounds
            self.maxPhalanxMotorPosition.append(
                self.rphalanx[i].getMaxPosition())
            self.minPhalanxMotorPosition.append(
                self.rphalanx[i].getMinPosition())

        # shoulder pitch motors
        self.RShoulderPitch = self.getDevice("RShoulderPitch")
        self.LShoulderPitch = self.getDevice("LShoulderPitch")

        # keyboard
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(10 * self.timeStep)

    def __init__(self):
        Supervisor.__init__(self)
        self.currentlyPlaying = False

        # initialize stuff
        self.findAndEnableDevices()
        self.__loadMotionFiles()

        self.robot_node = Supervisor.getSelf(self)
        self.name = self.robot_node.getField("name").getSFString()

        self.initial_node = self.robot_node.saveState('initial')
        self.target = Supervisor.getFromDef(
            self, 'target'+self.name).getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        self.translation_field = self.robot_node.getField("translation")

        # step control
        self.__synchronize_iteration = 0
        self.go_forward = False
        self.go_back = False
        self.go_left = False
        self.go_right = False
        self.turn_left40 = False
        self.turn_Right40 = False
        self.turn_left60 = False
        self.turn_Right60 = False

    def reset_target(self, Xtarget, Ytarget):
        self.target.setSFVec3f([Xtarget, -0.14, Ytarget])

    def getName(self):
        return self.name

    def getRotation(self):
        return self.rotation_field.getSFRotation()

    def getTranslation(self):
        return self.translation_field.getSFVec3f()

    def getInertialUnit(self):
        rpy = self.inertialUnit.getRollPitchYaw()
        # print("Yaw: "+str(rpy[2]))
        return rpy[0], rpy[1], rpy[2]

    def getUltrasoundSensors(self):
        self.step(self.timeStep)
        dist = []
        for i in range(0, len(self.us)):
            dist.append(self.us[i].getValue())

        self.step(self.timeStep)

        return dist[0], dist[1]

    def pushSimulation(self):
        self.step(self.timeStep)

    def __startMotion(self, motion):
        # interrupt current motion
        if self.currentlyPlaying:
            self.currentlyPlaying.stop()

        # start new motion
        motion.play()
        self.currentlyPlaying = motion

    def __setMotion(self, command):
        if command == "forward":
            self.go_forward = True
            self.__synchronize_iteration = 85
        elif command == "backward":
            self.go_back = True
            self.__synchronize_iteration = 85
        elif command == "left60":
            self.turn_left60 = True
            self.__synchronize_iteration = 160
        elif command == "right60":
            self.turn_Right60 = True
            self.__synchronize_iteration = 160
        elif command == "left40":
            self.turn_left40 = True
            self.__synchronize_iteration = 120
        elif command == "right40":
            self.turn_Right40 = True
            self.__synchronize_iteration = 120

    def Move(self, command):
        # a is used to synchronize the motion simulation
        a = 0
        # set direction
        if command == "left20":
            self.Move("left60")
            self.Move("right40")
            return
        elif command == "right20":
            self.Move("right60")
            self.Move("left40")
            return
        elif command == "left100":
            self.Move("left40")
            self.Move("left60")
        elif command == "right100":
            self.Move("right40")
            self.Move("right60")
        else:
            self.__setMotion(command)
            while self.step(self.timeStep) != -1 and a < self.__synchronize_iteration:

                if self.go_forward:
                    self.__startMotion(self.forwards)

                elif self.go_back:
                    self.__startMotion(self.backwards)

                elif self.turn_left60:
                    self.__startMotion(self.turnLeft60)

                elif self.turn_Right60:
                    self.__startMotion(self.turnRight60)

                elif self.turn_left40:
                    self.__startMotion(self.turnLeft40)

                elif self.turn_Right40:
                    self.__startMotion(self.turnRight40)

                # make sure all controls are set to false after playingc
                if a < 10:
                    a += 1
                else:
                    self.go_forward = False
                    self.go_back = False
                    self.turn_left40 = False
                    self.turn_Right40 = False
                    self.turn_left60 = False
                    self.turn_Right60 = False
                    a += 1

        return


"""
    Similar to gym interfaces.
    Provide ways for the algorithm to interact with the environment
"""


class Env():
    def __init__(self):
        self.agent = Nao_agent()

        self.name = self.agent.getName()

        # get initial position
        self.initialTranslation = self.agent.getTranslation()
        self.initialRotation = self.agent.getRotation()

        # target location
        self.XTarget = self.agent.target.getSFVec3f()[0]
        self.YTarget = self.agent.target.getSFVec3f()[2]

        # pivot coordinates
        self.pivot_x = 0
        self.pivot_y = 0

        size = 1.4
        if (self.name == '2'):
            self.pivot_y = -size
        elif (self.name == '3'):
            self.pivot_y = size
        # self.resetTarget()

    def pushSimulation(self):
        self.agent.pushSimulation()

    def getName(self):
        return self.name

    def resetTarget(self):
        size = 2

        if (self.name == '2'):
            self.pivot_y = -size
        elif (self.name == '3'):
            self.pivot_y = size
        upper = 0.6
        lower = 0.1
        if(random.uniform(0, 1) < 0.5):
            self.XTarget = random.uniform(
                self.pivot_x+lower, self.pivot_x+upper)
        else:
            self.XTarget = random.uniform(
                self.pivot_x-lower, self.pivot_x-upper)
        if(random.uniform(0, 1) >= 0.5):
            self.YTarget = random.uniform(
                self.pivot_y+lower, self.pivot_y+upper)
        else:
            self.YTarget = random.uniform(
                self.pivot_y-lower, self.pivot_y-upper)
        self.agent.reset_target(self.XTarget, self.YTarget)

    def getState(self):
        roll, pitch, yaw = self.agent.getInertialUnit()
        translation = self.agent.getTranslation()
        # vector_pos = np.array([self.XTarget-translation[0], -self.YTarget-translation[2]])
        # vector_yaw = np.array([math.cos(yaw), math.sin(yaw)])
        # distance = math.sqrt(np.sum(np.power(vector_pos, 2)))
        # angle_difference = math.acos(np.dot(vector_pos, vector_yaw)/distance)
        left_sonar, right_sonar = self.agent.getUltrasoundSensors()

        return np.array([left_sonar, right_sonar, translation[0]-self.pivot_x, translation[2]-self.pivot_y])

    # to do
    def getReward(self, state):
        left_sonar = state[0]
        right_sonar = state[1]
        distance = math.sqrt((state[2]+self.pivot_x-self.XTarget)
                             ** 2+(state[3]+self.pivot_y-self.YTarget)**2)

        return (1-math.e**(0.99*distance))

    def Step(self, action):
        Action = np.argmax(action)

        if Action == 0:
            # self.agent.Move("forward")
            pass
        elif Action == 1:
            # self.agent.Move("left20")
            self.agent.Move("left40")
        elif Action == 2:
            # self.agent.Move("right20")
            self.agent.Move("right40")
        self.agent.Move("forward")

        next_state = self.getState()
        reward = self.getReward(next_state)
        done = False

        distance = math.sqrt(
            (next_state[2]+env.pivot_x-self.XTarget)**2+(next_state[3]+env.pivot_y-self.YTarget)**2)
        if (next_state[0] <= 0.25 or next_state[1] <= 0.25 or self.agent.fallCheck()):
            print("[robot"+self.name+"]: crash!!!")
            reward = -200
            # self.resetTarget()
            done = True
        elif (distance <= 0.2):
            print("[robot"+self.name+"]: hit!!!")
            reward = 300
            done = True
            # self.resetTarget()

        return next_state, reward, done

    def reset(self):
        self.agent.robot_node.loadState('initial')
        self.agent.robot_node.resetPhysics()

        return self.getState()


# create the Robot instance and run main loop
if __name__ == "__main__":

    with open('fed_control.txt', 'w') as f:
        f.write('0')
    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    print(current_time)

    env = Env()
    agent = DDPG(STATE_DIM, ACTION_DIM, 1, env.getName(), current_time)
    print("robot"+env.getName()+" initiated and running...")

    ep_r = 0
    if MODE == 'test':
        pass

    elif MODE == 'train':
        if LOAD:
            agent.load()
        total_step = 0

        # i: number of episodes
        for i in range(max_episode):
            st_time = time.time()
            total_reward = 0
            step = 0
            state = env.reset()

            num_hit = 0

            # t: steps
            for t in count():
                action = agent.select_action(state)
                if (random.uniform(0, 1) <= EPSILON):
                    x = np.random.random_sample(ACTION_DIM,)
                    action = np.exp(x - np.max(x)) / \
                        np.sum(np.exp(x - np.max(x)))

                next_state, reward, done = env.Step(action)

                agent.replay_buffer.push(
                    (state, next_state, action, reward, np.float(done)))

                state = next_state

                step += 1
                total_reward += reward

                if reward == 300:
                    num_hit += 1

                if done or step > max_step:
                    break

            total_step += step+1
            print("[robot"+env.getName()+"]: Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(
                total_step, i, total_reward))

            ## conf configuration
            conf = f'.F{fed_episode}'
            with open('./reward/'+env.getName()+'/reward' + current_time + conf+ '.txt', 'a+') as f:
                f.write(str(total_reward) + '\n')

            with open('./hit/'+env.getName()+'/hit' + current_time + conf + '.txt', 'a+') as f:
                f.write(str(num_hit) + '\n')

            if EPSILON > EPSILON_MIN:
                EPSILON = EPSILON*EPSILON_DECAY

            if total_step > train_start_step:
                print("====================================")
                print("[robot"+env.getName()+"]:Start training...")
                agent.update()
                print("[robot"+env.getName()+"]:Training ended.")
                print("====================================")
           # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % log_interval == 0:
                agent.save(i)
            print("ep time ", time.time() - st_time)

            # start fed process
            if i % fed_episode == 0 and i != 0:
                # update control file
                print("[robot"+env.getName()+"]: set control...")
                cur = 0
                with open('fed_control.txt', 'r') as f:
                    cur = f.read()
                cur = (int)(cur) + 1
                with open('fed_control.txt', 'w') as f:
                    f.write(str(cur))
                print("[robot"+env.getName()+"]: ready for federation...")
                if env.getName() == '1':
                    cont = 0
                    while(True):
                        cont += 1
                        env.pushSimulation()
                        sync = 0
                        if cont % 10 == 0:
                            with open("fed_control.txt", 'r') as f:
                                sync = (int)(f.read())
                            if (sync >= NUM_ROBOT):
                                break
                    federation(NUM_ROBOT, current_time, i)
                else:
                    cont = 0
                    while(True):
                        cont += 1
                        env.pushSimulation()
                        cur = 1
                        if cont % 10 == 0:
                            with open('fed_control.txt', 'r') as f:
                                cur = (int)(f.read())
                            if cur == 0:
                                break
                agent.load(i)

    else:
        raise NameError("mode wrong!!!")
