import sys
if "." not in sys.path:
    sys.path.append(".")


import motor_brain
import nengo
import numpy as np
import HiwonderSDK.Board as Board
import time
def go_straight():
    Board.setMotor(1,50)
    Board.setMotor(2,50)
def turn_left():
    Board.setMotor(1,0)
    Board.setMotor(2,50)
def turn_right():
    Board.setMotor(1,50)
    Board.setMotor(2,0)
def set_stop():
    Board.setMotor(1,0)
    Board.setMotor(2,0)

    
class Delay:
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((timesteps, dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        return self.history[0]
    
    
def make_sensor(rt_val):
    def update(t):
        left_signal_rt,right_signal_rt = rt_val[0],rt_val[1]
        if (t * 10) % 10 > 3:
            left_signal,right_signal =0,0
        else:
            left_signal,right_signal=left_signal_rt,right_signal_rt
        # if (t>1):
        #     if (t*10) % 10 > 3:
        #         left_signal,right_signal =0,0
        #     else:
        #         right_signal = 1
        # elif (t< 1):
        #
        #     if (t * 10) % 10 >3 :
        #         left_signal, right_signal = 0, 0
        #     else:
        #         left_right = 1
        #print(left_signal,right_signal)
        return (left_signal, right_signal)
    return nengo.Node(update)

def car_movement():
    def update(t, x):
        print(x)
        leftv = 0
        rightv = 0
        if (x < -0.3):
            leftv = 50
        

        elif (x > 0.3):
            rightv = 54
        else:
            leftv =50
            rightv = 54
        if int(time.time()*10)%2==1:
            leftv =0
            rightv = 0
        Board.setMotor(1, leftv)
        Board.setMotor(2, rightv)

        return x

    return nengo.Node(update, size_in=1)

def make_movement(rt_val):
    model = nengo.Network()
    delay = Delay(1, timesteps=int(0.2 / 0.001))
    with model:
        brain = motor_brain.MothBrainNengo(noise=0, inhib=0, N=100)
        sensor = make_sensor(rt_val)
        dx_movement = car_movement()
        movement = nengo.Node(size_in=1)
        delayedmovement = nengo.Node(size_in = 1)
        nengo.Connection(sensor[0], brain.inputL, transform=1, synapse=None)
        nengo.Connection(sensor[1], brain.inputR, transform=1, synapse=None)


        delaynode = nengo.Node(delay.step, size_in=1, size_out=1)
        nengo.Connection(brain.turn, delaynode,synapse=0.03)
        nengo.Connection(delaynode,delayedmovement, transform = -1)

        nengo.Connection(brain.turn, movement, synapse=0.03)
        nengo.Connection(movement, dx_movement)
        nengo.Connection(delayedmovement, dx_movement)
#         turn_probe = nengo.Probe(brain.turn, synapse=0.03)  # )
#         rotation_probe = nengo.Probe(movement,synapse = None)
#         rotation_probe_delayed = nengo.Probe(delayedmovement, synapse =None)
#         dx_probe= nengo.Probe(dx_movement,synapse=None)
    sim = nengo.Simulator(model, progress_bar=False)
    sim.run(0.5, progress_bar=True)
#


