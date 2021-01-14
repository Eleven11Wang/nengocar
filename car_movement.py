import time
import HiwonderSDK.Board as Board
def car_act_not_found():
    step =0
    cnt = 0 
    while(cnt < 10):
        if step == 0:
            Board.setMotor(1, -60)
            Board.setMotor(2, 60)
            time.sleep(0.1)
            Board.setMotor(1, 0)
            Board.setMotor(2, 0)
            step = 1
        elif step == 1:
            Board.setMotor(1, 60)
            Board.setMotor(2, -60)
            time.sleep(0.1)
            Board.setMotor(1, 0)
            Board.setMotor(2, 0)
            step = 0
        cnt +=1 
def car_act_found():

    step = 0
    cnt = 0
    while cnt < 10:
        if step == 0:
            Board.setPWMServoPulse(1, 1200, 200)
            time.sleep(0.1)
            step = 1
        elif step == 1:
            Board.setPWMServoPulse(1, 1800, 200)
            step = 0
            time.sleep(0.1)
        cnt +=1
    Board.setPWMServoPulse(1, 1500, 100)
