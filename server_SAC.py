import socketio
from aiohttp import web
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_D, MoveTank
from ev3dev2.sensor import INPUT_2
from ev3dev2.sensor.lego import GyroSensor
import sys 
import threading 
from time import sleep 
#netstat -nap | grep 5000    running program check 
# fuser -k -n tcp 5000      port kill
sio = socketio.AsyncServer(async_mode='aiohttp',ping_timeout=1500000, ping_interval=600000)
app = web.Application()
sio.attach(app)

gy = GyroSensor(INPUT_2)
gy._ensure_mode(gy.MODE_GYRO_CAL)
gy._direct = gy.set_attr_raw(gy._direct, 'direct', bytes(17,))
gy._ensure_mode(gy.MODE_GYRO_G_A)
gy.reset()
motors = MoveTank(OUTPUT_A, OUTPUT_D)
#motors.on(50,50)
sleep(0.2)
motors.off(brake=False)

@sio.event
async def connect(sid, environ):
    #raise ConnectionRefusedError('authentication failed')
    print('connected with Client', sid)

steps = 0 

@sio.event
async def motor_on(sid, run):    
    #NdimArrs :[arbitrary N-dim array of random multivariate_normal distribution for 20 samples]
    #gyro estimation start by thread
    #sensor reset.  gy.reset() > error exists
    #print(run)

    global steps 
    steps += 1 
    if steps == 1 :

        print("wait untill push")
        while(1) :
            sleep(0.05)
            if abs(gy.angle_and_rate[1]) > 10 :
                break
            else: continue 
    while(True):
        st = gy.angle_and_rate
        if st[1]<4:
            
            sleep(run[0])
            motors.on(-70,-70)
            sleep(0.15)
            motors.on(100,100)
            sleep(0.15)

    if steps > 32 : 
        while(True):
            print("wait")
            sleep(3)
            gy.reset()
            morots.off(brake=False)
            temp = gy.angle_and_rate
            if abs(temp[0]) < 5 : 
            
                sleep(3)
                temp = gy.angle_and_rate
                if abs(temp[1])<5:
                    sleep(5)
                    print("swing stops")
                    gy._ensure_mode(gy.MODE_GYRO_G_A)
                    gy.reset()
                    print("ang&rate initialized : ", gy.angle_and_rate)
                    break

        steps = 0 
        await sio.emit("init", 1, room=sid)
               
    else:
        await sio.emit("resp", [st[0],st[1]], room=sid)
    #prior ang, prior rate, post ang, rate, action, steps
        

@sio.event
async def disconnect(sid):
    motors.off(brake=False)
    print('disconnected with Client ', sid)

if __name__ == '__main__':
    web.run_app(app)
    



    