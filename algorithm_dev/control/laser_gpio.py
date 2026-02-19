# laser_gpio.py

from machine import Pin, PWM
import time
import sys


#pin number may change
laser = PWM(Pin(4)) # gpio04 -- function #7 on PIN #7 which is the PWM hookup

# set PWM frequency (laser spec allows up to 5kHz)
laser.freq(500) # 500 hz, safe to start TODO figure out if this needs to change
laser.duty_u16(0) # laser OFF

MIN_FIRE_TIME = 0.25 # seconds

while True: 
    cmd = sys.stdin.readline().strip()

    if cmd == "FIRE":
        laser.duty_u16(32768) #50% duty
        time.sleep(MIN_FIRE_TIME)
        laser.duty_u16(0) # off 

    elif cmd.startswith("DUTY"): 
        # ex: duty 20000
        _, val = cmd.split()
        laser.duty_u16(int(val))

    elif cmd.startswith("FREQ"):
        _, val = cmd.split()
        laser.freq(int(val))
