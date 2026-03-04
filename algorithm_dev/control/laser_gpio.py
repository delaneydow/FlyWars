# laser_gpio.py

from machine import Pin, PWM
import time
import sys


# CONFIG

LASER_PIN = 4
PWM_FREQ = 500          # Hz
FIRE_DUTY = 32768       # 50% duty (0–65535)
MIN_FIRE_TIME = 0.25    # seconds


# SETUP PWM ONCE
laser_pwm = PWM(Pin(LASER_PIN))
laser_pwm.freq(PWM_FREQ)
laser_pwm.duty_u16(0)   # laser OFF

firing = False


def fire_laser():
    global firing

    if firing:
        return

    firing = True

    # turn laser ON
    laser_pwm.duty_u16(FIRE_DUTY)

    time.sleep(MIN_FIRE_TIME)

    # turn laser OFF
    laser_pwm.duty_u16(0)

    firing = False


# SERIAL LOOP
while True:

    cmd = sys.stdin.readline()

    if not cmd:
        continue

    cmd = cmd.strip()

    if cmd == "FIRE":
        fire_laser()
        sys.stdout.write("OK\n")
        sys.stdout.flush()

    elif cmd.startswith("DUTY"):
        # example: DUTY 20000
        try:
            _, val = cmd.split()
            laser_pwm.duty_u16(int(val))
        except:
            pass

    elif cmd.startswith("FREQ"):
        # example: FREQ 1000
        try:
            _, val = cmd.split()
            laser_pwm.freq(int(val))
        except:
            pass

