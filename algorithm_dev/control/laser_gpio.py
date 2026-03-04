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


def fire_laser():
    while True:
        cmd = sys.stdin.readline()
        if not cmd:
            continue
        cmd = cmd.strip()
        if cmd == "FIRE":
            laser_pwm.duty_u16(FIRE_DUTY)
            sys.stdout.write("OK\n")
            sys.stdout.flush()
        elif cmd == "OFF":
            laser_pwm.duty_u16(0)
            sys.stdout.write("OK\n")
            sys.stdout.flush()
        elif cmd.startswith("DUTY"):
            try:
                _, val = cmd.split()
                laser_pwm.duty_u16(int(val))
            except:
                pass




