# laser_gpio.py

from machine import Pin, PWM, Timer
import sys


# CONFIG

LASER_PIN = 4
PWM_FREQ = 500          # Hz
FIRE_DUTY = 65535       # 100% duty (0–65535)


# SETUP PWM ONCE
laser_pwm = PWM(Pin(LASER_PIN))
laser_pwm.freq(PWM_FREQ)
laser_pwm.duty_u16(0)   # laser OFF

_timer = Timer(-1) #software timer

def _off_cb(t):
    laser_pwm.duty_u16(0)
    sys.stdout.write("DONE\n")



while True:
    cmd = sys.stdin.readline()
    if not cmd:
        continue
    cmd = cmd.strip()
    if cmd == "FIRE":
        laser_pwm.duty_u16(FIRE_DUTY)
        sys.stdout.write("OK\n")
    elif cmd.startswith("FIRE "):
        # fire for exactly N millseconds
        try: 
            ms = int(cmd.split()[1])
            _timer.deinit() # cancel any prior timer
            laser_pwm.duty_u16(FIRE_DUTY)
            sys.stdout.write("OK\n")
            _timer.init(mode=Timer.ONE_SHOT, period=ms, callback=_off_cb)
        except Exception:
            pass 
    elif cmd == "OFF":
        _timer.deinit()
        laser_pwm.duty_u16(0)
        sys.stdout.write("OK\n")
    elif cmd.startswith("DUTY "): 
        try:
            _, val = cmd.split()
            laser_pwm.duty_u16(int(val))
        except:
            pass




