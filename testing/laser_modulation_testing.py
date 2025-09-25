# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 17:17:11 2025

@author: laney
"""

import gpiod
import time

# === CONFIGURATION === TODO CHECK HOW THIS WORKS W/ DEFAULTS OF 20 40 60 POWER
CHIP = "tbd"         # TODO check with `gpiodetect`
LINE = TBD           # TODO GPIO line number
FREQ_HZ = TBD        # PWM frequency for test (100 Hz)
DUTY_CYCLE = 0.5     # 50% duty cycle

# === SETUP ===
chip = gpiod.Chip(CHIP)
line = chip.get_line(LINE)
line.request(consumer="laser_test", type=gpiod.LINE_REQ_DIR_OUT)

# === MAIN LOOP ===
period = 1.0 / FREQ_HZ
high_time = period * DUTY_CYCLE
low_time = period - high_time

print(f"Starting laser test on GPIO {LINE} at {FREQ_HZ} Hz, {DUTY_CYCLE*100:.0f}% duty cycle")
try:
    while True:
        line.set_value(1)   # Laser ON
        time.sleep(high_time)
        line.set_value(0)   # Laser OFF
        time.sleep(low_time)
except KeyboardInterrupt:
    print("Exiting...")
    line.set_value(0)  # make sure laser is OFF (safety/fallback)
    line.release()


