# gpio pulse test

from machine import Pin, PWM
import time

#pin number may change
laser_pin = PWM(Pin(4)) # gpio04 -- function #7 on PIN #7 which is the PWM hookup

# set PWM frequency (laser spec allows up to 5kHz)
laser_pin.freq(500) # 500 hz, safe to start

try:
    while True:
        print("low power")
        laser_pin.duty_u16(10000) #15% duty
        time.sleep(5)
        
        print("med power")
        laser_pin.duty_u16(30000) # ~45% duty
        time.sleep(5)
        
        print("off")
        laser_pin.duty_u16(0)
        time.sleep(3)

except KeyboardInterrupt:
    laser_pin.deinit()
