# -*- coding: utf-8 -*-
#imports
import serial
import glob
import time

#define devices
serial_dev=glob.glob('/dev/ttyACM*') #looked up where the control was registered to 
if not serial_dev: 
    print("No devices found.")
    exit(1)
device_path=serial_dev[0] #only one, take first element
ser=serial.Serial(device_path, 115200, timeout=1)
time.sleep(2)

#command testing
commands= ['STATUS\r\n', 'GETID\r\n', 'ACKNOWLEDGE\r\n'] #defaults from manual
for cmd in commands:
    print("Sending:", repr(cmd))
    ser.write(cmd.encode())
    time.sleep(0.5) #wait between commands
    response=ser.readline()
    print("Response:", response.decode('utf-8').strip())
ser.close()
    

