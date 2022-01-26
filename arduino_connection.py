import serial
import time

port = '/dev/ttyACM0'
data = serial.Serial(port, baudrate=9600, bytesize=serial.EIGHTBITS)

i = str(0)

time.sleep(1)
data.write(i.encode())

data.close()
