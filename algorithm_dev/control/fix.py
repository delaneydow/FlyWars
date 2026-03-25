data = open('laser_gpio_unix.py', 'rb').read()
data = data.replace(b'\xef\xbb\xbf', b'').lstrip(b'\n\r')
open('laser_gpio_unix.py', 'wb').write(data)
