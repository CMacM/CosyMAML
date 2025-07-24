import time
import os

time.sleep(10)

# list environment variables
for key, value in os.environ.items():
    print(f'{key}: {value}')

time.sleep(30)