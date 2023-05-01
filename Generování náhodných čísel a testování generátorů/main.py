import os
import time
import platform
import hashlib
import threading
from pynput.keyboard import Listener as KeyboardListener

def on_key(key):
    global keyboard_data
    keyboard_data.append(str(key))

def collect_keyboard_data(duration):
    global keyboard_data
    keyboard_data = []
    with KeyboardListener(on_press=on_key) as listener:
        timer = threading.Timer(duration, listener.stop)
        timer.start()
        listener.join()

def generate_random_hash():
    seed_sources = []

    # Collect current time
    seed_sources.append(str(time.time()).encode())

    # Collect operating system information
    seed_sources.append(platform.platform().encode())

    # Collect keyboard data
    print("Please type some random keys for 5 seconds.")
    collect_keyboard_data(5)
    seed_sources.append(str(keyboard_data).encode())

    # Combine and hash seed sources
    combined_data = b''.join(seed_sources)
    random_hash = hashlib.sha256(combined_data).hexdigest()
    return random_hash

if __name__ == "__main__":
    random_hash = generate_random_hash()
    print("Random hash generated:", random_hash)
