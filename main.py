# Imports
import pyaudio
import numpy as np
from openwakeword.model import Model
from playsound import playsound
import os
import time
from datetime import datetime, timedelta

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained "hey jarvis" model
model_path = "./test_venv/lib/python3.9/site-packages/openwakeword/resources/models/hey_jarvis_v0.1.tflite"
owwModel = Model(wakeword_models=[model_path], inference_framework='tflite')

# Define cooldown time
cooldown_time = 2  # Change this to your desired cooldown in seconds

# Initialize last wakeword detection time outside the loop
last_wakeword_detected = datetime.now() - timedelta(seconds=cooldown_time)  

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    while True:
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        # Check if "hey jarvis" wakeword is detected
        if owwModel.prediction_buffer["hey_jarvis_v0.1"][-1] > 0.5:
            current_time = datetime.now()
            if (current_time - last_wakeword_detected).total_seconds() >= cooldown_time:
                print("Wakeword detected")
                wakeword_detected = True
                
                # Play the detected.mp3 sound
                sound_file = os.path.join('sounds', 'detected.mp3')
                playsound(sound_file)
                last_wakeword_detected = current_time  # Update last detection time
            else:
                print("Wakeword suppressed (in cooldown)")
        else:
            wakeword_detected = False

        # Clear the audio buffer if wakeword was detected in the previous iteration
        if wakeword_detected:
            owwModel.prediction_buffer["hey_jarvis_v0.1"].clear()
