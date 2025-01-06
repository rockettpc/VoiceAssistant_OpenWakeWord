import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pyaudio
import numpy as np
from openwakeword.model import Model
import time

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

# Load pre-trained openwakeword models
owwModel = Model()

# Detection settings
DETECTION_THRESHOLD = 0.5  # Lowered from 0.7
DOUBLE_CHECK_THRESHOLD = 0.4  # Lowered from 0.6
NOISE_FLOOR_THRESHOLD = 300  # Lowered from 500

def calculate_noise_floor(audio_data):
    return np.abs(audio_data).mean()

def listen_for_wake_word():
    p_audio = pyaudio.PyAudio()
    mic_stream = p_audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening for wake word... (Press Ctrl+C to stop)")

    try:
        while True:
            audio = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            
            noise_floor = calculate_noise_floor(audio)
            print(f"Noise floor: {noise_floor}")  # Debug information

            prediction = owwModel.predict(audio)

            for mdl in owwModel.prediction_buffer.keys():
                scores = list(owwModel.prediction_buffer[mdl])
                print(f"Wake word score: {scores[-1]}")  # Debug information
                if scores[-1] > DETECTION_THRESHOLD:
                    # Double check
                    time.sleep(0.1)  # Short pause before second check
                    audio = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                    prediction = owwModel.predict(audio)
                    second_score = list(owwModel.prediction_buffer[mdl])[-1]
                    print(f"Second check score: {second_score}")  # Debug information
                    if second_score > DOUBLE_CHECK_THRESHOLD:
                        return True
            
            time.sleep(0.01)  # Small delay to prevent CPU overuse

    except KeyboardInterrupt:
        print("\\nStopping wake word detection...")
        return False
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        p_audio.terminate()

def reset_wake_word_detection():
    global owwModel
    owwModel = Model()  # Recreate the model to fully reset its state
    print("Wake word detection reset.")

def print_available_models():
    print("Available wake word models:")
    for model in owwModel.models.keys():
        print(f"- {model}")

def flush_audio_buffer():
    p_audio = pyaudio.PyAudio()
    mic_stream = p_audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    # Read and discard audio data for a short period
    for _ in range(10):  # Adjust this number if needed
        mic_stream.read(CHUNK, exception_on_overflow=False)
    
    mic_stream.stop_stream()
    mic_stream.close()
    p_audio.terminate()
    print("Audio buffer flushed.")