import time
import wave
import pyaudio
from wake_word import listen_for_wake_word, reset_wake_word_detection, print_available_models, flush_audio_buffer
from assistant import run_assistant
import atexit
import sounddevice as sd

def play_activation_sound():
    wf = wave.open('audio/activation.wav', 'rb')
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    chunk_size = 1024
    data = wf.readframes(chunk_size)

    while data:
        stream.write(data)
        data = wf.readframes(chunk_size)

    stream.stop_stream()
    stream.close()
    p.terminate()

def cleanup():
    try:
        sd.stop()
    except:
        pass

atexit.register(cleanup)

def main():
    print_available_models()  # Print available wake word models
    try:
        while True:
            if listen_for_wake_word():
                print("Wake word detected!")
                play_activation_sound()
                time.sleep(0.33)  # Reduced wait time to 1/3 of original
                print("Running assistant...")
                run_assistant()
                print("Assistant interaction complete. Resetting wake word detection.")
                reset_wake_word_detection()
                flush_audio_buffer()
            else:
                # If listen_for_wake_word returns False, it means it was interrupted
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("\\nExiting...")

if __name__ == "__main__":
    main()