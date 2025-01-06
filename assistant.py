import os
import wave
import tempfile
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import warnings

# Suppress ALSA errors
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
assistant_id = os.getenv("ASSISTANT_ID")
thread_id = os.getenv("THREAD_ID")

def transcribe_audio(audio_data):
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_name = temp_wav.name
        with wave.open(temp_wav_name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(44100)  # Sample rate
            wav_file.writeframes(audio_data)
    
    try:
        # Open and transcribe the temporary file
        with open(temp_wav_name, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcript.text.strip()
    finally:
        # Ensure the temporary file is removed
        os.remove(temp_wav_name)

def get_assistant_response(message):
    if not message:
        return "I'm sorry, I couldn't hear anything. Could you please speak again?"

    # Send message to the assistant
    thread = client.beta.threads.retrieve(thread_id)
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message
    )
    
    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )
    
    # Wait for the run to complete
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    
    # Get the latest message from the thread
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    return messages.data[0].content[0].text.value

def text_to_speech(text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )
    
    # Save the audio to a file using the streaming approach
    with open("response.mp3", "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
    
    try:
        # Play the audio
        data, fs = sf.read("response.mp3")
        sd.play(data, fs)
        sd.wait()
    except sd.PortAudioError as e:
        print(f"Error playing audio: {e}")
        print("Trying alternative playback method...")
        os.system(f"mpg123 response.mp3")  # Make sure mpg123 is installed
    finally:
        # Remove the temporary audio file
        os.remove("response.mp3")

def run_assistant():
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
        
        print("Transcribing...")
        audio_data = audio.get_wav_data()
        transcription = transcribe_audio(audio_data)
        
        if not transcription:
            print("No speech detected. Please try again.")
            return
        
        print(f"You said: {transcription}")
        
        print("Getting response from assistant...")
        response = get_assistant_response(transcription)
        print(f"Assistant: {response}")
        
        print("Speaking response...")
        text_to_speech(response)
        
    except Exception as e:
        print(f"An error occurred: {e}")