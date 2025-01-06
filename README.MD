# OpenWakeWord for Python Template

## Create python venv
```
python3 -m venv test_venv
```

## Activate venv
```
source test_venv/bin/activate
```

## Install pyaudio
```
sudo apt-get install python3-dev portaudio19-dev
pip install pyaudio
```

## Install MPG123
```
sudo apt-get install mpg123
```

## Install openwakeword version .040
```
pip install openwakeword
```

## Create .env file
```
OPENAI_API_KEY=YOUR_KEY_HERE
ASSISTANT_ID=YOUR_ASST_HERE
THREAD_ID=YOUR_THREAD_HERE
```

## Install requirements
```
pip install -r requirements.txt
```

## Test scripts
```
python main.py
```
