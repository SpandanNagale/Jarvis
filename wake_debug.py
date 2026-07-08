import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHUNK = 1280

print("Available input devices:")
print(sd.query_devices())
print(f"\nDefault input device: {sd.query_devices(kind='input')['name']}\n")

wake_model = Model(wakeword_models=["hey_jarvis"])

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK)
stream.start()

print("Listening... say 'hey jarvis' a few times and watch the numbers.")
print("Ctrl+C to stop.\n")

try:
    while True:
        chunk, _ = stream.read(CHUNK)
        chunk = chunk.flatten()
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        prediction = wake_model.predict(chunk)
        print(f"RMS: {rms:7.1f}  |  {prediction}")
except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    stream.close()