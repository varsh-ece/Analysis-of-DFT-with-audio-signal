
# EXP 1 :  ANALYSIS OF DFT WITH AUDIO SIGNAL

# AIM: 

  To analyze audio signal by removing unwanted frequency. 

# APPARATUS REQUIRED: 
   
   PC installed with SCILAB/Python. 

# PROGRAM: 
```
# ==============================
# AUDIO DFT ANALYSIS IN COLAB
# ==============================

# Step 1: Install required packages
!pip install -q librosa soundfile

# Step 2: Upload audio file
from google.colab import files
uploaded = files.upload()   # choose your .wav / .mp3 / .flac file
filename = next(iter(uploaded.keys()))
print("Uploaded:", filename)

# Step 3: Load audio
import librosa, librosa.display
import numpy as np
import soundfile as sf

y, sr = librosa.load(filename, sr=None, mono=True)  # keep original sample rate
duration = len(y) / sr
print(f"Sample rate = {sr} Hz, duration = {duration:.2f} s, samples = {len(y)}")

# Step 4: Play audio
from IPython.display import Audio, display
display(Audio(y, rate=sr))

# Step 5: Full FFT (DFT) analysis
import matplotlib.pyplot as plt

n_fft = 2**14   # choose large power of 2 for smoother spectrum
Y = np.fft.rfft(y, n=n_fft)
freqs = np.fft.rfftfreq(n_fft, 1/sr)
magnitude = np.abs(Y)

plt.figure(figsize=(12,4))
plt.plot(freqs, magnitude)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Magnitude Spectrum (linear scale)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.semilogy(freqs, magnitude+1e-12)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (log scale)")
plt.title("FFT Magnitude Spectrum (log scale)")
plt.grid(True)
plt.show()

# Step 6: Top 10 dominant frequencies
N = 10
idx = np.argsort(magnitude)[-N:][::-1]
print("\nTop 10 Dominant Frequencies:")
for i, k in enumerate(idx):
    print(f"{i+1:2d}. {freqs[k]:8.2f} Hz  (Magnitude = {magnitude[k]:.2e})")

# Step 7: Spectrogram (STFT)
n_fft = 2048
hop_length = n_fft // 4
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12,5))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.ylim(0, sr/2)
plt.show()
```

# OUTPUT: 

Top 10 Dominant Frequencies:

69.98 Hz (Magnitude = 6.79e+00)

282.62 Hz (Magnitude = 5.87e+00)

258.40 Hz (Magnitude = 5.65e+00)

67.29 Hz (Magnitude = 5.61e+00)

301.46 Hz (Magnitude = 5.03e+00)

261.09 Hz (Magnitude = 4.82e+00)

279.93 Hz (Magnitude = 4.69e+00)

298.77 Hz (Magnitude = 4.41e+00)

253.02 Hz (Magnitude = 4.36e+00)

304.16 Hz (Magnitude = 4.30e+00)


<img width="1058" height="414" alt="image" src="https://github.com/user-attachments/assets/4a7b8861-b84b-4ca6-8a8e-1a8f60e47597" />


<img width="1055" height="412" alt="image" src="https://github.com/user-attachments/assets/fa2905ce-6a1b-4ea8-8d5c-68870d587244" />


<img width="1032" height="521" alt="image" src="https://github.com/user-attachments/assets/9a2e1dbb-9e1a-4c54-b377-aeb4b3841082" />



# RESULTS

Thus,the analysis of DFT with audio signal is verified
