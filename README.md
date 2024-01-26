# WEEK 1 
## a) Intensity Transformations

```python
import cv2
fname=input("Enter the path of the image to display: ")
img_arr=cv2.imread(fname)
cv2.imshow("Displaying Image",img_arr)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## b) Gray Scale Image

```python
import cv2
fname=input("Enter the path of the image to display: ")
img_arr=cv2.imread(fname)
grayscale_img_arr=cv2.cvtColor(img_arr,cv2.COLOR_BGR2GRAY)
cv2.imshow("GrayScale Image",grayscale_img_arr)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## c) Display the Size of the Image

```python
import cv2
fname=input("Enter the path of the image to display: ")
img_arr=cv2.imread(fname)
height,width,channel=img_arr.shape
print(f"Size of the image : {height}x{width}x{channel}")
```

# WEEK 2
## a) Read an Image and Perform Arithmetic Operations on Images - ADDITION

```python
import cv2

fname = input("Enter the path of the image - 1: ")
img_arr = cv2.imread(fname)
cv2.imshow("Image - 1", img_arr)
fname2 = input("Enter the path of the image - 2: ")
img_arr_2 = cv2.imread(fname2)

img_arr_2 = cv2.resize(img_arr_2, img_arr.shape[:2][::-1])
cv2.imshow("Image - 2 after resizing", img_arr_2)
add_arr = cv2.add(img_arr, img_arr_2)
cv2.imshow("Image - 1 + Image - 2", add_arr)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## b) Read an Image and Perform Arithmetic Operations on Images - Subtraction

```python
import cv2

fname = input("Enter the path of the image - 1: ")
img_arr = cv2.imread(fname)
cv2.imshow("Image - 1", img_arr)
fname2 = input("Enter the path of the image - 2: ")
img_arr_2 = cv2.imread(fname2)
img_arr_2 = cv2.resize(img_arr_2, img_arr.shape[:2][::-1])
cv2.imshow("Image - 2 after resizing", img_arr_2)
sub_arr_1 = cv2.subtract(img_arr, img_arr_2)
cv2.imshow("Image - 1 - Image - 2", sub_arr_1)
sub_arr_2 = cv2.subtract(img_arr_2, img_arr)
cv2.imshow("Image - 2 - Image - 1", sub_arr_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# WEEK 3
## a) Plot a Histogram of an Image

```python
import cv2
import matplotlib.pyplot as plt

fname = input("Enter the path of the image - 1: ")
img_arr = cv2.imread(fname)
hist_data = cv2.calcHist([img_arr], [0], None, [256], [0, 256])
plt.plot(hist_data)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## b) Use Histogram Equalization for Contrast Adjustment. Display the Original Image and the Histogram Equalization by Stacking Side by Side

```python
import cv2
import numpy as np

fname = input("Enter the path of the image - 1: ")
img_arr = cv2.imread(fname, 0)

equalized_img = cv2.equalizeHist(img_arr)

stack = np.hstack((img_arr, equalized_img))
cv2.imshow("Original | Histogram Equalization", stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# WEEK 4
## a) Demonstrate Various Intensity Transformations Operations on Images such as
### 1. Image Negatives (Linear)

```python
import cv2
import numpy as np

fname = input("Enter the path of the image - 1: ")
img_arr = cv2.imread(fname)
dim1, dim2, channels = img_arr.shape
negative_img_arr = 255 - img_arr
stack = np.hstack((img_arr, negative_img_arr))
cv2.imshow("Image | Image Negative", stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. Log Transformations

```python
import cv2
import numpy as np

fname = input("Enter the path of the image - 1: ")
img_arr = cv2.imread(fname)
dim1, dim2, channels = img_arr.shape

scaling_constant = 255 / np.log(1 + np.max(img_arr))
log_img_arr = scaling_constant * np.log(img_arr + 1)
log_img_arr = np.array(log_img_arr, dtype=np.uint8)
stack = np.hstack((img_arr, log_img_arr))

cv2.imshow("Image | Log Image", stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 3. Power-Law (Gamma) Transformations

```python
import cv2
import numpy as np

fname = input("Enter the path of the image: ")
img_arr = cv2.imread(fname)
dim1, dim2, channels = img_arr.shape

for gamma in (0.1, 0.4, 0.5, 1.0, 1.2, 2.0):
    gamma_img = np.array(255 * (img_arr / 255) ** gamma, dtype=np.uint8)
cv2.imshow("Gamma: {}".format(gamma), gamma_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## b) Apply Piece-wise Linear Transformation for Enhancing Contrast of an Image

```python
import cv2
import numpy as np

fname = input("Enter the path of the image: ")
img_arr = cv2.imread(fname, 0) #grayscale

def contrast_stretch(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1) * pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

contrast_stretch = np.vectorize(contrast_stretch)

r1, s1, r2, s2 = 70, 0, 140, 255
contrast_stretched_img = contrast_stretch(img_arr, r1, s1, r2, s2)
stack = np.hstack((img_arr, contrast_stretched_img))
cv2.imshow("Image | Contrast Strectched", stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# WEEK 5
## a) Use the following Smoothing Spatial Filters for Reduction of Blur and Noise in the Image
### 1. Linear Filter (Mean Filter)

```python
import cv2
import numpy as np

fname = input("Enter the path of the image: ")
img_arr = cv2.imread(fname, 0)
img_arr = np.array(img_arr, dtype=np.uint8)
cv2.imshow("Noisy Image", img_arr)

for kernel_size in range(3, 30, 3):
    blurred_noisy_img = cv2.blur(img_arr, (kernel_size, kernel_size))

blurred_noisy_img = np.array(blurred_noisy_img, dtype=np.uint8)
cv2.imshow("Kenel size: {}".format(kernel_size), blurred_noisy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. Order Statistics (Non-linear) Filter

```python
import cv2
import numpy as np

fname = input("Enter the path of the image: ")
img_arr = cv2.imread(fname, 0)
cv2.imshow("Noisy Image", img_arr)

for kernel_size in range(3, 30, 3):
    if kernel_size:
        median_blur_noisy_img = cv2.medianBlur(img_arr, kernel_size)
        cv2.imshow("Kernel Size: {}".format(kernel_size), median_blur_noisy_img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
```

# WEEK 6
## a) Loading and Displaying Characteristics of an Audio fileLoading and Displaying Characteristics of an Audio File

```python
import librosa

fpath = input("Enter path of the audio file: ")
samples, sampling_rate = librosa.load(fpath, sr=None, mono=False)

print("Sampling rate:", sampling_rate)
print("Number of samples:", samples.shape[-1])
n_channels = samples.shape[-2] if len(samples.shape) != 1 else 1
print("Number of channels:", n_channels)
duration = samples.shape[-1] / sampling_rate
print("Duration:", duration, "s")
```

## b) Spectral Representations

```python
import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np

fpath = input("Enter path of the audio file: ")
samples, sampling_rate = librosa.load(fpath, sr=None, mono=True)

signal_stft = librosa.stft(samples)
signal_db = librosa.amplitude_to_db(np.abs(signal_stft), ref=np.max)

fig, ax = plt.subplots()
img = display.specshow(signal_db, x_axis="time", y_axis="linear", ax=ax)
ax.set(title="Spectrogram of {}".format(fpath))
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()
```

## c) Feature Extraction and Manipulation

```python
import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np

fpath = input("Enter path of the audio file: ")
samples, sampling_rate = librosa.load(fpath, sr=None, mono=True, duration=2.0)

signal_stft = librosa.stft(samples)
signal_db = librosa.amplitude_to_db(np.abs(signal_stft), ref=np.max)

spec_cent = librosa.feature.spectral_centroid(y=samples, sr=sampling_rate)
spec_cent_times_like = librosa.times_like(spec_cent, sr=sampling_rate)

spec_contrast = librosa.feature.spectral_contrast(S=np.abs(signal_stft), sr=sampling_rate)

spec_flatness = librosa.feature.spectral_flatness(y=samples)
spec_flatness_times_like = librosa.times_like(spec_flatness, sr=sampling_rate)

spec_rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sampling_rate, roll_percent=0.85)
spec_rolloff_times_like = librosa.times_like(spec_rolloff, sr=sampling_rate)

spec_bw = librosa.feature.spectral_bandwidth(y=samples, sr=sampling_rate, p=2)
spec_bw_times_like = librosa.times_like(spec_bw, sr=sampling_rate)

mel_spec = librosa.feature.melspectrogram(y=samples, sr=sampling_rate)

rms_frames = librosa.feature.rms(y=samples)
rms_frames_times_like = librosa.times_like(rms_frames, sr=sampling_rate)

fig, ax = plt.subplots(nrows=2, ncols=2)
normal_spec_img = librosa.display.specshow(signal_db, x_axis="time", y_axis="linear", ax=ax[1, 1])
fig.colorbar(normal_spec_img, ax=[ax[1, 1]], format="%+2.0f dB")
ax[1, 1].set(title="Audio Spectrogram")

ax[0, 1].plot(spec_cent_times_like, spec_cent.T, label="Spectral Centroid")
spec_cont_img = librosa.display.specshow(spec_contrast, x_axis='time', ax=ax[0, 0])
fig.colorbar(spec_cont_img, ax=[ax[0, 0]])
ax[0, 0].set(ylabel='Frequency bands', title='Spectral Contrast')

ax[0, 1].plot(spec_flatness_times_like, spec_flatness.T, label="Spectral Flatness")
ax[0, 1].plot(spec_rolloff_times_like, spec_rolloff.T, label="Spectral Rolloff (85%)")
ax[0, 1].plot(spec_bw_times_like, spec_bw.T, label="Spectral Bandwidth (Order - 2)")

mel_spec_img = librosa.display.specshow(mel_spec, x_axis="time", y_axis="log", ax=ax[1, 0])
fig.colorbar(mel_spec_img, ax=[ax[1, 0]])
ax[1, 0].set(title="Mel Spectrogram")

ax[0, 1].semilogy(rms_frames_times_like, rms_frames[0], label="RMS Energy")
ax[0, 1].legend(loc='lower right')
ax[0, 1].set(xlabel="Time", title="Other Features")

plt.show()
```

## d) Visualize Audio

```python
import librosa
from librosa import display
import matplotlib.pyplot as plt

fpath = input("Enter path of the audio file: ")
samples_mono, sr_mono = librosa.load(fpath, sr=None, mono=True)
samples_multi, sr_multi = librosa.load(fpath, sr=None, mono=False)

plt.subplot(2, 1, 1)
librosa.display.waveshow(y=samples_mono, sr=sr_mono)
plt.title("Mono")

plt.subplot(2, 1, 2)
librosa.display.waveshow(y=samples_multi, sr=sr_multi)
plt.title("Multi (here stereo)")

plt.show()
```
