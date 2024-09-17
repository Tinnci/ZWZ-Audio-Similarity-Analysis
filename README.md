## Overview

The **ZWZ Audio Similarity Analysis** project is designed to compare audio files recorded by different devices and evaluate their similarity to a ground truth audio file. This is achieved using various audio features such as MFCC, spectral contrast, and wavelet transform, and calculating similarity scores through Dynamic Time Warping (DTW) and Euclidean distance methods.

This repository contains Python scripts for loading audio files, performing pre-processing (such as normalization and alignment), extracting features, and computing similarity scores between test audio recordings and the ground truth.

------

### Data

The dataset used for this project is called **ZWZ_**, which consists of multiple audio files recorded by different devices, including ground truth recordings for comparison.

Ground truth is the highest quality audio stream snippets retrieved from AM. Using a Samsung android device as Bluetooth A2dp audio source, a pipewire ubuntu device as A2dp sink, then Bluetooth PCM stream being captured use pipewire build-in capture tools. The ALAC source uses 16/24 bit 44.1Khz sampling rate, so the audio stream will go through a SRC progress using android's audio stack. The audio is encoded at 48Khz using the AAC codec from android and decoded using the codec form pipewire, which might result in difference from the target process (the audio SRC processing mechanism and codec on iOS/earpods might not be the same, because it might avoid the SRC process by utilizing AAC sample rate switching whereas android A2dp codec doesn't support).

Audio snippets are being retrieved from audio track: (Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 78 kb/s (default)), processed using RX 10 editor in 32bit float and 48Khz, I can only assure that no loss is introduced during limited processing, that is, cut and connect these snippets back.

- **Dataset Structure is similar to this:**
  - `ZWZ_Adele_1.wav`: Test audio 1.
  - `ZWZ_Adele_2.wav`: Test audio 2.
  - `ZWZ_Adele_3.wav`: Test audio 3.
  - `ZWZ_Adele_groundtruth.wav`: Ground truth audio for comparison.

------

### Features Used

- **MFCC (Mel Frequency Cepstral Coefficients):** Extracts spectral features that mimic human auditory perception.
- **Spectral Contrast:** Measures the difference in amplitude between peaks and valleys in the frequency spectrum.
- **Wavelet Transform:** Captures time-frequency details using a multi-resolution analysis of the audio signal.

### Similarity Calculation

The similarity between test audios and the ground truth is computed using the following methods:

- **Dynamic Time Warping (DTW):** Aligns audio sequences of different lengths by minimizing the time distortion required to match one sequence with another.
- **Euclidean Distance:** Measures straight-line distance between two vectors in feature space.

------

### How to Use

#### Prerequisites

Before running the project, ensure you have the necessary dependencies installed. You can install them using `pip` or `conda`. (`conda-forge`)

List of required Python packages:
- `librosa`
- `numpy`
- `scipy`
- `fastdtw`
- `pywavelets`

------

#### Repository Structure

```
ZWZ_Audio_Similarity_Analysis/
│
├── data/                             # Directory containing the ZWZ_ audio files
│   ├── ZWZ_Adele_1.wav
│   ├── ZWZ_Adele_2.wav
│   ├── ZWZ_Adele_3.wav
│   └── ZWZ_Adele_groundtruth.wav
│
├── scripts/                          # Python scripts for feature extraction and similarity computation
│   └── cmp_adele.py                  # Main script for audio comparison
│
├── README.md                         # Project documentation
└── results                           #output and result files
.....et cetra
```

------

#### E.g. on How to Running the Script

Once the dataset is prepared, you can run the main script to compute similarity scores between the test audios and the ground truth:

```bash
python scripts/cmp_adele.py
```

The output will show the similarity score of each test audio against the ground truth, indicating which device's audio is the most similar.

Sample output:

```
--- Similarity Results ---
audio1 vs Groundtruth: 1137.6584
audio2 vs Groundtruth: 1028.9112
audio3 vs Groundtruth: 1104.5651
```

------

#### Analyzing Results

The similarity scores give insight into how close the recordings from different devices are to the ground truth. A lower score indicates a higher similarity to the reference audio.

The repository also provides a visualization of the similarity scores across different scenarios to help understand the comparison more clearly.

------

### Customization

- Modify the `WEIGHTS` dictionary in the script to adjust the contribution of each feature (MFCC, spectral contrast, wavelet) in the final similarity score.
- Add or remove feature extraction methods as per your needs.

------

### Contribution

Feel free to fork this repository, open issues, or submit pull requests to improve the project. Contributions are welcome!

------

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



------

### Extra: Determining Authenticity of a Sample

In some cases, it's important to evaluate whether an audio sample has been tampered with or altered in a way that could indicate it is not authentic. To determine if a sample might be falsified, you can apply a combination of techniques to compare the suspect audio with the ground truth.

#### 1. Unusual Similarity in SPL or Feature Matching

If a sample matches the **Sound Pressure Level (SPL)** or specific audio features (e.g., MFCC, spectral contrast) of the ground truth *too perfectly* despite being recorded with a different or lower-quality device, this could be a sign of tampering:

- **SPL Matching**: Perfect SPL alignment between a suspect sample and the ground truth, especially when using a lower-quality or less calibrated device, may indicate post-processing or manipulation to artificially match the reference.
- **Feature Similarity**: Extremely high similarity scores in multiple audio features across different recording conditions may indicate that the suspect sample was derived from the ground truth using techniques such as resampling, post-processing, or other forms of signal manipulation.

*How to Analyze*:  
Compare the SPL values, MFCC features, and spectral contrast of the suspect sample with the ground truth. If the similarity is unusually high despite known differences in recording devices or environments, this could suggest the sample has been modified to mimic the ground truth.

#### 2. Frequency-Domain Anomalies

Analyzing the frequency spectrum of the suspect sample can reveal signs of manipulation, such as:

- **Frequency Gaps or Enhancements**: Artificially boosting or reducing certain frequency ranges to mask differences between the ground truth and the suspect sample can introduce anomalies.
- **Harmonic Distortion**: Signs of non-linear editing or excessive processing may be indicated by the presence of unexpected harmonics or other spectral distortions not present in the original ground truth.

*How to Analyze*:  
Use frequency-domain tools such as STFT or spectrograms to compare the suspect sample with the ground truth. Look for unexpected patterns, such as missing high-frequency details, enhanced low-end, or unusual harmonic content that could indicate post-processing.

#### 3. Time-Domain Irregularities

If a sample has been tampered with, there may be irregularities in the waveform, such as:

- **Waveform Smoothing**: Over-processed audio may exhibit overly smooth waveforms, where transients and natural fluctuations are artificially dampened.
- **Clipping or Artifacts**: If a suspect sample has been altered through aggressive compression or normalization, it may introduce clipping or other unnatural artifacts that deviate from the original signal’s natural dynamics.

*How to Analyze*:  
Compare the time-domain waveform of the suspect sample to the ground truth. Look for unusual smoothness, clipped peaks, or other artifacts that suggest post-processing or signal manipulation.

#### 4. Dynamic Range Compression and Transient Loss

Suspicious samples may exhibit unnatural dynamic compression that differs from the expected behavior of the original recording:

- **Dynamic Range Reduction**: A falsified sample may have an unusually narrow dynamic range, with minimal differences between loud and soft sections.
- **Loss of Transient Detail**: If transients (sharp, quick sounds) like drum hits are less clear or sharp in the suspect sample, this could indicate it has been reprocessed.

*How to Analyze*:  
Compare the dynamic range and transient response of the suspect sample with the ground truth. If transients are unnaturally softened or the dynamic range is significantly reduced, it may suggest that the sample has been manipulated.

#### 5. Metadata and Contextual Clues

Examining the metadata of the audio file or understanding the context of how the sample was recorded can provide additional evidence:

- **Metadata Inconsistencies**: Check the file’s metadata for signs of editing, such as unusual timestamps, encoding details, or processing software tags that differ from what would be expected for a raw recording.
- **Contextual Analysis**: Consider the environment in which the sample was recorded. If the recording conditions should have resulted in significant differences from the ground truth, but the suspect sample closely matches the reference audio, this may indicate tampering.

*How to Analyze*:  
Check the metadata of the suspect audio file for any unusual discrepancies or unexpected information. Cross-check with known details about how the recording should have been made.

#### 6. Cross-Referencing Multiple Samples

If there are multiple suspect samples available, you can cross-reference them to detect inconsistencies:

- **Unexpected Similarities Between Suspect Samples**: If multiple suspect samples are suspiciously similar to each other or to the ground truth (but should not be), this can be an indication that they were generated from the same altered source.
- **Pattern Analysis**: Look for consistent processing artifacts, such as identical noise profiles or compression patterns, across multiple samples that suggest the same post-processing workflow was applied.

*How to Analyze*:  
Compare multiple suspect samples to each other and the ground truth. Look for patterns of similarity or shared artifacts that suggest they were processed in the same way, potentially indicating manipulation.

#### 7. Conclusion

By analyzing a suspect sample's similarity to the ground truth across time-domain, frequency-domain, and dynamic range features, as well as reviewing metadata and contextual information, you can detect signs of tampering. While high similarity alone doesn't prove falsification, combining these techniques can reveal whether a sample has been unnaturally manipulated to mimic the original audio.

