### Overview

The **ZWZ Audio Similarity Analysis** project is designed to compare audio files recorded by different devices and evaluate their similarity to a ground truth audio file. This is achieved using various audio features such as MFCC, spectral contrast, and wavelet transform, and calculating similarity scores through Dynamic Time Warping (DTW) and Euclidean distance methods.

This repository contains Python scripts for loading audio files, performing pre-processing (such as normalization and alignment), extracting features, and computing similarity scores between test audio recordings and the ground truth.

### Data

The dataset used for this project is called **ZWZ_**, which consists of multiple audio files recorded by different devices, including ground truth recordings for comparison.

- **Dataset Structure is similar to this:**
  - `ZWZ_Adele_1.wav`: Test audio 1.
  - `ZWZ_Adele_2.wav`: Test audio 2.
  - `ZWZ_Adele_3.wav`: Test audio 3.
  - `ZWZ_Adele_groundtruth.wav`: Ground truth audio for comparison.

### Features Used

- **MFCC (Mel Frequency Cepstral Coefficients):** Extracts spectral features that mimic human auditory perception.
- **Spectral Contrast:** Measures the difference in amplitude between peaks and valleys in the frequency spectrum.
- **Wavelet Transform:** Captures time-frequency details using a multi-resolution analysis of the audio signal.

### Similarity Calculation

The similarity between test audios and the ground truth is computed using the following methods:

- **Dynamic Time Warping (DTW):** Aligns audio sequences of different lengths by minimizing the time distortion required to match one sequence with another.
- **Euclidean Distance:** Measures straight-line distance between two vectors in feature space.

### How to Use

#### Prerequisites

Before running the project, ensure you have the necessary dependencies installed. You can install them using `pip` or `conda`. (`conda-forge`)

List of required Python packages:
- `librosa`
- `numpy`
- `scipy`
- `fastdtw`
- `pywavelets`

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
├── requirements.txt                  # List of dependencies
├── README.md                         # Project documentation
└── results/                          # Directory for storing output and result files
```

#### Running the Script

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

#### Analyzing Results

The similarity scores give insight into how close the recordings from different devices are to the ground truth. A lower score indicates a higher similarity to the reference audio.

The repository also provides a visualization of the similarity scores across different scenarios to help understand the comparison more clearly.

### Customization

- Modify the `WEIGHTS` dictionary in the script to adjust the contribution of each feature (MFCC, spectral contrast, wavelet) in the final similarity score.
- Add or remove feature extraction methods as per your needs.

### Contribution

Feel free to fork this repository, open issues, or submit pull requests to improve the project. Contributions are welcome!

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

