import librosa
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pywt
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 统一采样率
TARGET_SR = 48000

# 权重配置
WEIGHTS = {
    'mfcc': 0.4,
    'spectral_contrast': 0.4,
    'wavelet': 0.2
}

def load_audio(path, sr=TARGET_SR):
    """
    加载音频文件，统一采样率，并处理加载错误。
    """
    try:
        audio, loaded_sr = librosa.load(path, sr=sr)
        logging.info(f"成功加载 {path}，采样率: {loaded_sr} Hz")
        return audio, sr
    except Exception as e:
        logging.error(f"加载音频文件 {path} 失败: {e}")
        return None, None

def normalize_audio(audio):
    """
    使用RMS进行响度归一化。
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        logging.warning("音频信号的RMS为零，无法归一化。")
        return audio
    return audio / rms

def align_audio(audio1, audio2):
    """
    通过互相关进行时移对齐，确保audio1和audio2同步。
    返回对齐后的audio1_aligned和audio2_aligned。
    """
    correlation = np.correlate(audio1, audio2, mode='full')
    lag = np.argmax(correlation) - len(audio2) + 1
    logging.info(f"计算得到的时移量: {lag}")

    if lag > 0:
        audio1_aligned = audio1[lag:]
        audio2_aligned = audio2[:len(audio1_aligned)]
    else:
        audio1_aligned = audio1[:len(audio1)+lag]
        audio2_aligned = audio2[-lag:]
    
    min_len = min(len(audio1_aligned), len(audio2_aligned))
    return audio1_aligned[:min_len], audio2_aligned[:min_len]

def extract_mfcc(audio, sr, n_mfcc=13):
    """
    提取MFCC特征并进行标准化。
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    # 标准化
    mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
    return mfcc

def extract_spectral_contrast(audio, sr):
    """
    提取频谱对比度特征并进行标准化。
    """
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).T
    # 标准化
    spectral_contrast = (spectral_contrast - np.mean(spectral_contrast, axis=0)) / (np.std(spectral_contrast, axis=0) + 1e-8)
    return spectral_contrast

def extract_wavelet(audio, wavelet='db4', level=4):
    """
    提取小波变换特征，保留各层小波系数的均值和标准差。
    """
    coeffs = pywt.wavedec(audio, wavelet, level=level)
    features = []
    for c in coeffs:
        features.append(np.mean(c))
        features.append(np.std(c))
    return np.array(features)

def compute_dtw_distance(features1, features2):
    """
    使用DTW计算两个特征序列的距离。
    """
    distance, path = fastdtw(features1, features2, dist=euclidean)
    return distance

def compute_euclidean_distance(features1, features2):
    """
    计算两个特征向量的欧氏距离。
    """
    return euclidean(features1, features2)

def compute_composite_similarity(sim_mfcc, sim_spectral, sim_wavelet, weights=WEIGHTS):
    """
    综合不同特征的相似度得分，按权重加权。
    """
    return (weights['mfcc'] * sim_mfcc +
            weights['spectral_contrast'] * sim_spectral +
            weights['wavelet'] * sim_wavelet)

def main():
    # 定义音频文件路径
    audio_paths = {
        'audio1': 'data/ZWZ_CQ_2_1.wav',
        'audio2': 'data/ZWZ_CQ_2_2.wav',
        'audio3': 'data/ZWZ_CQ_2_3.wav',
        'groundtruth': 'data/ZWZ_CQ_1_groundtruth.wav'
    }

    # 加载音频
    audios = {}
    for key, path in audio_paths.items():
        audio, sr = load_audio(path)
        if audio is None:
            logging.error(f"音频文件 {path} 加载失败，程序终止。")
            return
        audio = normalize_audio(audio)
        audios[key] = audio

    groundtruth = audios['groundtruth']

    # 对齐音频
    aligned_audios = {}
    for key in ['audio1', 'audio2', 'audio3']:
        aligned, gt_aligned = align_audio(audios[key], groundtruth)
        aligned_audios[key] = aligned
        audios['groundtruth_aligned'] = gt_aligned
        logging.info(f"{key} 与 groundtruth 对齐完成。")

    groundtruth_aligned = audios['groundtruth_aligned']

    # 提取特征
    features = {}
    # Groundtruth 特征
    features['groundtruth'] = {
        'mfcc': extract_mfcc(groundtruth_aligned, TARGET_SR),
        'spectral_contrast': extract_spectral_contrast(groundtruth_aligned, TARGET_SR),
        'wavelet': extract_wavelet(groundtruth_aligned)
    }

    # 测试音频特征
    for key in ['audio1', 'audio2', 'audio3']:
        features[key] = {
            'mfcc': extract_mfcc(aligned_audios[key], TARGET_SR),
            'spectral_contrast': extract_spectral_contrast(aligned_audios[key], TARGET_SR),
            'wavelet': extract_wavelet(aligned_audios[key])
        }
        logging.info(f"{key} 的特征提取完成。")

    # 计算相似度
    similarities = {}
    for key in ['audio1', 'audio2', 'audio3']:
        # MFCC 相似度
        sim_mfcc = compute_dtw_distance(features[key]['mfcc'], features['groundtruth']['mfcc'])
        # 频谱对比度相似度
        sim_spectral = compute_dtw_distance(features[key]['spectral_contrast'], features['groundtruth']['spectral_contrast'])
        # 小波变换相似度
        sim_wavelet = compute_euclidean_distance(features[key]['wavelet'], features['groundtruth']['wavelet'])
        # 综合相似度
        sim_composite = compute_composite_similarity(sim_mfcc, sim_spectral, sim_wavelet)
        similarities[key] = sim_composite
        logging.info(f"{key} 的相似度计算完成。")

    # 输出结果
    print("\n--- 相似度结果 ---")
    for key in ['audio1', 'audio2', 'audio3']:
        print(f"{key} 与 Groundtruth 的综合相似度得分: {similarities[key]:.4f}")

if __name__ == "__main__":
    main()
