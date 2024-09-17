import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pywt

# 设置统一采样率
TARGET_SR = 48000

# 设置日志保存路径
def setup_logging(output_dir):
    """
    设置日志保存到文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file = os.path.join(output_dir, "RDWPS_analysis_log.txt")
    
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

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

def save_frequency_spectrum_plot(freqs, mean_spectrum, output_file):
    """
    保存频谱图到指定文件
    """
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, mean_spectrum)
    plt.title("Frequency Spectrum (to detect Harmonic Distortion)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim([0, TARGET_SR / 2])  # 只显示到奈奎斯特频率（采样率的一半）
    plt.grid(True)
    
    # 保存到文件
    plt.savefig(output_file)
    plt.close()

def compute_harmonic_distortion(audio, sr, output_dir, filename_prefix, threshold_ratio=0.05):
    """
    计算音频信号的频谱，检测谐波失真，包含对频率倍数关系的分析。
    """
    stft = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)
    mean_spectrum = np.mean(stft, axis=1)
    
    # 保存频谱图到文件
    output_file = os.path.join(output_dir, f"{filename_prefix}_frequency_spectrum.png")
    save_frequency_spectrum_plot(freqs, mean_spectrum, output_file)
    
    # 使用阈值查找频谱中的峰值，用于识别谐波成分
    peak_indices = np.where(mean_spectrum > np.max(mean_spectrum) * threshold_ratio)[0]
    harmonic_frequencies = freqs[peak_indices]
    
    logging.info(f"Harmonic frequencies detected: {harmonic_frequencies}")
    
    return harmonic_frequencies

def calculate_consistency_score(harmonic_frequencies, groundtruth_harmonic_frequencies):
    """
    计算一致性评分，返回一个0到1的分数，1表示完全一致。
    """
    matched_harmonics = len(set(harmonic_frequencies).intersection(set(groundtruth_harmonic_frequencies)))
    total_harmonics = len(groundtruth_harmonic_frequencies)
    
    if total_harmonics == 0:
        return 0
    
    score = matched_harmonics / total_harmonics
    return score

def analyze_harmonic_distortion_comparison(harmonic_frequencies, groundtruth_harmonic_frequencies):
    """
    比较音频和groundtruth之间的谐波频率差异，分析谐波失真，并计算一致性评分。
    """
    difference = set(harmonic_frequencies) - set(groundtruth_harmonic_frequencies)
    extra_harmonics = [freq for freq in difference if not any(np.isclose(freq % gt, 0, atol=1) for gt in groundtruth_harmonic_frequencies)]
    
    # 一致性评分
    consistency_score = calculate_consistency_score(harmonic_frequencies, groundtruth_harmonic_frequencies)
    
    return extra_harmonics, consistency_score

def save_results_to_file(output_dir, results):
    """
    将分析结果保存到文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "RDWPS_harmonic_distortion_analysis.txt")
    with open(output_file, 'w') as f:
        for key, data in results.items():
            f.write(f"{key} 与 Groundtruth 相比，额外谐波成分: {data['extra_harmonics']}, 一致性评分: {data['consistency_score']}\n")
    
    logging.info(f"分析结果保存至 {output_file}")

def main():
    # 定义音频文件路径
    audio_paths = {
        'audio1': 'data/ZWZ_RDWPS_1.wav',
        'audio2': 'data/ZWZ_RDWPS_2.wav',
        'audio3': 'data/ZWZ_RDWPS_3.wav',
        'groundtruth': 'data/ZWZ_RDWPS_groundtruth.wav'
    }

    # 设置日志保存路径
    output_dir = "output"
    setup_logging(output_dir)

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

    # 提取谐波失真特征
    logging.info("检测 Groundtruth 的谐波失真...")
    harmonic_frequencies_groundtruth = compute_harmonic_distortion(groundtruth_aligned, TARGET_SR, output_dir, "groundtruth")

    # 结果字典
    results = {}

    for key in ['audio1', 'audio2', 'audio3']:
        logging.info(f"检测 {key} 的谐波失真...")
        harmonic_frequencies = compute_harmonic_distortion(aligned_audios[key], TARGET_SR, output_dir, key)
        extra_harmonics, consistency_score = analyze_harmonic_distortion_comparison(harmonic_frequencies, harmonic_frequencies_groundtruth)
        logging.info(f"{key} 相比 groundtruth 的额外谐波: {extra_harmonics}, 一致性评分: {consistency_score}")
        results[key] = {'extra_harmonics': extra_harmonics, 'consistency_score': consistency_score}
    
    # 输出谐波分析结果
    print("\n--- 谐波失真分析结果 ---")
    for key in ['audio1', 'audio2', 'audio3']:
        print(f"{key} 与 Groundtruth 相比，额外谐波成分: {results[key]['extra_harmonics']}, 一致性评分: {results[key]['consistency_score']}")
    
    # 保存分析结果到文件
    save_results_to_file(output_dir, results)

if __name__ == "__main__":
    main()
