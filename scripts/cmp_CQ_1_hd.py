import os
import matplotlib
matplotlib.use('Agg')  # 使用无GUI后端
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_SR = 48000  # 目标采样率
OUTPUT_PREFIX = "CQ_1_"  # 输出文件前缀

def setup_logging(output_dir):
    """设置日志记录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file = os.path.join(output_dir, f"{OUTPUT_PREFIX}analysis_log.txt")
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])

def normalize_audio(audio):
    """归一化音频信号"""
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        logging.warning("音频信号的RMS为零，无法归一化。")
        return audio
    return audio / rms

def align_audio(audio1, audio2):
    """对齐两个音频信号"""
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
    """加载音频文件"""
    try:
        audio, loaded_sr = librosa.load(path, sr=sr)
        logging.info(f"成功加载 {path}，采样率: {loaded_sr} Hz")
        return audio, sr
    except Exception as e:
        logging.error(f"加载音频文件 {path} 失败: {e}")
        return None, None

def save_frequency_spectrum_plot_with_harmonics_and_phase(freqs, mean_spectrum, harmonic_frequencies, harmonic_phases, output_file):
    """保存带有谐波和相位标记的频谱图"""
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, mean_spectrum, label='Mean Spectrum')
    plt.scatter(harmonic_frequencies, mean_spectrum[np.searchsorted(freqs, harmonic_frequencies)], 
                color='red', label='Detected Harmonics')
    for freq, phase in zip(harmonic_frequencies, harmonic_phases):
        plt.annotate(f"{phase:.2f}", (freq, mean_spectrum[np.searchsorted(freqs, freq)]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
    plt.title("Frequency Spectrum with Detected Harmonics and Phase Information")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim([0, TARGET_SR / 2])
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()

def compute_harmonic_distortion_with_hps(
    audio, 
    sr, 
    output_dir, 
    filename_prefix, 
    threshold_ratio=0.04, 
    min_amplitude=0.1,
    frame_duration=0.01,  # 每帧的持续时间（秒）
    hps=False  # 是否使用谐波乘积谱
):
    """
    检测谐波频率、相位和持续时间。
    可选择是否使用谐波乘积谱（HPS）方法。
    """
    if hps:
        # 使用谐波乘积谱增强谐波检测
        harmonic = librosa.effects.harmonic(audio)
        stft = librosa.stft(harmonic)
    else:
        stft = librosa.stft(audio)
    
    magnitude, phase = np.abs(stft), np.angle(stft)
    freqs = librosa.fft_frequencies(sr=sr)
    mean_magnitude = np.mean(magnitude, axis=1)
    mean_phase = np.mean(phase, axis=1)
    
    # 动态调整阈值
    dynamic_threshold = threshold_ratio * np.max(mean_magnitude)
    peak_indices = np.where(mean_magnitude > dynamic_threshold)[0]
    
    # 过滤低幅度谐波
    peak_amplitudes = mean_magnitude[peak_indices]
    significant_indices = peak_indices[peak_amplitudes > min_amplitude * np.max(mean_magnitude)]
    harmonic_frequencies = freqs[significant_indices]
    harmonic_phases = mean_phase[significant_indices]
    
    # 计算每个谐波的持续时间
    harmonic_durations = {}
    for freq in harmonic_frequencies:
        freq_idx = np.argmin(np.abs(freqs - freq))
        active_frames = magnitude[freq_idx, :] > dynamic_threshold
        duration = np.sum(active_frames) * frame_duration
        harmonic_durations[freq] = duration
    
    logging.info(f"Harmonic frequencies detected: {harmonic_frequencies}")
    logging.info(f"Harmonic phases detected: {harmonic_phases}")
    logging.info(f"Harmonic durations detected: {harmonic_durations}")
    
    # 保存带有谐波和相位标记的频谱图
    output_file = os.path.join(output_dir, f"{OUTPUT_PREFIX}{filename_prefix}_frequency_spectrum.png")
    save_frequency_spectrum_plot_with_harmonics_and_phase(freqs, mean_magnitude, harmonic_frequencies, harmonic_phases, output_file)
    
    return harmonic_frequencies, harmonic_phases, harmonic_durations, mean_magnitude, freqs

def calculate_consistency_score_with_weights(
    matched_harmonics_count, 
    phase_matches_count, 
    duration_weight, 
    total_harmonics, 
    penalty, 
    weight_freq=0.4, 
    weight_phase=0.3, 
    weight_duration=0.3
):
    """
    计算一致性评分，考虑频率匹配、相位匹配和持续时间权重。
    确保评分在0到1之间。
    """
    if total_harmonics == 0:
        return 0.0
    matched_ratio = matched_harmonics_count / total_harmonics
    phase_ratio = phase_matches_count / total_harmonics
    duration_ratio = duration_weight / total_harmonics
    base_score = (matched_ratio * weight_freq) + (phase_ratio * weight_phase) + (duration_ratio * weight_duration)
    consistency_score = max(base_score - penalty, 0.0)
    return consistency_score

def analyze_harmonic_distortion_comparison_with_weights(
    harmonic_frequencies, 
    harmonic_phases, 
    harmonic_durations, 
    groundtruth_harmonic_frequencies, 
    groundtruth_harmonic_phases, 
    groundtruth_harmonic_durations, 
    tolerance_freq=10, 
    tolerance_phase=np.pi / 4
):
    """
    比较谐波失真，考虑频率、相位和持续时间，并应用权重机制。
    """
    matched_harmonics = 0
    phase_matches = 0
    duration_weight = 0
    extra_harmonics = []
    
    for gt_freq, gt_phase, gt_duration in zip(groundtruth_harmonic_frequencies, groundtruth_harmonic_phases, groundtruth_harmonic_durations.values()):
        # 查找匹配频率
        freq_matches = [ (freq, phase, harmonic_durations[freq]) 
                        for freq, phase in zip(harmonic_frequencies, harmonic_phases) 
                        if abs(freq - gt_freq) <= tolerance_freq ]
        if freq_matches:
            # 选择相位差最小的匹配
            freq, phase, duration = min(freq_matches, key=lambda x: abs(x[1] - gt_phase))
            if abs(phase - gt_phase) <= tolerance_phase:
                matched_harmonics += 1
                phase_matches += 1
                # 持续时间加权（假设 gt_duration 为参考）
                duration_weight += min(duration / gt_duration, 1.0)  # 不超过1.0
            else:
                extra_harmonics.append(freq)
        else:
            extra_harmonics.append(gt_freq)
    
    # 查找额外谐波
    for freq in harmonic_frequencies:
        if not any(abs(freq - gt_freq) <= tolerance_freq for gt_freq in groundtruth_harmonic_frequencies):
            extra_harmonics.append(freq)
    
    total_harmonics = len(groundtruth_harmonic_frequencies)
    if total_harmonics == 0:
        return extra_harmonics, 0.0
    
    # 使用比例惩罚，限制最大惩罚为 0.5 分
    penalty = min(len(extra_harmonics) / total_harmonics * 0.5, 0.5)
    
    # 计算一致性评分
    consistency_score = calculate_consistency_score_with_weights(
        matched_harmonics_count=matched_harmonics,
        phase_matches_count=phase_matches,
        duration_weight=duration_weight,
        total_harmonics=total_harmonics,
        penalty=penalty,
        weight_freq=0.4,
        weight_phase=0.3,
        weight_duration=0.3
    )
    
    return extra_harmonics, consistency_score

def save_results_to_file(output_dir, results):
    """保存分析结果到文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{OUTPUT_PREFIX}harmonic_distortion_analysis.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for key, data in results.items():
            f.write(f"{key} 与 Groundtruth 相比，额外谐波成分: {data['extra_harmonics']}, 一致性评分: {data['consistency_score']:.4f}\n")
    logging.info(f"分析结果保存至 {output_file}")

def main():
    audio_paths = {
        'audio1': 'data/ZWZ_CQ_1_1.wav',
        'audio2': 'data/ZWZ_CQ_1_2.wav',
        'audio3': 'data/ZWZ_CQ_1_3.wav',
        'groundtruth': 'data/ZWZ_CQ_1_groundtruth.wav'
    }

    output_dir = "output"
    setup_logging(output_dir)

    audios = {}

    # 并行加载和归一化音频
    with ThreadPoolExecutor() as executor:
        audio_futures = {executor.submit(load_audio, path): key for key, path in audio_paths.items()}
        for future in as_completed(audio_futures):
            key = audio_futures[future]
            audio, sr = future.result()
            if audio is None:
                logging.error(f"音频文件加载失败，程序终止。")
                return
            audios[key] = normalize_audio(audio)

    groundtruth = audios['groundtruth']

    # 并行对齐音频
    aligned_audios = {}
    with ThreadPoolExecutor() as executor:
        align_futures = {executor.submit(align_audio, audios[key], groundtruth): key for key in ['audio1', 'audio2', 'audio3']}
        for future in as_completed(align_futures):
            key = align_futures[future]
            aligned, gt_aligned = future.result()
            aligned_audios[key] = aligned
            audios['groundtruth_aligned'] = gt_aligned

    groundtruth_aligned = audios['groundtruth_aligned']

    # 检测 Groundtruth 的谐波失真（使用 HPS 方法）
    gt_harmonic_frequencies, gt_harmonic_phases, gt_harmonic_durations, gt_spectrum, gt_freqs = compute_harmonic_distortion_with_hps(
        groundtruth_aligned, TARGET_SR, output_dir, "groundtruth", hps=True)  # 使用 HPS 方法

    # 分析 Groundtruth 自身对比（理想情况）
    extra_harmonics_gt, consistency_score_gt = analyze_harmonic_distortion_comparison_with_weights(
        harmonic_frequencies=gt_harmonic_frequencies, 
        harmonic_phases=gt_harmonic_phases, 
        harmonic_durations=gt_harmonic_durations, 
        groundtruth_harmonic_frequencies=gt_harmonic_frequencies, 
        groundtruth_harmonic_phases=gt_harmonic_phases, 
        groundtruth_harmonic_durations=gt_harmonic_durations, 
        tolerance_freq=10, 
        tolerance_phase=np.pi / 4
    )

    print("\n--- Groundtruth 自身对比 ---")
    print(f"Groundtruth 与自身相比，额外谐波成分: {extra_harmonics_gt}, 一致性评分: {consistency_score_gt:.4f}")

    results = {}

    # 并行检测谐波失真（使用 HPS 方法）
    with ThreadPoolExecutor() as executor:
        harmonic_futures = {
            executor.submit(compute_harmonic_distortion_with_hps, aligned_audios[key], TARGET_SR, output_dir, key, hps=True): key 
            for key in ['audio1', 'audio2', 'audio3']
        }
        for future in as_completed(harmonic_futures):
            key = harmonic_futures[future]
            harmonic_frequencies, harmonic_phases, harmonic_durations, spectrum, freqs = future.result()
            extra_harmonics, consistency_score = analyze_harmonic_distortion_comparison_with_weights(
                harmonic_frequencies=harmonic_frequencies, 
                harmonic_phases=harmonic_phases, 
                harmonic_durations=harmonic_durations, 
                groundtruth_harmonic_frequencies=gt_harmonic_frequencies, 
                groundtruth_harmonic_phases=gt_harmonic_phases, 
                groundtruth_harmonic_durations=gt_harmonic_durations, 
                tolerance_freq=10, 
                tolerance_phase=np.pi / 4
            )
            logging.info(f"{key} 相比 groundtruth 的额外谐波: {extra_harmonics}, 一致性评分: {consistency_score:.4f}")
            results[key] = {'extra_harmonics': extra_harmonics, 'consistency_score': consistency_score}

    print("\n--- 谐波失真分析结果 ---")
    for key in ['audio1', 'audio2', 'audio3']:
        print(f"{key} 与 Groundtruth 相比，额外谐波成分: {results[key]['extra_harmonics']}, 一致性评分: {results[key]['consistency_score']:.4f}")

    save_results_to_file(output_dir, results)

if __name__ == "__main__":
    main()
