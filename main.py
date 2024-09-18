from collections import Counter
import numpy as np
import pywt
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
from scipy.signal import medfilt
from wfdb.processing import (find_peaks,correct_peaks,gqrs_detect)
import argparse
import sys

import tritonclient.http as httpclient


def normalize_signal(signal):
    # 找到信号的最大值和最小值
    max_value = np.max(signal)
    min_value = np.min(signal)

    # 计算信号的范围
    signal_range = max_value - min_value

    # 对信号进行归一化
    normalized_signal = (signal - min_value) / signal_range

    # 将归一化后的信号值映射到（1，-1）区间内
    normalized_signal = 2 * normalized_signal - 1

    return normalized_signal

def wavelet_denoising(signal, wavelet='db6', level=5):
    # 进行小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 计算每个尺度上的噪声标准差
    sigma = [np.median(np.abs(coeff)) / 0.6745 for coeff in coeffs[1:]]

    # 对细节系数应用阈值处理
    thresholds = [sigma_i * np.sqrt(2 * np.log(len(coeff))) for sigma_i, coeff in zip(sigma, coeffs[1:])]
    denoised_coeffs = [pywt.threshold(coeff, value=threshold, mode='soft') for coeff, threshold in
                       zip(coeffs[1:], thresholds)]
    denoised_coeffs.insert(0, coeffs[0])

    # 重构去噪信号
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    return denoised_signal

def mysql():
    MYSQL_HOST = args.MySQL_host
    MYSQL_PORT = args.MySQL_port
    MYSQL_USER = args.MySQL_user
    MYSQL_PASSWORD = args.MySQL_password
    MYSQL_DB = args.MySQL_db

    engine = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8'
                               % (MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB))

    sql = 'SELECT * FROM `ecg_database`.`data`'

    df = pd.read_sql(sql, engine)

    target_column = None
    reshaped_data = []

    for col in df.columns:
        if not df[col].isnull().all():
            target_column = col
            break

    if target_column is None:
        print("所有列均无数据，请检查输入数据。")
    else:
        print(target_column)
        selected_column = df[target_column]
        reshaped_data = selected_column.values.reshape((1, len(selected_column)))

    signal = np.squeeze(reshaped_data)

    return signal

def filtered_r_peaks(selected_r_peaks,ventricular_signal):
    threshold = 330
    filtered_r_peaks = []
    for i in range(len(selected_r_peaks)):
        if i == 0:
            filtered_r_peaks.append(selected_r_peaks[i])
        else:
            prev_interval = selected_r_peaks[i] - selected_r_peaks[i - 1]
            next_interval = selected_r_peaks[i] - filtered_r_peaks[-1]
            # 如果前后间隔都小于阈值
            if prev_interval < threshold or next_interval < threshold:
                # 比较当前R峰与前一个R峰、后一个R峰的实际信号值，保留实际信号值最大的R峰
                if ventricular_signal[selected_r_peaks[i]] >= ventricular_signal[filtered_r_peaks[-1]]:
                    filtered_r_peaks.pop()
                    filtered_r_peaks.append(selected_r_peaks[i])
            else:
                filtered_r_peaks.append(selected_r_peaks[i])

    return filtered_r_peaks

def heartbeat_segment(pre_points, post_points, filtered_r_peak, ventricular_leads):
    heartbeats = []
    for peak in filtered_r_peak:
        start = peak - pre_points
        end = peak + post_points
        if start < 0:
            heartbeat = np.hstack((np.zeros(-start), ventricular_leads[:end]))
            heartbeats.append(heartbeat)
        else:
            if end > len(ventricular_leads):
                heartbeat = np.hstack((ventricular_leads[start:], np.zeros(end - len(ventricular_leads))))
                heartbeats.append(heartbeat)
            else:
                heartbeat = ventricular_leads[start:end]
                heartbeats.append(heartbeat)

    heartbeats = np.array(heartbeats)
    return heartbeats

def show(filtered_signal,filtered_r_peak,signal):
    # 绘制心电信号和R峰位置
    plt.subplot(2, 1, 2)
    plt.plot(filtered_signal)
    plt.plot(filtered_r_peak, filtered_signal[filtered_r_peak], 'ro', label='R Peaks')
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Amplitude(mV)', fontsize=12)
    plt.title('(b)ECG signal after denoising', fontsize=12)
    plt.legend()

    # 绘制折线图
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Amplitude(mV)', fontsize=12)
    plt.title('(a)Original ECG signal', fontsize=12)
    plt.show()

def find_output(output_data):
    parsed_results = []
    for row in output_data:
        if len(row) > 0:
            # 只处理每一行的第一个元素
            item = row[0]
            # 解析冒号后的标签
            label = item.split(':')[1]
            parsed_results.append(int(label))

    # 使用 Counter 统计每个标签的出现次数
    counter = Counter(parsed_results)
    most_common = counter.most_common(1)
    if most_common:
        overall_result = most_common[0][0]
    else:
        overall_result = None

    return overall_result

def main(args):
    triton_client = httpclient.InferenceServerClient(url=args.server_url)
    signal = mysql() # 从MySQL服务器中获取ECG信号
    # signal = np.load("normal_signal.npy")
    filtered_signal = wavelet_denoising(signal)
    normalize_signals = normalize_signal(filtered_signal)
    r_peaks = find_peaks(normalize_signals)
    r_peaks = r_peaks[0]
    selected_r_peaks = [r_peak for r_peak in r_peaks if normalize_signals[r_peak] > 0.2]
    filtered_r_peak = filtered_r_peaks(selected_r_peaks, normalize_signals)
    heartbeats = heartbeat_segment(250, 400, filtered_r_peak, filtered_signal).astype(np.float32)#(72.650)  ndarray

    # show(filtered_signal,filtered_r_peak,signal)

    # 取出相应的数据行
    batch_data = heartbeats[:args.max_batch, :]
    batch_data = np.expand_dims(batch_data, axis=1)

    inputs = []
    inputs.append(httpclient.InferInput(
            "input", batch_data.shape, "FP32"))
    # 将数据设置为输入
    inputs[0].set_data_from_numpy(batch_data)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput(
            'output', binary_data= False, class_count=2))

    results = triton_client.infer(
            'm-2101644986-lxdf7', inputs=inputs, outputs=outputs)

    # 获取输出数据
    output_data = results.as_numpy('output')

    overall_result = find_output(output_data)
    # 检测结果为 overall_result的值，0为心梗，1为正常，使用时调用该值即可
    if overall_result == 0:
        print("检测到可能的心梗信号，请尽快就医。")
    else:
        print("检测到正常信号，未发现异常。")



# a="192.168.69.2"

def parse_args():
    parser = argparse.ArgumentParser(description='Triton HTTP Client')
    parser.add_argument("--server_url", help="Server http url", required=False,
                        default= "192.168.124.37:30000", type=str)
    parser.add_argument("--MySQL_host", help="MySQL Server http url", required=False,
                        default="192.168.124.37", type=str)
    parser.add_argument("--MySQL_port", help="MySQL Server http port", required=False,
                        default="3306", type=str)
    parser.add_argument("--MySQL_user", help="MySQL Server user", required=False,
                        default="ye", type=str)
    parser.add_argument("--MySQL_password", help="MySQL Server password", required=False,
                        default="123456", type=str)
    parser.add_argument("--MySQL_db", help="MySQL Server Database", required=False,
                        default="ecg_database", type=str)
    parser.add_argument("--max_batch", help="Max batch size",
                        required=False, default=6, type=int)

    args = parser.parse_args()
    return args

#返回数据的结果
def result_ana(filepath):
    args = parse_args()
    triton_client = httpclient.InferenceServerClient(url=args.server_url)
    signal = np.load(filepath)
    filtered_signal = wavelet_denoising(signal)
    normalize_signals = normalize_signal(filtered_signal)
    r_peaks = find_peaks(normalize_signals)
    r_peaks = r_peaks[0]
    selected_r_peaks = [r_peak for r_peak in r_peaks if normalize_signals[r_peak] > 0.2]
    filtered_r_peak = filtered_r_peaks(selected_r_peaks, normalize_signals)
    heartbeats = heartbeat_segment(250, 400, filtered_r_peak, filtered_signal).astype(np.float32)  # (72.650)  ndarray


    batch_data = heartbeats[:args.max_batch, :]
    batch_data = np.expand_dims(batch_data, axis=1)

    inputs = []
    inputs.append(httpclient.InferInput(
        "input", batch_data.shape, "FP32"))
    # 将数据设置为输入
    inputs[0].set_data_from_numpy(batch_data)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput(
        'output', binary_data=False, class_count=2))

    results = triton_client.infer(
        'm-2101644986-jd6jg', inputs=inputs, outputs=outputs)

    # 获取输出数据
    output_data = results.as_numpy('output')

    overall_result = find_output(output_data)
    # 检测结果为 overall_result的值，0为心梗，1为正常，使用时调用该值即可
    if overall_result == 0:
        print("检测到可能的心梗信号，请尽快就医。")
    else:
        print("检测到正常信号，未发现异常。")

    return  overall_result


def mysql2():
    MYSQL_HOST = "192.168.124.34"
    MYSQL_PORT = "3306"
    MYSQL_USER = "ye"
    MYSQL_PASSWORD = "123456"
    MYSQL_DB = "ecg_database"

    engine = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8'
                               % (MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB))

    sql = 'SELECT * FROM `ecg_database`.`data`'

    df = pd.read_sql(sql, engine)

    target_column = None
    reshaped_data = []

    for col in df.columns:
        if not df[col].isnull().all():
            target_column = col
            break

    if target_column is None:
        print("所有列均无数据，请检查输入数据。")
    else:
        print(target_column)
        selected_column = df[target_column]
        reshaped_data = selected_column.values.reshape((1, len(selected_column)))

    signal = np.squeeze(reshaped_data)
    file_path = "signal.npy"
    np.save(file_path, signal)
    print(f"Signal data saved to {file_path}")
    return file_path


