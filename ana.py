from flask import Flask
from main import parse_args,mysql,httpclient,np,wavelet_denoising





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
        'm-2101644986-lxdf7', inputs=inputs, outputs=outputs)

    # 获取输出数据
    output_data = results.as_numpy('output')

    overall_result = find_output(output_data)
    # 检测结果为 overall_result的值，0为心梗，1为正常，使用时调用该值即可
    if overall_result == 0:
        print("检测到可能的心梗信号，请尽快就医。")
    else:
        print("检测到正常信号，未发现异常。")

    return  overall_result


def loadfile():
    signal=mysql()
    np.save("updatefile\\signal.npy", signal)
    return "updatefile\\signal.npy"


filepath ="updatefile\\filtered_signal.npy"
result_ana(filepath)
