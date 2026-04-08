import numpy as np
from scipy import signal
import pywt
from scipy.ndimage import zoom


class SignalTransforms:

    @staticmethod
    def _resize(data, target_size=(32, 32)):
        if data.shape != target_size:
            zf = (target_size[0] / data.shape[0], target_size[1] / data.shape[1])
            data = zoom(data, zf, order=3)
        return data

    @staticmethod
    def cwt(signal_window, sampling_rate, scales=64, target_size=(32, 32)):
        coeffs, _ = pywt.cwt(signal_window, np.arange(1, scales + 1), "morl",
                              sampling_period=1 / sampling_rate)
        return SignalTransforms._resize(np.abs(coeffs) ** 2, target_size)

    @staticmethod
    def stft(signal_window, sampling_rate, nperseg=64, noverlap=32, target_size=(32, 32)):
        _, _, Zxx = signal.stft(signal_window, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
        return SignalTransforms._resize(np.abs(Zxx), target_size)

    @staticmethod
    def gaf(signal_window, method="summation", target_size=(32, 32)):
        mn, mx = np.min(signal_window), np.max(signal_window)
        scaled = (2 * (signal_window - mn) / (mx - mn) - 1) if mx != mn else np.zeros_like(signal_window)
        phi = np.arccos(np.clip(scaled, -1, 1))
        matrix = np.cos(phi[:, None] + phi[None, :]) if method == "summation" else np.sin(phi[:, None] - phi[None, :])
        return SignalTransforms._resize(matrix, target_size)

    @staticmethod
    def recurrence_plot(signal_window, eps=0.1, target_size=(32, 32)):
        N = len(signal_window)
        rp = (np.abs(signal_window[:, None] - signal_window[None, :]) < eps).astype(float)
        return SignalTransforms._resize(rp, target_size)

    @staticmethod
    def scalogram(signal_window, sampling_rate, scales=64, target_size=(32, 32)):
        coeffs, _ = pywt.cwt(signal_window, np.logspace(0, 2, scales), "morl",
                              sampling_period=1 / sampling_rate)
        return SignalTransforms._resize(np.abs(coeffs) ** 2, target_size)


def apply_transform(signal_window, sampling_rate, args):
    target_size = getattr(args, "target_size", (32, 32))
    tt = args.transform_type

    if tt == "None":
        return signal_window
    elif tt == "cwt":
        return SignalTransforms.cwt(signal_window, sampling_rate, args.cwt_scales, target_size)
    elif tt == "stft":
        return SignalTransforms.stft(signal_window, sampling_rate, args.stft_nperseg, args.stft_noverlap, target_size)
    elif tt == "gaf":
        return SignalTransforms.gaf(signal_window, args.gaf_method, target_size)
    elif tt == "rp":
        return SignalTransforms.recurrence_plot(signal_window, args.rp_eps, target_size)
    elif tt == "scalogram":
        return SignalTransforms.scalogram(signal_window, sampling_rate, args.cwt_scales, target_size)
    else:
        raise ValueError(f"Unknown transform_type: {tt}")
