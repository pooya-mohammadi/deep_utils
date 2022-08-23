import numpy as np
import librosa


class LibrosaUtils:
    @staticmethod
    def stft(audio,
             n_fft,
             n_shift,
             win_length=None,
             window="hann",
             center=True,
             pad_mode="reflect"):
        """
        Computes Short-time Fourier transform for a given audio signal using librosa
        :param audio: audio's shape is [time, n_channel]
        :param n_fft:
        :param n_shift:
        :param win_length:
        :param window:
        :param center:
        :param pad_mode:
        :return:
        """

        if audio.ndim == 1:
            single_channel = True
            audio = np.expand_dims(audio, 1)
        else:
            single_channel = False
        audio = audio.astype(np.float32)

        # audio: [time, channel, freq]
        audio = np.stack([librosa.stft(audio[:, ch],
                                       n_fft=n_fft,
                                       hop_length=n_shift,
                                       win_length=win_length,
                                       window=window,
                                       center=center,
                                       pad_mode=pad_mode, ).T for ch in range(audio.shape[1])], axis=1)
        if single_channel:
            # audio: [time, channel, freq] -> [time, freq]
            audio = audio[:, 0]

        return audio

    @staticmethod
    def stft2log_mel_spectrogram(x_stft,
                                 fs,
                                 n_mels,
                                 n_fft,
                                 fmin=None,
                                 fmax=None,
                                 eps=1e-10):
        """
        Converts STFT to log_mel_spectrogram
        :param x_stft: 
        :param fs: 
        :param n_mels: 
        :param n_fft: 
        :param fmin: 
        :param fmax: 
        :param eps: 
        :return: 
        """
        # x_stft: (time, channel, freq) or (time, freq)
        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax

        # spc: (time, channel, freq) or (time, freq)
        spc = np.abs(x_stft)
        # mel_basis: (mel_freq, freq)
        mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
        # lmspc: (time, channel, mel_freq) or (time, mel_freq)
        lmspc = np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

        return lmspc

    @staticmethod
    def log_mel_spectrogram(audio,
                            sample_rate,
                            n_fft,
                            hop_length,
                            win_length,
                            num_mels,
                            mel_fmin,
                            mel_fmax,
                            window="hann",
                            pad_mode="reflect"):
        """
        Computes log mel_spectrogram of an audio signal
        :param audio: 
        :param sample_rate: 
        :param n_fft: 
        :param hop_length: 
        :param win_length: 
        :param num_mels: 
        :param mel_fmin: 
        :param mel_fmax: 
        :param window: 
        :param pad_mode: 
        :return: 
        """
        # Compute STFT
        x_stft = LibrosaUtils.stft(audio,
                                   n_fft=n_fft,
                                   n_shift=hop_length,
                                   win_length=win_length,
                                   window=window,
                                   pad_mode=pad_mode)

        # Compute log-mel-spectrogram
        return LibrosaUtils.stft2log_mel_spectrogram(x_stft,
                                                     fs=sample_rate,
                                                     n_mels=num_mels,
                                                     n_fft=n_fft,
                                                     fmin=mel_fmin,
                                                     fmax=mel_fmax,
                                                     eps=1e-10).T

    @staticmethod
    def trim_silence(auido, margin_db, frame_length=2048, hop_length=256):
        """
        Trims input wav based on margin_db
        :param auido:
        :param margin_db:
        :param frame_length:
        :param hop_length:
        :return:
        """
        return librosa.effects.trim(auido,
                                    top_db=margin_db,
                                    frame_length=frame_length,
                                    hop_length=hop_length)[0]
