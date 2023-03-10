import matplotlib.pyplot as plt
import numpy as np
import librosa as liro

def main():
    rng = np.random.default_rng(1234)

    # this preprocessing parameters should be in a
    # config file for your runs AND should be logged as well!
    sr = 22100
    n_fft = 256
    n_hop = 128
    n_mels = 48
    fmin = 80       # high pass filter cutoff in [Hz]
    fmax = sr // 2  # low pass filter cutoff in [Hz] (at f_nyquist)

    f0 = 220
    tones = []
    for p in [1, 4, 8, 16]:
        tone = liro.tone(f0 * p, sr=sr, length=sr // 8)
        envelope = np.exp(np.linspace(1, 0, len(tone)))
        tones.append(tone * envelope)

    tones = np.hstack(tones)
    stft = liro.stft(tones, n_fft=n_fft, hop_length=n_hop)
    spec = np.abs(stft)
    fft_frequencies = liro.fft_frequencies(
        sr=sr,
        n_fft=n_fft
    )

    filterbank = liro.filters.mel(
        sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    mel_spec = filterbank.dot(spec)
    mel_frequencies = liro.mel_frequencies(
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    fig, axes = plt.subplots(nrows=3)
    fig_time, ax_time = plt.subplots()
    fig_spec, ax_spec = plt.subplots()
    fig_mel_spec, ax_mel_spec = plt.subplots()

    fig_mel_filt, axes_mel_filt = plt.subplots(nrows=2)
    
    ax = axes_mel_filt[0]
    ax.imshow(filterbank, origin='lower', interpolation='nearest')
    ax.set_xlabel('FFT bin index')
    ax.set_ylabel('MEL bin index')
    ax.set_aspect('auto')

    ax = axes_mel_filt[1]
    offset = filterbank.max()
    for fi, filt in enumerate(filterbank):
        ax.plot(fi * offset + filt, linewidth=0.5, color='k')
    ax.set_aspect('auto')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim([0, filterbank.shape[1]])
    ax.axis('off')

    fig_mel_filt.tight_layout()
    fig_mel_filt.savefig('filterbank.pdf', dpi=300)
    
    for ax in [axes[0], ax_time]:
        ax.plot(tones, color='k', linewidth=0.1)

        ax.spines.right.set_color('none')
        ax.spines.top.set_color('none')

        ax.set_xlim([0, len(tones)])
        ax.set_xlabel('time (in samples)')
        ax.set_ylabel('volume')

    for ax in [axes[1], ax_spec]:
        ax.imshow(
            spec,
            cmap='viridis',
            origin='lower',
            interpolation='nearest'
        )
        ax.set_xlabel('time (in frames)')
        ax.set_ylabel('frequency in [Hz]\n(linear)')
        ax.set_aspect('auto')

        yticks = []
        ylabels = []
        for fi in range(0, len(fft_frequencies), 30):
            ytick = fi
            ylabel = f'{fft_frequencies[fi]:5.1f}'
            yticks.append(ytick)
            ylabels.append(ylabel)
        ax.set_yticks(yticks, ylabels)

    for ax in [axes[2], ax_mel_spec]:
        ax.imshow(
            mel_spec,
            cmap='viridis',
            origin='lower',
            interpolation='nearest'
        )
        yticks = []
        ylabels = []
        for mi in range(0, len(mel_frequencies), 10):
            ytick = mi
            ylabel = f'{mel_frequencies[mi]:5.1f}'
            yticks.append(ytick)
            ylabels.append(ylabel)
        ax.set_yticks(yticks, ylabels)

        ax.set_aspect('auto')
        ax.set_xlabel('time (in frames)')
        ax.set_ylabel('frequency in [Hz]\n(logarithmic, mel)')

    fig.tight_layout()
    fig_time.tight_layout()
    fig_spec.tight_layout()
    fig_mel_spec.tight_layout()

    fig.savefig('all_spec.pdf', dpi=300)
    fig_time.savefig('time.pdf', dpi=300)
    fig_spec.savefig('spec.pdf', dpi=300)
    fig_mel_spec.savefig('mel_spec.pdf', dpi=300)


if __name__ == '__main__':
    main()
