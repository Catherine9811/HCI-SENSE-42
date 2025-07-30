import mne
import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler


def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1, split_return=False):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    if not split_return:
        return psd / mean_noise
    else:
        return psd, mean_noise


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_psd_snr(psds, freqs, snrs, fmin, fmax, ymin=-0.5, ymax=4, init=2, targets=None, harmonic_num=12):
    fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
    freq_range = range(
        np.where(np.floor(freqs) == fmin)[0][0], np.where(np.round(freqs) == fmax - 1)[0][0]
    )

    psds_plot = 10 * np.log10(psds)
    psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
    psds_std = psds_plot.std(axis=(0, 1))[freq_range]
    axes[0].plot(freqs[freq_range], psds_mean, color="b")
    axes[0].fill_between(
        freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
    )
    axes[0].set(title="PSD Spectrum", ylabel="Power Spectral Density [dB]")

    # SNR spectrum
    snr_mean = snrs.mean(axis=(0, 1))[freq_range]
    snr_std = snrs.std(axis=(0, 1))[freq_range]

    axes[1].plot(freqs[freq_range], snr_mean, color="r")
    axes[1].fill_between(
        freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.2
    )
    axes[1].set(
        title="SNR Spectrum",
        xlabel="Frequency [Hz]",
        ylabel="SNR",
        ylim=[ymin, ymax],
        xlim=[fmin, fmax],
    )
    if targets is not None:
        if not isinstance(targets, list):
            targets = [targets]
        cmap = get_cmap(len(targets) + 4)
        for n, target in enumerate(targets):
            for harmonic_i in range(init, harmonic_num):
                for axis in axes:
                    axis.axvline(x=target * harmonic_i, color=cmap(n),
                                 linestyle='dashed', linewidth=1.0, alpha=init / harmonic_i,
                                 label=f'{target * harmonic_i:.2f} Hz')
                axis.legend(loc='upper right')
    fig.show()
    plt.show()
    return fig

def barplot_annotate_brackets(num1, num2, data, center, height,
                              yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None, ax=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """
    if ax is None:
        ax = plt

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)