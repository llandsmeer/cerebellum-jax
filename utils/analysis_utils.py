import numpy as np
import brainpy.math as bm
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert


def calculate_avg_firing_rate(spike_train, dt, duration):
    """
    Calculates the average firing rate per neuron from a spike train.

    Args:
        spike_train (np.ndarray or bm.ndarray): Boolean or binary array of spikes.
                                                Shape: (num_steps, num_neurons)
        dt (float): Simulation time step in ms.
        duration (float): Total simulation duration in ms.

    Returns:
        float: Average firing rate across all neurons in Hz.
    """
    if isinstance(spike_train, bm.ndarray):
        spike_train = spike_train.to_numpy()  # Convert if BrainPy array

    total_spikes = np.sum(spike_train)
    num_neurons = spike_train.shape[1]
    # Duration in seconds
    duration_sec = duration / 1000.0

    if num_neurons == 0 or duration_sec == 0:
        return 0.0

    # Avg spikes per neuron over the duration
    avg_spikes_per_neuron = total_spikes / num_neurons
    # Avg firing rate in Hz (spikes per second)
    avg_rate = avg_spikes_per_neuron / duration_sec
    return avg_rate


def calculate_io_sto_synchrony(
    voltage_traces, dt, lowcut=4.0, highcut=12.0, fs_scale=1000.0
):
    """
    Calculates the Kuramoto Order Parameter for neuronal voltage oscillations.

    Args:
        voltage_traces (np.ndarray or bm.ndarray): Membrane potential traces (e.g., V_soma, V_dend).
                                                 Shape: (num_steps, num_neurons)
        dt (float): Simulation time step in ms.
        lowcut (float): Lower bound of the frequency band for filtering (Hz).
        highcut (float): Upper bound of the frequency band for filtering (Hz).
        fs_scale (float): Factor to convert dt to sampling frequency (e.g., 1000.0 for ms).

    Returns:
        np.ndarray: Time series of the Kuramoto Order Parameter magnitude |R(t)|.
    """
    if isinstance(voltage_traces, bm.ndarray):
        voltage_traces = voltage_traces.to_numpy()

    num_steps, num_neurons = voltage_traces.shape
    fs = fs_scale / dt  # Sampling frequency in Hz

    if num_neurons <= 1:
        return np.ones(num_steps)  # Synchrony is 1 for a single or zero neurons

    # Design the Butterworth band-pass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Relax the filter order if necessary for stability
    order = 2
    try:
        b, a = butter(order, [low, high], btype="band")
    except ValueError:
        print(
            f"Warning: Filter critical frequencies might be unstable ({low=}, {high=}). Trying order 1."
        )
        try:
            order = 1
            b, a = butter(order, [low, high], btype="band")
        except ValueError as e:
            print(f"Error designing filter even with order 1: {e}. Returning NaN.")
            return np.full(num_steps, np.nan)

    # Apply the filter to each neuron's V_dend trace
    filtered_v_dend = np.zeros_like(voltage_traces)
    for i in range(num_neurons):
        # Detrend data before filtering
        trace = voltage_traces[:, i]
        trace_detrended = trace - np.mean(trace)
        # Pad the signal to reduce edge artifacts - use constant padding
        padlen = min(
            len(trace_detrended) - 1, max(15, 3 * order)
        )  # SciPy recommends 3*order padding
        filtered_v_dend[:, i] = filtfilt(
            b, a, trace_detrended, padlen=padlen, padtype="constant"
        )

    # Calculate the analytic signal using Hilbert transform to get phase
    analytic_signal = hilbert(filtered_v_dend, axis=0)
    phases = np.angle(analytic_signal)  # Shape: (num_steps, num_neurons)

    # Calculate the Kuramoto Order Parameter R(t) = (1/N) * sum(exp(1j * phase_k(t)))
    kuramoto_param = np.mean(np.exp(1j * phases), axis=1)  # Shape: (num_steps,)

    # Return the magnitude |R(t)|
    return np.abs(kuramoto_param)


def check_network_stability(
    runner,
    pc_rate_thresh=(5, 200),
    cn_rate_thresh=(5, 150),
    io_rate_thresh=(0.01, 2.0),
    pc_v_thresh=(-90, 10),
    cn_v_thresh=(-90, 10),
    io_v_thresh=(-90, 50),
    pc_w_thresh=(-0.5, 1.5),
    cn_w_thresh=(-0.5, 1.5),
):
    """
    Performs a basic check for signs of network instability in the simulation results.

    Args:
        runner (bp.DSRunner): The BrainPy DSRunner object after simulation.
        pc_rate_thresh (tuple): (min_Hz, max_Hz) acceptable average PC firing rate.
        cn_rate_thresh (tuple): (min_Hz, max_Hz) acceptable average CN firing rate.
        io_rate_thresh (tuple): (min_Hz, max_Hz) acceptable average IO firing rate.
                                Note: IO firing is based on soma voltage crossing a threshold.
        pc_v_thresh (tuple): (min_mV, max_mV) acceptable PC membrane potential range.
        cn_v_thresh (tuple): (min_mV, max_mV) acceptable CN membrane potential range.
        io_v_thresh (tuple): (min_mV, max_mV) acceptable IO membrane potential range (all compartments).
        pc_w_thresh (tuple): (min, max) acceptable PC adaptation variable 'w' range.
        cn_w_thresh (tuple): (min, max) acceptable CN adaptation variable 'w' range.


    Returns:
        dict: A dictionary containing stability flags and messages.
              Keys: 'stable' (bool), 'messages' (list of strings).
    """
    results = runner.mon
    dt = runner.dt
    duration = runner.duration
    stability_report = {"stable": True, "messages": []}

    # --- Check for NaNs ---
    for key, value in results.items():
        # Check both numpy and brainpy arrays for NaNs
        is_nan = False
        if isinstance(value, np.ndarray) or isinstance(value, bm.ndarray):
            if isinstance(value, bm.ndarray):
                value_np = value.to_numpy()
            else:
                value_np = value

            # Explicitly check for NaNs, ignoring if checks fail (e.g., boolean array)
            try:
                if np.isnan(value_np).any():
                    is_nan = True
            except TypeError:
                pass  # Cannot check NaNs for this dtype (e.g. bool)

        if is_nan:
            stability_report["stable"] = False
            stability_report["messages"].append(
                f"Instability detected: NaN found in '{key}'."
            )
            # If NaNs are found, further checks might be unreliable
            return stability_report

    # --- Check Firing Rates ---
    # PC Firing Rate
    if "pc.spike" in results:
        pc_rate = calculate_avg_firing_rate(results["pc.spike"], dt, duration)
        if not (pc_rate_thresh[0] <= pc_rate <= pc_rate_thresh[1]):
            stability_report["stable"] = False
            stability_report["messages"].append(
                f"Potential Instability: PC avg firing rate ({pc_rate:.2f} Hz) outside threshold ({pc_rate_thresh} Hz)."
            )

    # CN Firing Rate
    if "cn.spike" in results:
        cn_rate = calculate_avg_firing_rate(results["cn.spike"], dt, duration)
        if not (cn_rate_thresh[0] <= cn_rate <= cn_rate_thresh[1]):
            stability_report["stable"] = False
            stability_report["messages"].append(
                f"Potential Instability: CN avg firing rate ({cn_rate:.2f} Hz) outside threshold ({cn_rate_thresh} Hz)."
            )

    io_thresh = (
        runner.net.io_to_pc.io_threshold if hasattr(runner.net, "io_to_pc") else -30.0
    )  # Default fallback
    if "io.V_soma" in results:
        # Crude spike detection: crossing threshold from below
        io_v_soma_np = results["io.V_soma"].to_numpy()
        io_spikes = (io_v_soma_np[1:] > io_thresh) & (io_v_soma_np[:-1] <= io_thresh)
        # Prepend False to match original length for the function
        io_spikes_full = np.vstack(
            [np.zeros((1, io_spikes.shape[1]), dtype=bool), io_spikes]
        )
        io_rate = calculate_avg_firing_rate(io_spikes_full, dt, duration)
        if not (io_rate_thresh[0] <= io_rate <= io_rate_thresh[1]):
            stability_report["stable"] = False
            stability_report["messages"].append(
                f"Potential Instability: IO avg firing rate ({io_rate:.4f} Hz, based on V_soma > {io_thresh}mV) outside threshold ({io_rate_thresh} Hz)."
            )

    # --- Check Variable Ranges ---
    def check_range(data, name, thresholds):
        if not (isinstance(data, np.ndarray) or isinstance(data, bm.ndarray)):
            stability_report["messages"].append(
                f"Warning: Cannot check range for '{name}', unexpected data type: {type(data)}."
            )
            return

        if isinstance(data, bm.ndarray):
            data_np = data.to_numpy()
        else:
            data_np = data

        min_val, max_val = np.min(data_np), np.max(data_np)
        if not (
            thresholds[0] <= min_val <= thresholds[1]
            and thresholds[0] <= max_val <= thresholds[1]
        ):
            stability_report["stable"] = False
            stability_report["messages"].append(
                f"Potential Instability: '{name}' values (min={min_val:.2f}, max={max_val:.2f}) outside threshold ({thresholds})."
            )

    if "pc.V" in results:
        check_range(results["pc.V"], "PC Vm", pc_v_thresh)
    if "cn.V" in results:
        check_range(results["cn.V"], "CN Vm", cn_v_thresh)
    if "io.V_soma" in results:
        check_range(results["io.V_soma"], "IO Soma Vm", io_v_thresh)
    if "io.V_dend" in results:
        check_range(results["io.V_dend"], "IO Dend Vm", io_v_thresh)
    if "io.V_axon" in results:
        check_range(results["io.V_axon"], "IO Axon Vm", io_v_thresh)
    if "pc.w" in results:
        check_range(results["pc.w"], "PC w", pc_w_thresh)
    if "cn.w" in results:
        check_range(results["cn.w"], "CN w", cn_w_thresh)
    # Add checks for other variables like IO gating variables, Ca2+ concentration if needed

    if not stability_report["messages"]:
        stability_report["messages"].append(
            "Network appears stable within defined thresholds."
        )

    return stability_report


def calculate_fourier_peak(voltage_traces, dt, freq_range=(1.0, 20.0), fs_scale=1000.0):
    """
    Calculates the peak frequency in the average power spectrum of voltage traces.

    Args:
        voltage_traces (np.ndarray or bm.ndarray): Membrane potential traces.
                                                 Shape: (num_steps, num_neurons)
        dt (float): Simulation time step in ms.
        freq_range (tuple): (min_Hz, max_Hz) frequency range to find the peak.
        fs_scale (float): Factor to convert dt to sampling frequency (e.g., 1000.0 for ms).

    Returns:
        tuple: (peak_freq, avg_power_spectrum, frequencies)
               - peak_freq (float): Frequency (Hz) with the maximum power in the specified range.
               - avg_power_spectrum (np.ndarray): Average power spectrum across neurons.
               - frequencies (np.ndarray): Frequencies corresponding to the spectrum.
    """
    if isinstance(voltage_traces, bm.ndarray):
        voltage_traces = voltage_traces.to_numpy()

    num_steps, num_neurons = voltage_traces.shape
    fs = fs_scale / dt  # Sampling frequency in Hz

    if num_neurons == 0:
        return np.nan, np.array([]), np.array([])

    total_power_spectrum = np.zeros(num_steps // 2)  # Accumulate power spectra

    for i in range(num_neurons):
        # Detrend before FFT
        trace = voltage_traces[:, i]
        trace_detrended = trace - np.mean(trace)

        # Compute FFT
        fft_vals = np.fft.fft(trace_detrended)
        # Compute Power Spectrum (magnitude squared), take only positive frequencies
        power = np.abs(fft_vals[: num_steps // 2]) ** 2
        total_power_spectrum += power

    # Average power spectrum
    avg_power_spectrum = total_power_spectrum / num_neurons

    # Get corresponding frequencies
    frequencies = np.fft.fftfreq(num_steps, d=dt / fs_scale)[: num_steps // 2]

    # Find peak frequency within the specified range
    min_freq, max_freq = freq_range
    idx_range = np.where((frequencies >= min_freq) & (frequencies <= max_freq))[0]

    if len(idx_range) == 0:
        print(f"Warning: No frequencies found in the range {freq_range} Hz.")
        peak_freq = np.nan
    else:
        peak_idx = idx_range[np.argmax(avg_power_spectrum[idx_range])]
        peak_freq = frequencies[peak_idx]

    return peak_freq, avg_power_spectrum, frequencies


def calculate_voltage_std_synchrony(voltage_traces):
    """
    Calculates synchrony based on the standard deviation of voltage across neurons.
    Lower standard deviation indicates higher synchrony.

    Args:
        voltage_traces (np.ndarray or bm.ndarray): Membrane potential traces.
                                                 Shape: (num_steps, num_neurons)

    Returns:
        np.ndarray: Time series of the standard deviation of voltage across neurons.
    """
    if isinstance(voltage_traces, bm.ndarray):
        voltage_traces = voltage_traces.to_numpy()

    if voltage_traces.shape[1] <= 1:
        # Standard deviation is 0 (or undefined) for 0 or 1 neuron
        return np.zeros(voltage_traces.shape[0])

    # Calculate standard deviation across the neuron dimension (axis=1)
    voltage_std = np.std(voltage_traces, axis=1)

    return voltage_std


def calculate_pairwise_correlation_binary(binary_series):
    """
    Calculates the average pairwise Pearson correlation between binary time series.

    Useful for spike train synchrony (after binning) or event synchrony.

    Args:
        binary_series (np.ndarray): Array of binary time series.
                                    Shape: (num_steps, num_neurons)

    Returns:
        float: Average Pearson correlation coefficient across unique pairs.
               Returns np.nan if fewer than 2 neurons or in case of errors.
    """
    if not isinstance(binary_series, np.ndarray):
        # Try converting if it's a BrainPy array, otherwise raise error
        if hasattr(binary_series, "to_numpy"):
            binary_series = binary_series.to_numpy()
        else:
            raise TypeError(
                "Input 'binary_series' must be a NumPy array or convertible."
            )

    num_steps, num_neurons = binary_series.shape

    if num_neurons < 2:
        return np.nan  # Correlation requires at least 2 series

    correlations = []
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):
            series_i = binary_series[:, i].astype(float)  # Ensure float for correlation
            series_j = binary_series[:, j].astype(float)

            # Check for zero standard deviation (e.g., neuron never active/inactive)
            std_i = np.std(series_i)
            std_j = np.std(series_j)

            if std_i > 1e-9 and std_j > 1e-9:  # Avoid numerical issues
                # Using np.corrcoef returns a 2x2 matrix
                corr_matrix = np.corrcoef(series_i, series_j)
                correlations.append(corr_matrix[0, 1])
            elif std_i < 1e-9 and std_j < 1e-9:
                # If both series are constant, they are perfectly correlated
                correlations.append(1.0)
            else:
                # If one is constant and the other is not, correlation is undefined (or zero)
                # Depending on interpretation, let's append 0 or skip. Appending 0.
                correlations.append(0.0)

    if not correlations:
        return np.nan  # Should not happen if num_neurons >= 2, but safety check

    return np.mean(correlations)


def calculate_avg_cv_isi_from_spikes(spike_train, dt):
    """
    Calculates the average Coefficient of Variation (CV) of Inter-Spike Intervals (ISIs)
    across a population of neurons from their spike trains.

    CV = std(ISIs) / mean(ISIs). CV > 1 suggests burstiness, CV < 1 suggests regularity.

    Args:
        spike_train (np.ndarray or bm.ndarray): Boolean or binary array of spikes.
                                                Shape: (num_steps, num_neurons)
        dt (float): Simulation time step in ms.

    Returns:
        float: Average CV across all neurons that fired at least two spikes.
               Returns np.nan if no neuron fired >= 2 spikes.
    """
    if isinstance(spike_train, bm.ndarray):
        spike_train = spike_train.to_numpy()

    num_steps, num_neurons = spike_train.shape
    all_cvs = []

    for i in range(num_neurons):
        spike_indices = np.where(spike_train[:, i] > 0)[0]
        if len(spike_indices) >= 2:
            spike_times_ms = spike_indices * dt
            isis_ms = np.diff(spike_times_ms)

            if len(isis_ms) > 0:
                mean_isi = np.mean(isis_ms)
                std_isi = np.std(isis_ms)

                if mean_isi > 1e-9:  # Avoid division by zero
                    cv = std_isi / mean_isi
                    all_cvs.append(cv)
                # else: CV is undefined or could be considered 0 if std is also 0

    if not all_cvs:
        return np.nan  # No neurons fired enough to calculate CV

    return np.mean(all_cvs)


def calculate_population_rate(spike_train, dt, bin_width_ms):
    """
    Calculates the total population firing rate in Hz over time bins.

    Args:
        spike_train (np.ndarray or bm.ndarray): Boolean or binary array of spikes.
                                                Shape: (num_steps, num_neurons)
        dt (float): Simulation time step in ms.
        bin_width_ms (float): Width of the time bins in ms.

    Returns:
        tuple: (population_rate_hz, bin_centers_ms)
               - population_rate_hz (np.ndarray): Population rate in Hz for each bin.
               - bin_centers_ms (np.ndarray): Time corresponding to the center of each bin (in ms).
    """
    if isinstance(spike_train, bm.ndarray):
        spike_train = spike_train.to_numpy()

    num_steps, num_neurons = spike_train.shape
    if num_neurons == 0:
        return np.array([]), np.array([])

    steps_per_bin = max(1, int(round(bin_width_ms / dt)))
    num_bins = num_steps // steps_per_bin
    bin_width_sec = (steps_per_bin * dt) / 1000.0

    if num_bins == 0:
        return np.array([]), np.array([])  # Not enough data for even one bin

    # Sum spikes across neurons first
    total_spikes_per_step = np.sum(spike_train, axis=1)

    # Truncate to fit whole bins
    total_spikes_per_step_trunc = total_spikes_per_step[: num_bins * steps_per_bin]

    # Reshape and sum within bins
    spikes_in_bins = np.sum(
        total_spikes_per_step_trunc.reshape(num_bins, steps_per_bin), axis=1
    )

    # Calculate rate in Hz
    population_rate_hz = spikes_in_bins / bin_width_sec

    # Calculate bin centers
    bin_start_times_ms = np.arange(num_bins) * bin_width_ms
    bin_centers_ms = bin_start_times_ms + bin_width_ms / 2.0

    return population_rate_hz, bin_centers_ms


def calculate_relative_power(
    time_series, dt, target_band_hz, analysis_band_hz=(0.1, None), fs_scale=1000.0
):
    """
    Calculates the relative power of a signal within a target frequency band
    compared to a broader analysis band.

    Args:
        time_series (np.ndarray): 1D time series data (e.g., population rate).
        dt (float): Time step of the input series in ms.
        target_band_hz (tuple): (min_Hz, max_Hz) for the band of interest.
        analysis_band_hz (tuple, optional): (min_Hz, max_Hz) for the normalization band.
                                          Defaults to (0.1 Hz, Nyquist frequency).
        fs_scale (float, optional): Factor to convert dt to sampling frequency (e.g., 1000.0 for ms).

    Returns:
        float: Ratio of power in target_band to power in analysis_band.
               Returns np.nan if analysis band power is zero or on error.
    """
    if not isinstance(time_series, np.ndarray):
        if hasattr(time_series, "to_numpy"):  # Handle BrainPy array
            time_series = time_series.to_numpy()
        else:
            raise TypeError("Input 'time_series' must be a NumPy array or convertible.")

    if time_series.ndim != 1:
        raise ValueError("Input 'time_series' must be 1-dimensional.")

    num_steps = len(time_series)
    if num_steps < 2:
        return np.nan  # Need at least 2 points for FFT

    fs = fs_scale / dt  # Sampling frequency in Hz
    nyquist = fs / 2.0

    # Detrend before FFT
    series_detrended = time_series - np.mean(time_series)

    # Compute FFT
    fft_vals = np.fft.fft(series_detrended)
    # Compute Power Spectrum (magnitude squared), take only positive frequencies
    power = np.abs(fft_vals[: num_steps // 2]) ** 2

    # Get corresponding frequencies
    frequencies = np.fft.fftfreq(num_steps, d=dt / fs_scale)[: num_steps // 2]

    # Define frequency indices for bands
    target_min, target_max = target_band_hz
    analysis_min, analysis_max = analysis_band_hz
    if analysis_max is None:
        analysis_max = nyquist

    target_idx = np.where((frequencies >= target_min) & (frequencies <= target_max))[0]
    analysis_idx = np.where(
        (frequencies >= analysis_min) & (frequencies <= analysis_max)
    )[0]

    if len(target_idx) == 0 or len(analysis_idx) == 0:
        # If either band is empty, result is meaningless or zero
        print(
            f"Warning: Frequency band resulted in empty index array. Target: {target_band_hz}, Analysis: {analysis_band_hz}"
        )
        return 0.0  # Or np.nan depending on desired behavior

    # Sum power in bands (using trapezoidal integration for better accuracy might be overkill here)
    power_target_band = np.sum(power[target_idx])
    power_analysis_band = np.sum(power[analysis_idx])

    if power_analysis_band < 1e-12:
        # Avoid division by zero if the analysis band has negligible power
        return np.nan

    relative_power = power_target_band / power_analysis_band

    return relative_power
