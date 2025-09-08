% === Parameters ===
message = [1 0 1 0];
msg_length = length(message);
num_subcarriers = 4;
bits_per_subcarrier = msg_length / num_subcarriers;

% RF parameters
carrier_freq = 400e6; % Center frequency
BW = 10e6; % Bandwidth

% Coherent sampling parameters
cycles_per_bit = 100;
samples_per_cycle = 20;
Fs = carrier_freq * samples_per_cycle;
Ts = 1 / Fs;

% Calculate samples_per_bit based on desired cycles_per_bit
samples_per_bit = round(cycles_per_bit * (Fs / carrier_freq));
cycles_per_bit = samples_per_bit * (carrier_freq / Fs);

% Time for one full bit
t_bit = (0:samples_per_bit-1) * Ts;
total_samples = bits_per_subcarrier * samples_per_bit;
t = (0:total_samples-1) * Ts;

% Subcarrier frequencies
subcarrier_spacing = BW / (num_subcarriers - 1);
subcarriers = linspace(carrier_freq - BW/2, carrier_freq + BW/2, num_subcarriers);

% Split message into subcarrier rows
subcarrier_matrix = zeros(num_subcarriers, bits_per_subcarrier);
for i = 1:num_subcarriers
    start_idx = (i - 1) * bits_per_subcarrier + 1;
    end_idx = i * bits_per_subcarrier;
    subcarrier_matrix(i, :) = message(start_idx:end_idx);
end

% Generate OOK-modulated signals per subcarrier
signal_matrix = zeros(num_subcarriers, total_samples);

for k = 1:num_subcarriers
    bits = subcarrier_matrix(k, :);
    carrier_wave = sin(2 * pi * subcarriers(k) * t_bit);  % row vector
    check_edge_cases = sum(subcarrier_matrix);  % column-wise sum

    for bit_idx = 1:bits_per_subcarrier
        start_sample = (bit_idx-1) * samples_per_bit + 1;
        end_sample = bit_idx * samples_per_bit;
        
        if (bits(bit_idx) == 1) && (check_edge_cases(bit_idx) ~= 1)
            % Normal: assign carrier
            signal_matrix(k, start_sample:end_sample) = carrier_wave;
        else
            % Edge case
            edge_code = subcarrier_matrix(:, bit_idx);
            subcarrier_loc = find(edge_code == 1, 1);  % Ensure scalar
            mod_wave = sin(2 * pi * (subcarrier_loc * 1e6) * t_bit);  % Row vector
            signal_matrix(k, start_sample:end_sample) = carrier_wave .* mod_wave;
        end
    end
end

% Combine subcarriers
sig_MC_OOK = sum(signal_matrix, 1);

% === Plot individual subcarriers ===
figure;
for k = 1:num_subcarriers
    subplot(5, 1, k);
    plot(t * 1e6, signal_matrix(k, :), 'LineWidth', 1.2);
    ylabel(['SC ' num2str(k)], 'FontWeight', 'bold');
    title(['Subcarrier ' num2str(k) ': ' num2str(subcarriers(k)/1e6, '%.2f') ' MHz']);
    ylim([-1.2 1.2]);
    grid on;
    
    if k == num_subcarriers
        xlabel('Time (µs)');
    end
end

subplot(5, 1, 5);
plot(t * 1e6, sig_MC_OOK, 'LineWidth', 1.5, 'Color', [0.8, 0.2, 0.2]);
xlabel('Time (µs)');
ylabel('Amplitude');
title('Combined MC-OOK Signal (Sum of all 4 Subcarriers)', 'FontWeight', 'bold');

% === Envelope using Hilbert Transform ===
analytic_signal = hilbert(sig_MC_OOK);
amplitude_envelope = abs(analytic_signal);

% === High-Resolution FFT ===
N = length(amplitude_envelope);
dt = t(2) - t(1);
Fs_X = 1/dt;
padding_factor = 4;
N_fft = padding_factor * N;
X = fft(amplitude_envelope, N_fft);
X_shifted = fftshift(X);
f = (-N_fft / 2 : N_fft / 2 - 1) * (Fs_X / N_fft);
X_magnitude = abs(X_shifted) / N;

% === 16-point DFT (for on-chip implementation) ===
num_dft_points = 16;
sample_indices = round(linspace(1, length(amplitude_envelope), num_dft_points));
dft_input = amplitude_envelope(sample_indices);
DFT_16 = fft(dft_input, 16);
f_16 = (0:15) * (Fs / 16);  % Frequency in Hz

% === Second Figure: Envelope + FFT + 16-point DFT ===
figure;

% 1. Envelope
subplot(3, 1, 1);
plot(t * 1e6, amplitude_envelope, 'r-', 'LineWidth', 1.5);
xlabel('Time (µs)');
ylabel('Amplitude');
title('Upper Envelope of MC-OOK Signal');
grid on;

% 2. Ideal FFT
subplot(3, 1, 2);
plot(f / 1e6, X_magnitude, 'b', 'LineWidth', 1.2);
xlabel('Frequency (MHz)');
ylabel('Magnitude');
title('High-Resolution FFT of Envelope');
grid on;
xlim([-30 30]);

% 3. 16-Point DFT
subplot(3, 1, 3);
stem(f_16 / 1e6, abs(DFT_16)/max(abs(DFT_16)), 'filled', 'LineWidth', 1.2);
xlabel('Frequency (MHz)');
ylabel('Magnitude');
title('Normalized 16-Point DFT of Envelope');
grid on;
