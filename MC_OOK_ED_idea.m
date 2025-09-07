% Parameters
message = [1 1 0 0];
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
    % Get the bit sequence for this subcarrier
    bits = subcarrier_matrix(k, :);
    
    % Create the base carrier wave for one *complete, seamless* bit period
    carrier_wave = sin(2 * pi * subcarriers(k) * t_bit);
    
    % For each bit, assign the carrier wave or zeros
    check_edge_cases = sum(subcarrier_matrix);
    for bit_idx = 1:bits_per_subcarrier
        start_sample = (bit_idx-1) * samples_per_bit + 1;
        end_sample = bit_idx * samples_per_bit;
        
        if (bits(bit_idx) == 1) && (check_edge_cases(bit_idx) ~= 1)
            % insert the pre-computed carrier wave segment
            signal_matrix(k, start_sample:end_sample) = carrier_wave;
        else % AM modulate the signal instead of it's an edge case
            signal_matrix(k, start_sample:end_sample) = carrier_wave .* sin(2 * pi * k * t_bit);
        end
    end
end

% Combine all subcarriers into one signal (MC-OOK)
sig_MC_OOK = sum(signal_matrix, 1);

% Plot all four individual subcarriers
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

% Plot the combined MC-OOK signal
subplot(5, 1, 5);
plot(t * 1e6, sig_MC_OOK, 'LineWidth', 1.5, 'Color', [0.8, 0.2, 0.2]);
xlabel('Time (µs)');
ylabel('Amplitude');
title('Combined MC-OOK Signal (Sum of all 4 Subcarriers)', 'FontWeight', 'bold');

% Calculate the upper envelope of the MC-OOK signal using the Hilbert transform
analytic_signal = hilbert(sig_MC_OOK);
amplitude_envelope = abs(analytic_signal);

figure;
subplot(2, 1, 1)
plot(t * 1e6, amplitude_envelope, 'r-', 'LineWidth', 1.5);
xlabel('Time (µs)');
ylabel('Amplitude');
title('Upper Envelope of MC-OOK Signal');
grid on;

% === High-Resolution FFT (using zero-padding) ===
N = length(amplitude_envelope);
dt = t(2) - t(1);
Fs_X = 1/dt;

% Zero-padding factor
padding_factor = 10; % Improve resoltion with zero padding
N_fft = padding_factor * N;

% Perform zero-padded FFT
X = fft(amplitude_envelope, N_fft);
X_shifted = fftshift(X);
f = (-N_fft / 2 : N_fft / 2 - 1) * (Fs_X / N_fft);
X_magnitude = abs(X_shifted) / N;

subplot(2, 1, 2)
plot(f / 1e6, X_magnitude, 'b', 'LineWidth', 1.2);
xlabel('Frequency (MHz)');
ylabel('Magnitude');
title("FFT of Envelope");

zoom = 30; 
xlim([-zoom zoom]);

% Notes:
% All DFTs are unique and can be mapped to unique codes, OUTSIDE OF the
% edge case that only a single 1 is present in the OFDM symbol
% Solution: for each OFDM symbol that only contains one 1, transmit a
% custom AM signal (instead of OOK), envelope detect that, and map the DFT
% of the result to the corresponding source code.
