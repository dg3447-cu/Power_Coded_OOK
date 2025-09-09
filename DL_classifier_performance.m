% Test script that evaluates the performance of the neural network

clear; clc; close all;

% === LOAD TRAINED NETWORK ===
load('mc_ook_classifier_optimized.mat'); % Loads net, X_mean, X_std, all_messages

% === TEST PARAMETERS ===
num_tests = 16; % Test all 16 messages
num_noisy_trials = 10; % Number of noise realizations per message
SNR_dB = 5; % Signal-to-Noise Ratio for testing

% === RF PARAMETERS (Must match training) ===
num_subcarriers = 4;
carrier_freq = 400e6;
BW = 10e6;
cycles_per_bit = 100;
samples_per_cycle = 20;
Fs = carrier_freq * samples_per_cycle;
Ts = 1 / Fs;
samples_per_bit = round(cycles_per_bit * (Fs / carrier_freq));
t_bit = (0:samples_per_bit-1) * Ts;
subcarriers = linspace(carrier_freq - BW/2, carrier_freq + BW/2, num_subcarriers);

% Precompute carrier waves (must match training)
carrier_waves = zeros(num_subcarriers, samples_per_bit);
for k = 1:num_subcarriers
    carrier_waves(k, :) = sin(2 * pi * subcarriers(k) * t_bit);
end

% === TEST ALL MESSAGES ===
test_results = struct();

fprintf('Testing neural network on %d messages with %d noise trials each...\n', ...
        num_tests, num_noisy_trials);

% Store example data for plotting
example_data = [];

for test_idx = 1:num_tests
    true_message = all_messages(test_idx, :);
    correct_predictions = 0;
    
    fprintf('\nTesting message %d: [%d %d %d %d]\n', ...
            test_idx, true_message);
    
    for trial = 1:num_noisy_trials
        % === GENERATE AND EXTRACT ALL FEATURES (like training) ===
        [all_features, noisy_env] = extract_all_features(true_message, SNR_dB, carrier_waves, samples_per_bit);
        
        % === PREPROCESS (Normalize using training statistics) ===
        features_normalized = (all_features - X_mean) ./ X_std;
        features_normalized(isnan(features_normalized)) = 0;
        
        % === PREDICT USING NEURAL NETWORK ===
        prediction = classify(net, features_normalized);
        predicted_idx = double(prediction);
        predicted_bits = all_messages(predicted_idx, :);
        
        % Check if prediction is correct
        is_correct = all(predicted_bits == true_message);
        if is_correct
            correct_predictions = correct_predictions + 1;
        end
        
        % Store data from first trial for plotting
        if trial == 1
            % Extract just the DFT part for plotting (first 16 features after envelope)
            dft_features = all_features(21:36); % Positions 21-36 are DFT magnitudes
            dft_normalized = features_normalized(21:36);
            
            example_data.true_message = true_message;
            example_data.noisy_env = noisy_env;
            example_data.dft_features = dft_features;
            example_data.dft_normalized = dft_normalized;
            example_data.predicted_bits = predicted_bits;
            example_data.is_correct = is_correct;
            
            fprintf('  Trial %d: Predicted [%d %d %d %d] - %s\n', ...
                    trial, predicted_bits, ...
                    ternary(is_correct, 'CORRECT', 'WRONG'));
        end
    end
    
    accuracy = correct_predictions / num_noisy_trials * 100;
    test_results(test_idx).message = true_message;
    test_results(test_idx).accuracy = accuracy;
    test_results(test_idx).trials = num_noisy_trials;
    
    fprintf('  Message Accuracy: %.1f%% (%d/%d)\n', ...
            accuracy, correct_predictions, num_noisy_trials);
end

% === OVERALL STATISTICS ===
fprintf('\n=== OVERALL RESULTS ===\n');
overall_accuracy = mean([test_results.accuracy]);
fprintf('Overall Accuracy: %.1f%%\n', overall_accuracy);
fprintf('SNR: %d dB\n', SNR_dB);

% === PLOT EXAMPLE RESULTS ===
if ~isempty(example_data)
    plot_example_results(example_data, Fs, 16, SNR_dB); % 16 DFT points
end

% === HELPER FUNCTIONS ===
function [all_features, noisy_env] = extract_all_features(message, snr_db, carrier_waves, samples_per_bit)
    % Generate MC-OOK signal with specified SNR
    signal_matrix = zeros(4, samples_per_bit);
    for k = 1:4
        if message(k) == 1
            signal_matrix(k, :) = carrier_waves(k, :);
        end
    end
    
    sig_MC_OOK = sum(signal_matrix, 1);
    sig_MC_OOK_noisy = awgn(sig_MC_OOK, snr_db, 'measured');
    
    % Envelope detection
    analytic_signal = hilbert(sig_MC_OOK_noisy);
    noisy_env = abs(analytic_signal);
    
    % --- Extract ALL features (must match training exactly) ---
    all_features = [];
    
    % 1. Fixed number of envelope samples (20 points)
    env_samples = 20;
    if length(noisy_env) >= env_samples
        all_features = [all_features, noisy_env(1:env_samples)];
    else
        % Pad if shorter
        padded = [noisy_env, zeros(1, env_samples - length(noisy_env))];
        all_features = [all_features, padded];
    end
    
    % 2. Fixed DFT points (16)
    dft_points = 16;
    dft_magnitude = abs(fft(noisy_env, dft_points));
    all_features = [all_features, dft_magnitude];
    
    % 3. Spectral features (3)
    power_spectrum = dft_magnitude.^2;
    frequencies = 0:(dft_points-1);
    
    spectral_centroid = sum(frequencies .* power_spectrum) / (sum(power_spectrum) + 1e-6);
    spectral_spread = sqrt(sum(((frequencies - spectral_centroid).^2) .* power_spectrum) / (sum(power_spectrum) + 1e-6));
    
    geometric_mean = exp(mean(log(dft_magnitude + 1e-6)));
    arithmetic_mean = mean(dft_magnitude);
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-6);
    
    all_features = [all_features, spectral_centroid, spectral_spread, spectral_flatness];
    
    % 4. Statistical features (4)
    stats = [mean(noisy_env), std(noisy_env), ...
             skewness(noisy_env), kurtosis(noisy_env)];
    all_features = [all_features, stats];
    
    % Ensure exact size (43 features)
    target_size = 20 + 16 + 3 + 4;
    if length(all_features) > target_size
        all_features = all_features(1:target_size);
    elseif length(all_features) < target_size
        all_features = [all_features, zeros(1, target_size - length(all_features))];
    end
end

function plot_example_results(example_data, Fs, num_dft_points, snr_db)
    
    % 1. Noisy Envelope Signal
    subplot(3, 2, 1);
    plot((0:length(example_data.noisy_env)-1) * (1/Fs) * 1e6, example_data.noisy_env, 'LineWidth', 1.5);
    title('Noisy Envelope Signal (Time Domain)');
    xlabel('Time (\mus)');
    ylabel('Amplitude');
    grid on;
    
    % 2. High-Resolution FFT of envelope
    subplot(3, 2, 2);
    N = length(example_data.noisy_env);
    f = linspace(0, Fs/2, floor(N/2)+1) / 1e6;
    Y = abs(fft(example_data.noisy_env));
    Y = Y(1:floor(N/2)+1);
    Y = Y / max(Y);
    plot(f, Y, 'LineWidth', 1.5);
    title('FFT of Noisy Envelope');
    xlabel('Frequency (MHz)');
    ylabel('Normalized Magnitude');
    grid on;
    xlim([0 50]);
    
    % 3. Raw 16-point DFT Magnitude
    subplot(3, 2, 3);
    stem(0:num_dft_points-1, example_data.dft_features, 'filled', 'LineWidth', 1.5);
    title('16-Point DFT Magnitude (Raw)');
    xlabel('DFT Bin Index');
    ylabel('Magnitude');
    grid on;
    
    % 4. Normalized 16-point DFT
    subplot(3, 2, 4);
    stem(0:num_dft_points-1, example_data.dft_normalized, 'filled', 'LineWidth', 1.5, ...
         'Color', [0.8, 0.2, 0.2]);
    title('16-Point DFT (Normalized)');
    xlabel('DFT Bin Index');
    ylabel('Normalized Magnitude');
    grid on;
    
    % 5. Message comparison
    subplot(3, 2, [5, 6]);
    messages = {['True: [' num2str(example_data.true_message) ']'], ...
                ['Pred: [' num2str(example_data.predicted_bits) ']']};
    bar_data = [example_data.true_message; example_data.predicted_bits]';
    
    bar(bar_data);
    set(gca, 'XTickLabel', {'Bit 1', 'Bit 2', 'Bit 3', 'Bit 4'});
    legend(messages, 'Location', 'best');
    title(sprintf('Message Comparison (SNR = %d dB)', snr_db));
    ylabel('Bit Value');
    ylim([-0.1 1.1]);
    grid on;
    
    % Add accuracy text
    result_text = ternary(example_data.is_correct, 'CORRECT', 'WRONG');
    text(2, 0.5, result_text, 'FontSize', 14, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', ...
         'Color', ternary(example_data.is_correct, [0, 0.5, 0], [0.8, 0, 0]));
end

function result = ternary(condition, true_val, false_val)
    % Ternary operator helper function
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
