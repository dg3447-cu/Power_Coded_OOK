%% Neural Network Tester for MC-OOK Classification - MCU OPTIMIZED
clc; clear; close all;

% === LOAD TRAINED NETWORK ===
load('mcu_optimized_classifier.mat'); % Loads net, X_mean, X_std, all_messages

% === TEST PARAMETERS ===
num_tests = 16; % Test all 16 messages
num_noisy_trials = 10; % Number of noise realizations per message
SNR_dB = 20; % Signal-to-Noise Ratio for testing

% === RF PARAMETERS (Must match training) ===
num_subcarriers = 4;
carrier_freq = 400e6;
BW = 10e6;
cycles_per_bit = 100;
samples_per_cycle = 20;
num_dft_points = 16; % Only 16 features now
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

fprintf('Testing MCU-optimized neural network on %d messages with %d noise trials each...\n', ...
        num_tests, num_noisy_trials);

% Store example data for plotting
example_data = [];

for test_idx = 1:num_tests
    true_message = all_messages(test_idx, :);
    correct_predictions = 0;
    
    fprintf('\nTesting message %d: [%d %d %d %d]\n', ...
            test_idx, true_message);
    
    for trial = 1:num_noisy_trials
        % === GENERATE MCU-FRIENDLY FEATURES (16-point DFT only) ===
        [features, noisy_env] = extract_features_mcu(true_message, SNR_dB, carrier_waves, samples_per_bit, num_dft_points);
        
        % === PREPROCESS (Normalize using training statistics) ===
        features_normalized = (features - X_mean) ./ X_std;
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
            example_data.true_message = true_message;
            example_data.noisy_env = noisy_env;
            example_data.dft_features = features; % Already the 16 DFT features
            example_data.dft_normalized = features_normalized;
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
fprintf('Feature size: %d (MCU-optimized)\n', num_dft_points);

% === PLOT EXAMPLE RESULTS ===
if ~isempty(example_data)
    plot_example_results(example_data, Fs, num_dft_points, SNR_dB);
end

% === MCU-FRIENDLY FEATURE EXTRACTION FUNCTION ===
function [features, noisy_env] = extract_features_mcu(message, snr_db, carrier_waves, samples_per_bit, num_dft_points)
    % Generate signal
    signal_matrix = zeros(4, samples_per_bit);
    for k = 1:4
        if message(k) == 1
            signal_matrix(k, :) = carrier_waves(k, :);
        end
    end
    
    sig_clean = sum(signal_matrix, 1);
    sig_noisy = awgn(sig_clean, snr_db, 'measured');
    
    % Store the noisy envelope for plotting
    noisy_env = abs(sig_noisy);
    
    % Simple envelope detection (abs is cheap on MCU)
    envelope = abs(sig_noisy);
    
    % Downsample to 16 points (MCU-friendly averaging)
    downsampled = zeros(1, num_dft_points);
    samples_per_bin = floor(length(envelope) / num_dft_points);
    for i = 1:num_dft_points
        start_idx = (i-1)*samples_per_bin + 1;
        end_idx = min(i*samples_per_bin, length(envelope));
        downsampled(i) = mean(envelope(start_idx:end_idx));
    end
    
    % 16-point DFT (the main feature extraction)
    dft_magnitude = abs(fft(downsampled, num_dft_points));
    
    % MCU-friendly normalization: remove DC, scale to [0,1]
    dft_magnitude = dft_magnitude - min(dft_magnitude);
    if max(dft_magnitude) > 0
        dft_magnitude = dft_magnitude / max(dft_magnitude);
    end
    
    features = dft_magnitude;
end

function plot_example_results(example_data, Fs, num_dft_points, snr_db)
    % Plot results for the example data
    figure('Name', 'MCU-Optimized Neural Network Test Results', 'NumberTitle', 'off', ...
           'Position', [100, 100, 1000, 800]);
    
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
    
    % 4. Normalized 16-point DFT (Network Input)
    subplot(3, 2, 4);
    stem(0:num_dft_points-1, example_data.dft_normalized, 'filled', 'LineWidth', 1.5, ...
         'Color', [0.8, 0.2, 0.2]);
    title('16-Point DFT (Normalized - Network Input)');
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
