%% Neural Network Tester - Waveform Prediction Analysis
clc; clear; close all;

% === LOAD TRAINED NETWORK ===
load('high_accuracy_classifier.mat'); % Loads net, X_mean, X_std, all_messages

% === TEST PARAMETERS ===
num_tests_per_message = 20; % Tests per message
SNR_dB_range = [0, 1, 3, 5, 7, 10, 15]; % Multiple SNR levels to test
num_subcarriers = 4;

% === RF PARAMETERS (Must match training) ===
carrier_freq = 400e6;
BW = 10e6;
cycles_per_bit = 100;
samples_per_cycle = 20;
target_samples = 100; % Downsample to 100 points
Fs = carrier_freq * samples_per_cycle;
Ts = 1 / Fs;
samples_per_bit = round(cycles_per_bit * (Fs / carrier_freq));
t_bit = (0:samples_per_bit-1) * Ts;
subcarriers = linspace(carrier_freq - BW/2, carrier_freq + BW/2, num_subcarriers);

% Precompute carrier waves
carrier_waves = zeros(num_subcarriers, samples_per_bit);
for k = 1:num_subcarriers
    carrier_waves(k, :) = sin(2 * pi * subcarriers(k) * t_bit);
end

% === COMPREHENSIVE TESTING ===
fprintf('=== Neural Network Waveform Prediction Test ===\n');
fprintf('Testing all 16 messages at multiple SNR levels...\n\n');

% Initialize results storage
results = struct();
confusion_matrix = zeros(16, 16); % For overall confusion matrix
snr_results = struct();

for snr_idx = 1:length(SNR_dB_range)
    current_snr = SNR_dB_range(snr_idx);
    fprintf('Testing at SNR: %d dB\n', current_snr);
    fprintf('============================================\n');
    
    snr_correct = 0;
    snr_total = 0;
    snr_confusion = zeros(16, 16);
    
    for msg_idx = 1:16
        true_message = all_messages(msg_idx, :);
        message_correct = 0;
        
        for test_iter = 1:num_tests_per_message
            % Generate enhanced features
            features = generate_enhanced_features(true_message, current_snr, carrier_waves, samples_per_bit, target_samples);
            features_normalized = (features - X_mean) ./ X_std;
            features_normalized(isnan(features_normalized)) = 0;
            
            % Predict
            prediction = classify(net, features_normalized);
            predicted_idx = grp2idx(prediction);
            predicted_msg = all_messages(predicted_idx, :);
            
            % Get prediction probabilities
            prediction_probs = predict(net, features_normalized);
            [max_prob, max_idx] = max(prediction_probs);
            
            % Check if correct
            is_correct = (predicted_idx == msg_idx);
            if is_correct
                message_correct = message_correct + 1;
                snr_correct = snr_correct + 1;
            end
            
            % Update confusion matrices
            confusion_matrix(msg_idx, predicted_idx) = confusion_matrix(msg_idx, predicted_idx) + 1;
            snr_confusion(msg_idx, predicted_idx) = snr_confusion(msg_idx, predicted_idx) + 1;
            
            snr_total = snr_total + 1;
            
            % Display first few results for each message
            if test_iter <= 3 && msg_idx <= 4 % Show first 3 tests for first 4 messages
                status = ternary(is_correct, '✓', '✗');
                fprintf('  %s [%d%d%d%d] -> [%d%d%d%d] (Conf: %.2f)\n', ...
                        status, true_message, predicted_msg, max_prob);
            end
        end
        
        message_accuracy = message_correct / num_tests_per_message * 100;
        fprintf('Message [%d%d%d%d]: %.1f%% accuracy (%d/%d)\n', ...
                true_message, message_accuracy, message_correct, num_tests_per_message);
    end
    
    % Store SNR results
    snr_accuracy = snr_correct / snr_total * 100;
    snr_results(snr_idx).SNR_dB = current_snr;
    snr_results(snr_idx).accuracy = snr_accuracy;
    snr_results(snr_idx).confusion = snr_confusion;
    
    fprintf('SNR %d dB Overall Accuracy: %.1f%%\n\n', current_snr, snr_accuracy);
end

% === OVERALL RESULTS ===
fprintf('\n=== OVERALL RESULTS ===\n');
overall_accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix(:)) * 100;
fprintf('Overall Accuracy: %.1f%%\n', overall_accuracy);

% Plot SNR vs Accuracy
figure
snr_values = [snr_results.SNR_dB];
accuracies = [snr_results.accuracy];

subplot(1,2,1);
plot(snr_values, accuracies, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
xlabel('SNR (dB)');
ylabel('Accuracy (%)');
title('Network Performance vs SNR');
grid on;
ylim([0 100]);
xlim([-1 max(snr_values)+1]);

% Add accuracy labels
for i = 1:length(snr_values)
    text(snr_values(i), accuracies(i)+2, sprintf('%.1f%%', accuracies(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Plot Confusion Matrix
subplot(1,2,2);
confusion_norm = confusion_matrix ./ sum(confusion_matrix, 2);
imagesc(confusion_norm);
colorbar;
colormap('hot');
title('Normalized Confusion Matrix');
xlabel('Predicted Message');
ylabel('True Message');
set(gca, 'XTick', 1:16, 'YTick', 1:16);

% === DETAILED ANALYSIS FOR SPECIFIC MESSAGES ===
fprintf('\n=== DETAILED MESSAGE ANALYSIS ===\n');
test_messages = [0 1 1 0; 1 0 1 0; 1 1 1 1; 0 0 0 0];

for i = 1:size(test_messages, 1)
    test_msg = test_messages(i, :);
    true_idx = find(ismember(all_messages, test_msg, 'rows'));
    
    fprintf('\nDetailed analysis for message [%d%d%d%d]:\n', test_msg);
    
    % Test at various SNR levels
    snr_levels = [0, 5, 10];
    for snr_idx = 1:length(snr_levels)
        snr_db = snr_levels(snr_idx);
        
        % Run multiple tests
        correct_count = 0;
        total_tests = 50;
        confidences = [];
        
        for test_iter = 1:total_tests
            features = generate_enhanced_features(test_msg, snr_db, carrier_waves, samples_per_bit, target_samples);
            features_normalized = (features - X_mean) ./ X_std;
            features_normalized(isnan(features_normalized)) = 0;
            
            prediction_probs = predict(net, features_normalized);
            [max_prob, predicted_idx] = max(prediction_probs);
            
            if predicted_idx == true_idx
                correct_count = correct_count + 1;
                confidences = [confidences, max_prob];
            end
        end
        
        accuracy = correct_count / total_tests * 100;
        avg_confidence = mean(confidences) * 100;
        
        fprintf('  SNR %d dB: %.1f%% accuracy, Avg confidence: %.1f%%\n', ...
                snr_db, accuracy, avg_confidence);
    end
end

% === WORST-CASE ANALYSIS ===
fprintf('\n=== WORST-CASE PERFORMANCE ===\n');
[min_accuracy, worst_msg] = min(diag(confusion_norm) * 100);
worst_message = all_messages(worst_msg, :);
fprintf('Worst performing message: [%d%d%d%d] (%.1f%% accuracy)\n', ...
        worst_message, min_accuracy);

% Show confusion for worst message
fprintf('Most common misclassifications:\n');
[~, sorted_idx] = sort(confusion_norm(worst_msg, :), 'descend');
for i = 2:4 % Skip the correct classification (should be first)
    if sorted_idx(i) ~= worst_msg && confusion_norm(worst_msg, sorted_idx(i)) > 0
        misclassified_msg = all_messages(sorted_idx(i), :);
        error_rate = confusion_norm(worst_msg, sorted_idx(i)) * 100;
        fprintf('  → [%d%d%d%d] (%.1f%% of errors)\n', misclassified_msg, error_rate);
    end
end

% === WAVEFORM VISUALIZATION ===
fprintf('\n=== WAVEFORM VISUALIZATION ===\n');
figure

% Show example waveforms for different messages
example_messages = [0 0 0 0; 0 1 1 0; 1 1 1 1; 1 0 1 0];
snr_to_show = 5;

for i = 1:4
    msg = example_messages(i, :);
    
    % Generate signal
    features = generate_enhanced_features(msg, snr_to_show, carrier_waves, samples_per_bit, target_samples);
    time_signal = features(1:target_samples); % First 100 points are time domain
    
    subplot(2,2,i);
    plot(time_signal, 'LineWidth', 2);
    title(sprintf('Message [%d%d%d%d] at %d dB SNR', msg, snr_to_show));
    xlabel('Sample Index');
    ylabel('Normalized Amplitude');
    grid on;
    ylim([-3 3]);
    
    % Add prediction info
    features_normalized = (features - X_mean) ./ X_std;
    prediction_probs = predict(net, features_normalized);
    [max_prob, predicted_idx] = max(prediction_probs);
    predicted_msg = all_messages(predicted_idx, :);
    
    is_correct = isequal(msg, predicted_msg);
    status = ternary(is_correct, 'CORRECT', 'WRONG');
    text(10, 2.5, sprintf('Predicted: [%d%d%d%d] (%s)', predicted_msg, status), ...
         'FontWeight', 'bold', 'Color', ternary(is_correct, [0, 0.5, 0], [0.8, 0, 0]));
    text(10, 2.0, sprintf('Confidence: %.1f%%', max_prob*100), 'FontWeight', 'bold');
end

fprintf('\nTest completed! Overall network performance: %.1f%% accuracy\n', overall_accuracy);

% === HELPER FUNCTION ===
function features = generate_enhanced_features(message, snr_db, carrier_waves, samples_per_bit, target_samples)
    % Generate signal with phase variations
    signal_matrix = zeros(4, samples_per_bit);
    phase_variation = 0.1 * randn(4, 1);
    
    % Get subcarrier frequencies
    carrier_freq = 400e6;
    BW = 10e6;
    subcarriers = linspace(carrier_freq - BW/2, carrier_freq + BW/2, 4);
    Fs = carrier_freq * 20;
    Ts = 1 / Fs;
    t_bit = (0:samples_per_bit-1) * Ts;
    
    for k = 1:4
        if message(k) == 1
            phase = 2 * pi * subcarriers(k) * t_bit + phase_variation(k);
            signal_matrix(k, :) = sin(phase);
        end
    end
    
    sig_clean = sum(signal_matrix, 1);
    sig_noisy = awgn(sig_clean, snr_db, 'measured');
    
    % Envelope detection with Hilbert
    analytic_signal = hilbert(sig_noisy);
    envelope = abs(analytic_signal);
    
    % Downsample
    downsampled = resample(envelope, target_samples, length(envelope));
    
    % Extract additional features
    dft_magnitude = abs(fft(envelope, 16));
    spectral_centroid = sum((0:15) .* dft_magnitude.^2) / (sum(dft_magnitude.^2) + 1e-6);
    rms_value = rms(envelope);
    peak_value = max(envelope);
    crest_factor = peak_value / (rms_value + 1e-6);
    
    % Normalize
    downsampled = (downsampled - mean(downsampled)) / (std(downsampled) + 1e-6);
    
    % Combine features
    features = [downsampled, spectral_centroid, rms_value, peak_value, crest_factor, snr_db/20];
end

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
