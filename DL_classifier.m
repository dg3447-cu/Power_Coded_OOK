%% MC-OOK Classifier - Low Power Optimized (16-DFT Features Only)
clear; clc; close all;

% --- Configuration Parameters ---
num_subcarriers = 4;
carrier_freq = 400e6;
BW = 10e6;
cycles_per_bit = 100;
samples_per_cycle = 20;
num_dft_points = 16;
num_classes = 16; % 16 possible messages

% SNR range for training - focus on realistic conditions
SNR_dB_range = [0, 1, 2, 3, 5, 7, 10]; % Various noise levels
samples_per_message = 200;
total_samples = num_classes * samples_per_message * length(SNR_dB_range);

fprintf('Generating dataset with %d samples (16 features each)...\n', total_samples);

% --- Precompute RF parameters ---
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

% --- Generate Dataset ---
X = zeros(total_samples, num_dft_points); % Only 16 DFT features
Y = zeros(total_samples, 1);
all_messages = dec2bin(0:15, 4) - '0';

sample_idx = 1;
fprintf('Generating optimized dataset...\n');

for msg_idx = 1:num_classes
    message = all_messages(msg_idx, :);
    
    for snr_idx = 1:length(SNR_dB_range)
        snr_db = SNR_dB_range(snr_idx);
        
        for noise_iter = 1:samples_per_message
            % Generate signal
            signal_matrix = zeros(num_subcarriers, samples_per_bit);
            for k = 1:num_subcarriers
                if message(k) == 1
                    signal_matrix(k, :) = carrier_waves(k, :);
                end
            end
            
            sig_clean = sum(signal_matrix, 1);
            sig_noisy = awgn(sig_clean, snr_db, 'measured');
            
            % Simple envelope detection (abs is cheap on MCU)
            envelope = abs(sig_noisy);
            
            % Downsample to 16 points (MCU-friendly averaging)
            % This simulates what would happen on a low-power device
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
            
            % Store features
            X(sample_idx, :) = dft_magnitude;
            Y(sample_idx) = msg_idx;
            sample_idx = sample_idx + 1;
        end
    end
    fprintf('Generated message %d/%d\n', msg_idx, num_classes);
end

% --- Data Preprocessing ---
fprintf('Preprocessing data...\n');
% Simple normalization suitable for MCU
X_mean = mean(X);
X_std = std(X);
X_normalized = (X - X_mean) ./ X_std;
X_normalized(isnan(X_normalized)) = 0;

% --- Split dataset ---
rng(42); % For reproducibility
cv = cvpartition(length(Y), 'HoldOut', 0.2);
idxTrain = training(cv);
idxVal = test(cv);

XTrain = X_normalized(idxTrain, :);
YTrain = Y(idxTrain);
XVal = X_normalized(idxVal, :);
YVal = Y(idxVal);

YTrain_cat = categorical(YTrain);
YVal_cat = categorical(YVal);

fprintf('Training set: %d samples\n', sum(idxTrain));
fprintf('Validation set: %d samples\n', sum(idxVal));
fprintf('Feature size: %d (MCU-friendly)\n', size(XTrain, 2));

% --- Create MCU-Optimized Neural Network ---
fprintf('Creating MCU-optimized neural network...\n');

% Small network architecture suitable for low-power devices
layers = [
    featureInputLayer(num_dft_points, 'Name', 'input')  % 16 inputs
    
    % Hidden layers (small for MCU)
    fullyConnectedLayer(12, 'Name', 'fc1')  % Reduced from 64
    reluLayer('Name', 'relu1')
    
    fullyConnectedLayer(8, 'Name', 'fc2')   % Reduced from 32
    reluLayer('Name', 'relu2')
    
    % Output layer
    fullyConnectedLayer(num_classes, 'Name', 'output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

% Training options for better generalization
options = trainingOptions('adam', ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.002, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.6, ...
    'LearnRateDropPeriod', 25, ...
    'L2Regularization', 0.0005, ...  % Increased regularization
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal_cat}, ...
    'ValidationFrequency', 40, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

% --- Train the Model ---
fprintf('Training MCU-optimized network...\n');
[net, trainInfo] = trainNetwork(XTrain, YTrain_cat, layers, options);

% --- Evaluate ---
fprintf('Evaluating...\n');
YPred = classify(net, XVal);
accuracy = mean(YPred == YVal_cat);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Per-class accuracy
confusion_mat = confusionmat(YVal, grp2idx(YPred));
class_accuracy = diag(confusion_mat) ./ sum(confusion_mat, 2) * 100;

fprintf('\nPer-class accuracy:\n');
for i = 1:num_classes
    fprintf('Message [%d%d%d%d]: %.1f%%\n', all_messages(i,:), class_accuracy(i));
end

% --- Test Specific Examples ---
fprintf('\nTesting specific examples:\n');
test_messages = [0 1 1 0; 1 0 1 0; 1 1 1 1; 0 0 0 0];

for i = 1:size(test_messages, 1)
    test_msg = test_messages(i, :);
    msg_idx = find(ismember(all_messages, test_msg, 'rows'));
    
    for snr_db = [0, 3, 5, 10]
        % Generate features using MCU-friendly method
        features = extract_features_mcu(test_msg, snr_db, carrier_waves, samples_per_bit, num_dft_points);
        features_normalized = (features - X_mean) ./ X_std;
        features_normalized(isnan(features_normalized)) = 0;
        
        % Predict
        prediction = classify(net, features_normalized);
        predicted_idx = double(prediction);
        predicted_msg = all_messages(predicted_idx, :);
        
        is_correct = all(predicted_msg == test_msg);
        fprintf('Message [%d%d%d%d] at %d dB: Predicted [%d%d%d%d] - %s\n', ...
                test_msg, snr_db, predicted_msg, ternary(is_correct, 'CORRECT', 'WRONG'));
    end
end

% --- Save Model ---
save('mcu_optimized_classifier.mat', 'net', 'X_mean', 'X_std', 'all_messages', 'trainInfo');

fprintf('\nMCU-optimized training complete! Model saved.\n');
fprintf('Network architecture: 16 -> 12 -> 8 -> 16 (perfect for low-power devices)\n');

% --- MCU-Friendly Feature Extraction Function ---
function features = extract_features_mcu(message, snr_db, carrier_waves, samples_per_bit, num_dft_points)
    % Generate signal
    signal_matrix = zeros(4, samples_per_bit);
    for k = 1:4
        if message(k) == 1
            signal_matrix(k, :) = carrier_waves(k, :);
        end
    end
    
    sig_clean = sum(signal_matrix, 1);
    sig_noisy = awgn(sig_clean, snr_db, 'measured');
    
    % Simple envelope detection
    envelope = abs(sig_noisy);
    
    % Downsample to 16 points (MCU-friendly)
    downsampled = zeros(1, num_dft_points);
    samples_per_bin = floor(length(envelope) / num_dft_points);
    for i = 1:num_dft_points
        start_idx = (i-1)*samples_per_bin + 1;
        end_idx = min(i*samples_per_bin, length(envelope));
        downsampled(i) = mean(envelope(start_idx:end_idx));
    end
    
    % 16-point DFT
    dft_magnitude = abs(fft(downsampled, num_dft_points));
    
    % MCU-friendly normalization
    dft_magnitude = dft_magnitude - min(dft_magnitude);
    if max(dft_magnitude) > 0
        dft_magnitude = dft_magnitude / max(dft_magnitude);
    end
    
    features = dft_magnitude;
end

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
