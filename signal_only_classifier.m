%% MC-OOK Classifier - High Accuracy Time Domain Model
clear; clc; close all;

% --- Configuration Parameters ---
num_subcarriers = 4;
carrier_freq = 400e6;
BW = 10e6;
cycles_per_bit = 100;
samples_per_cycle = 20;
num_classes = 16; % 16 possible messages
target_samples = 100; % Downsample to 100 points

% SNR range for training - focus on challenging conditions
SNR_dB_range = [0, 1, 2, 3, 4, 5, 7, 10, 15]; % More SNR levels
samples_per_message = 300; % Increased samples
total_samples = num_classes * samples_per_message * length(SNR_dB_range);

fprintf('Generating enhanced dataset with %d samples...\n', total_samples);

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

% --- Generate Enhanced Dataset ---
X = zeros(total_samples, target_samples + 5); % Time domain + extra features
Y = zeros(total_samples, 1);

all_messages = dec2bin(0:15, 4) - '0';

sample_idx = 1;
fprintf('Generating enhanced dataset with additional features...\n');

for msg_idx = 1:num_classes
    message = all_messages(msg_idx, :);
    
    for snr_idx = 1:length(SNR_dB_range)
        snr_db = SNR_dB_range(snr_idx);
        
        for noise_iter = 1:samples_per_message
            % Generate signal with slight variations
            signal_matrix = zeros(num_subcarriers, samples_per_bit);
            
            % Add small random phase variations to make it more realistic
            phase_variation = 0.1 * randn(num_subcarriers, 1);
            
            for k = 1:num_subcarriers
                if message(k) == 1
                    % Add slight phase variation for realism
                    phase = 2 * pi * subcarriers(k) * t_bit + phase_variation(k);
                    signal_matrix(k, :) = sin(phase);
                end
            end
            
            sig_clean = sum(signal_matrix, 1);
            sig_noisy = awgn(sig_clean, snr_db, 'measured');
            
            % Envelope detection with Hilbert transform
            analytic_signal = hilbert(sig_noisy);
            envelope = abs(analytic_signal);
            
            % Downsample with anti-aliasing
            downsampled = resample(envelope, target_samples, length(envelope));
            
            % Extract additional features
            dft_magnitude = abs(fft(envelope, 16));
            spectral_centroid = sum((0:15) .* dft_magnitude.^2) / (sum(dft_magnitude.^2) + 1e-6);
            rms_value = rms(envelope);
            peak_value = max(envelope);
            crest_factor = peak_value / (rms_value + 1e-6);
            
            % Normalize time domain signal
            downsampled = (downsampled - mean(downsampled)) / (std(downsampled) + 1e-6);
            
            % Combine time domain and additional features
            features = [downsampled, spectral_centroid, rms_value, peak_value, crest_factor, snr_db/20];
            
            % Store features and class label
            X(sample_idx, :) = features;
            Y(sample_idx) = msg_idx;
            sample_idx = sample_idx + 1;
        end
    end
    fprintf('Generated message %d/%d: [%d%d%d%d]\n', msg_idx, num_classes, message);
end

% --- Data Preprocessing ---
fprintf('Preprocessing data...\n');
X_mean = mean(X);
X_std = std(X);
X_normalized = (X - X_mean) ./ X_std;
X_normalized(isnan(X_normalized)) = 0;

% Convert to categorical for classification
Y_categorical = categorical(Y);

% --- Split dataset with stratification ---
rng(42);
cv = cvpartition(Y_categorical, 'HoldOut', 0.15);
idxTrain = training(cv);
idxVal = test(cv);

XTrain = X_normalized(idxTrain, :);
YTrain = Y_categorical(idxTrain);
XVal = X_normalized(idxVal, :);
YVal = Y_categorical(idxVal);

fprintf('Training set: %d samples\n', sum(idxTrain));
fprintf('Validation set: %d samples\n', sum(idxVal));
fprintf('Feature size: %d\n', size(XTrain, 2));

% --- Create Enhanced Neural Network ---
fprintf('Creating enhanced neural network...\n');

layers = [
    featureInputLayer(size(XTrain, 2), 'Name', 'input')
    
    % First layers with more capacity
    fullyConnectedLayer(256, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    leakyReluLayer(0.1, 'Name', 'leaky_relu1')
    dropoutLayer(0.4, 'Name', 'dropout1')
    
    fullyConnectedLayer(256, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    leakyReluLayer(0.1, 'Name', 'leaky_relu2')
    dropoutLayer(0.4, 'Name', 'dropout2')
    
    fullyConnectedLayer(128, 'Name', 'fc3')
    batchNormalizationLayer('Name', 'bn3')
    leakyReluLayer(0.1, 'Name', 'leaky_relu3')
    dropoutLayer(0.3, 'Name', 'dropout3')
    
    fullyConnectedLayer(128, 'Name', 'fc4')
    batchNormalizationLayer('Name', 'bn4')
    leakyReluLayer(0.1, 'Name', 'leaky_relu4')
    dropoutLayer(0.3, 'Name', 'dropout4')
    
    fullyConnectedLayer(64, 'Name', 'fc5')
    batchNormalizationLayer('Name', 'bn5')
    leakyReluLayer(0.1, 'Name', 'leaky_relu5')
    
    fullyConnectedLayer(64, 'Name', 'fc6')
    batchNormalizationLayer('Name', 'bn6')
    leakyReluLayer(0.1, 'Name', 'leaky_relu6')

    fullyConnectedLayer(32, 'Name', 'fc7')
    batchNormalizationLayer('Name', 'bn7')
    leakyReluLayer(0.1, 'Name', 'leaky_relu7')
    
    fullyConnectedLayer(32, 'Name', 'fc8')
    batchNormalizationLayer('Name', 'bn8')
    leakyReluLayer(0.1, 'Name', 'leaky_relu8')

    fullyConnectedLayer(16, 'Name', 'fc9')
    batchNormalizationLayer('Name', 'bn9')
    leakyReluLayer(0.1, 'Name', 'leaky_relu9')
    
    fullyConnectedLayer(16, 'Name', 'fc10')
    batchNormalizationLayer('Name', 'bn10')
    leakyReluLayer(0.1, 'Name', 'leaky_relu10')
    
    % Output layer
    fullyConnectedLayer(num_classes, 'Name', 'output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

% Enhanced training options
options = trainingOptions('adam', ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.7, ...
    'LearnRateDropPeriod', 25, ...
    'L2Regularization', 0.0005, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 15, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

% --- Train the Model ---
fprintf('Training enhanced network...\n');
[net, trainInfo] = trainNetwork(XTrain, YTrain, layers, options);

% --- Evaluate ---
fprintf('Evaluating...\n');
YPred = classify(net, XVal);
accuracy = mean(YPred == YVal);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Detailed analysis
confusion_mat = confusionmat(grp2idx(YVal), grp2idx(YPred));
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
    true_idx = find(ismember(all_messages, test_msg, 'rows'));
    
    for snr_db = [0, 3, 5, 10]
        % Generate enhanced features
        features = generate_enhanced_features(test_msg, snr_db, carrier_waves, samples_per_bit, target_samples);
        features_normalized = (features - X_mean) ./ X_std;
        features_normalized(isnan(features_normalized)) = 0;
        
        % Predict
        prediction = classify(net, features_normalized);
        predicted_idx = grp2idx(prediction);
        predicted_msg = all_messages(predicted_idx, :);
        
        % Get prediction confidence
        prediction_probs = predict(net, features_normalized);
        max_prob = max(prediction_probs);
        
        is_correct = (predicted_idx == true_idx);
        fprintf('Message [%d%d%d%d] at %d dB: Predicted [%d%d%d%d] (confidence: %.3f) - %s\n', ...
                test_msg, snr_db, predicted_msg, max_prob, ...
                ternary(is_correct, 'CORRECT', 'WRONG'));
    end
end

% --- Save Model ---
save('high_accuracy_classifier.mat', 'net', 'X_mean', 'X_std', 'all_messages', 'trainInfo');

fprintf('\nHigh-accuracy training complete! Final accuracy: %.1f%%\n', accuracy * 100);

% --- Enhanced Feature Extraction Function ---
function features = generate_enhanced_features(message, snr_db, carrier_waves, samples_per_bit, target_samples)
    % Generate signal with phase variations
    signal_matrix = zeros(4, samples_per_bit);
    phase_variation = 0.1 * randn(4, 1);
    
    for k = 1:4
        if message(k) == 1
            phase = 2 * pi * carrier_waves(k, :) + phase_variation(k);
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
