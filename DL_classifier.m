% Neural network that generates a weight matrix and bias term to classify
% the envelope DFTs via their message in one OFDM symbol

clear; clc; close all;

% --- Configuration Parameters ---
num_subcarriers = 4;
carrier_freq = 400e6;
BW = 10e6;
cycles_per_bit = 100;
samples_per_cycle = 20;
num_dft_points = 16;
num_classes = 2 ^ num_subcarriers; % 16 possible messages

% SNR range for training
SNR_dB_range = [0, 1, 2, 3, 5, 7, 10, 15];
samples_per_message = 150;

% First, determine the exact feature size by generating one sample
fprintf('Determining feature size...\n');
test_message = [0 0 0 0];
carrier_waves = precompute_carrier_waves(carrier_freq, BW, samples_per_cycle, cycles_per_bit);
features = extract_features(test_message, 10, carrier_waves, cycles_per_bit, samples_per_cycle, carrier_freq);
feature_size = length(features);

total_samples = num_classes * samples_per_message * length(SNR_dB_range);
fprintf('Feature size: %d, Total samples: %d\n', feature_size, total_samples);

% --- Preallocate Arrays ---
X = zeros(total_samples, feature_size);
Y = zeros(total_samples, 1);
all_messages = dec2bin(0:15, 4) - '0';

% --- Generate Dataset ---
fprintf('Generating enhanced dataset...\n');
sample_idx = 1;

for msg_idx = 1:num_classes
    message = all_messages(msg_idx, :);
    
    for snr_idx = 1:length(SNR_dB_range)
        snr_db = SNR_dB_range(snr_idx);
        
        for noise_iter = 1:samples_per_message
            % Extract features
            features = extract_features(message, snr_db, carrier_waves, cycles_per_bit, samples_per_cycle, carrier_freq);
            
            % Store features (ensure correct size)
            if length(features) == feature_size
                X(sample_idx, :) = features;
                Y(sample_idx) = msg_idx;
                sample_idx = sample_idx + 1;
            else
                fprintf('Warning: Feature size mismatch for message %d, SNR %d, iter %d\n', msg_idx, snr_db, noise_iter);
            end
        end
    end
    fprintf('Generated message %d/%d\n', msg_idx, num_classes);
end

% Remove any unused rows
valid_samples = sample_idx - 1;
X = X(1:valid_samples, :);
Y = Y(1:valid_samples);

fprintf('Final dataset size: %d samples\n', valid_samples);

% --- Feature Normalization ---
fprintf('Normalizing features...\n');
X_mean = mean(X);
X_std = std(X);
X_normalized = (X - X_mean) ./ X_std;
X_normalized(isnan(X_normalized)) = 0;

% --- Split dataset ---
rng(42);
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

% --- Create Deep Neural Network ---
layers = [
    featureInputLayer(feature_size, 'Name', 'input')
    
    % First block
    fullyConnectedLayer(256, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.4, 'Name', 'dropout1')
    
    % Second block
    fullyConnectedLayer(128, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.3, 'Name', 'dropout2')
    
    % Third block
    fullyConnectedLayer(64, 'Name', 'fc3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    
    % Fourth block
    fullyConnectedLayer(32, 'Name', 'fc4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    
    % Output layer
    fullyConnectedLayer(num_classes, 'Name', 'output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 30, ...
    'L2Regularization', 0.0001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal_cat}, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

% --- Train the Model ---
fprintf('Training deep neural network...\n');
[net, trainInfo] = trainNetwork(XTrain, YTrain_cat, layers, options);

% --- Evaluate ---
fprintf('Evaluating...\n');
YPred = classify(net, XVal);
accuracy = mean(YPred == YVal_cat);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% --- Test Specific Examples ---
fprintf('\nTesting specific examples:\n');
test_messages = [0 1 1 0; 1 0 1 0; 1 1 1 1; 0 0 0 0];

for i = 1:size(test_messages, 1)
    test_msg = test_messages(i, :);
    msg_idx = find(ismember(all_messages, test_msg, 'rows'));
    
    for snr_db = [0, 3, 5, 10]
        features = extract_features(test_msg, snr_db, carrier_waves, cycles_per_bit, samples_per_cycle, carrier_freq);
        features_normalized = (features - X_mean) ./ X_std;
        features_normalized(isnan(features_normalized)) = 0;
        
        prediction = classify(net, features_normalized);
        predicted_idx = double(prediction);
        predicted_msg = all_messages(predicted_idx, :);
        
        is_correct = all(predicted_msg == test_msg);
        fprintf('Message [%d%d%d%d] at %d dB: Predicted [%d%d%d%d] - %s\n', ...
                test_msg, snr_db, predicted_msg, ternary(is_correct, 'CORRECT', 'WRONG'));
    end
end

% --- Save Model ---
save('mc_ook_classifier_optimized.mat', 'net', 'X_mean', 'X_std', 'all_messages', 'trainInfo');

fprintf('\nTraining complete! Model saved.\n');

% --- Helper Functions ---
function carrier_waves = precompute_carrier_waves(carrier_freq, BW, samples_per_cycle, cycles_per_bit)
    Fs = carrier_freq * samples_per_cycle;
    samples_per_bit = round(cycles_per_bit * (Fs / carrier_freq));
    t_bit = (0:samples_per_bit-1) * (1/Fs);
    
    num_subcarriers = 4;
    subcarriers = linspace(carrier_freq - BW/2, carrier_freq + BW/2, num_subcarriers);
    
    carrier_waves = zeros(num_subcarriers, samples_per_bit);
    for k = 1:num_subcarriers
        carrier_waves(k, :) = sin(2 * pi * subcarriers(k) * t_bit);
    end
end

function features = extract_features(message, snr_db, carrier_waves, cycles_per_bit, samples_per_cycle, carrier_freq)
    % Calculate samples per bit
    Fs = carrier_freq * samples_per_cycle;
    samples_per_bit = round(cycles_per_bit * (Fs / carrier_freq));
    
    % Generate signal
    signal_matrix = zeros(4, samples_per_bit);
    for k = 1:4
        if message(k) == 1
            signal_matrix(k, :) = carrier_waves(k, :);
        end
    end
    
    sig_clean = sum(signal_matrix, 1);
    sig_noisy = awgn(sig_clean, snr_db, 'measured');
    
    % Envelope detection
    analytic_signal = hilbert(sig_noisy);
    amplitude_envelope = abs(analytic_signal);
    
    % --- Extract consistent features ---
    features = [];
    
    % 1. Fixed number of envelope samples (20 points)
    env_samples = 20;
    if length(amplitude_envelope) >= env_samples
        features = [features, amplitude_envelope(1:env_samples)];
    else
        % Pad if shorter
        padded = [amplitude_envelope, zeros(1, env_samples - length(amplitude_envelope))];
        features = [features, padded];
    end
    
    % 2. Fixed DFT points
    dft_points = 16;
    dft_magnitude = abs(fft(amplitude_envelope, dft_points));
    features = [features, dft_magnitude];
    
    % 3. Spectral features (fixed number)
    power_spectrum = dft_magnitude.^2;
    frequencies = 0:(dft_points-1);
    
    spectral_centroid = sum(frequencies .* power_spectrum) / (sum(power_spectrum) + 1e-6);
    spectral_spread = sqrt(sum(((frequencies - spectral_centroid).^2) .* power_spectrum) / (sum(power_spectrum) + 1e-6));
    
    geometric_mean = exp(mean(log(dft_magnitude + 1e-6)));
    arithmetic_mean = mean(dft_magnitude);
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-6);
    
    features = [features, spectral_centroid, spectral_spread, spectral_flatness];
    
    % 4. Statistical features (fixed number)
    stats = [mean(amplitude_envelope), std(amplitude_envelope), ...
             skewness(amplitude_envelope), kurtosis(amplitude_envelope)];
    features = [features, stats];
    
    % 5. Ensure exact size by padding/truncating if needed
    target_size = 20 + 16 + 3 + 4; % envelope + dft + spectral + stats
    if length(features) > target_size
        features = features(1:target_size);
    elseif length(features) < target_size
        features = [features, zeros(1, target_size - length(features))];
    end
end

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
