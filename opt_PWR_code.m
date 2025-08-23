% Optimization Parameters
n = 9; % Length - 1 of signal vector
alpha = 1; % Weight for power term --> as alpha approaches 0, we get a flat plot at A
beta = 1; % Weight for bandwidth term --> as beta approaches 0, we get a whole variety of codes to test
A = 0.9; % Min average
B = 0.938; % Max average
v_min = 0; % Min voltage
v_max = 1; % Max voltage

% Construct first-difference matrix
e = ones(n,1);
D = spdiags([-e e], [0 1], n-1, n);
DtD = D' * D;

% Initialize v
rng(1); % for reproducibility
v = 0.5 + 0.1 * randn(n,1);
v = min(max(v, v_min), v_max);

% Gradient descent
max_iter = 5000;
step_size = 0.1;
tolerance = 1e-6;
decay = 0.995; % decay factor for learning rate

for iter = 1:max_iter
    % Compute terms
    v_norm_sq = (1/n) * (v' * v); % signal power
    d_norm_sq = (D * v)' * (D * v); % bandwidth

    % Avoid log(0)
    eps_reg = 1e-8;
    v_norm_sq = max(v_norm_sq, eps_reg);
    d_norm_sq = max(d_norm_sq, eps_reg);

    % Objective (optional for logging)
    obj = alpha * log(v_norm_sq) + beta * log(d_norm_sq);

    % Gradient of objective
    grad_v_power = (2*alpha/n) * v / v_norm_sq;
    grad_bw = 2 * beta * (DtD * v) / d_norm_sq;
    grad = grad_v_power + grad_bw;

    % Gradient descent step
    v_new = v - step_size * grad;

    % Clip to voltage range
    v_new = min(max(v_new, v_min), v_max);

    % Project mean constraint (rescale approach)
    avg_v = mean(v_new);
    if avg_v < A
        v_new = v_new * (A / avg_v);
    elseif avg_v > B
        v_new = v_new * (B / avg_v);
    end

    % Clip again in case rescaling exceeded bounds
    v_new = min(max(v_new, v_min), v_max);

    % Convergence check
    if norm(v_new - v) < tolerance
        fprintf('Converged in %d iterations.\n', iter);
        break;
    end

    % Update
    v = v_new;
    step_size = step_size * decay;
end

if iter == max_iter
    fprintf('Reached maximum iterations.\n');
end

% Final metrics
power = (1/n) * sum(v.^2);
bandwidth = sum((v(2:end) - v(1:end-1)).^2);
objective = alpha * log(power) + beta * log(bandwidth);

fprintf('Average voltage: %f (constraint [%f, %f])\n', mean(v), A, B);

figure;
plot(v, '-o');
xlabel('Slice');
ylabel('Voltage Level');
title('Optimized Signal (Minimizing Power × Bandwidth)');
grid on;

% Notes:
% Almost all outputs are symmetric. Only when beta = 0 does the output lose its symmetry.

% Plot power spectral density (PSD) of sequence to determine noise bandwidth
Fc = 402e6;
T = 2e-6;
N = (length(v) + 1);
v1_multiplier = 1 / (1 - exp(-2 / 3.1));

Fs = 1 / T;
t = 0:1/Fs:N;
t = t(1 : end - 1);

% Extend voltage vector into something multiply-able with the OOK carrier
v_unit_extended = ones(1, floor(length(t) / (length(v) + 1)));
v_extended = [];
for i = 1:(length(v) + 1)
    if i == 1
        extension = v_unit_extended * v1_multiplier; 
        v_extended = [v_extended, extension];
    else
        extension = v_unit_extended * v(i - 1);
        v_extended = [v_extended, extension];
    end
end

carrier = sind(2 * pi * Fc .* t);
s = carrier .* v_extended;

% Compute and plot PSD
[pxx, f] = pwelch(s, [], [], [], Fs);
figure;
plot(f/1e6, 10 * log10(pxx));
xlabel('Frequency (MHz)');
ylabel('Power/Frequency (dB/Hz)');
title('PSD of Power-Coded OOK Signal');
grid on;
