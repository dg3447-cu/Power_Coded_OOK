% Optimization Parameters
n = 9; % Length - 1 of signal vector
alpha = 1; % Weight for power term
beta = 0; % Weight for bandwidth term
A = 0.9; % Min average
B = 0.938; % Max average
v_min = 0; % Min voltage
v_max = 1; % Max voltage

% Construct first-difference matrix
e = ones(n,1);
D = spdiags([-e e], [0 1], n-1, n);
DtD = D' * D;

% Initialize v
rng(1);  % for reproducibility
v = 0.5 + 0.1 * randn(n,1);
v = min(max(v, v_min), v_max);

% Gradient descent
max_iter = 5000;
step_size = 0.1;
tolerance = 1e-6;
decay = 0.995;  % decay factor for learning rate

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

% Final evaluations
power = (1/n) * sum(v.^2);
bandwidth = sum((v(2:end) - v(1:end-1)).^2);
objective = alpha * log(power) + beta * log(bandwidth);

fprintf('\nFinal Results:\n');
fprintf('Objective: %f\n', objective);
fprintf('Average voltage: %f (constraint [%f, %f])\n', mean(v), A, B);
fprintf('Signal power: %f\n', power);
fprintf('Bandwidth: %f\n', bandwidth);

% Plot
figure;
plot(v, '-o');
xlabel('Slice');
ylabel('Voltage Level');
title('Optimized Signal (Minimizing Power × Bandwidth)');
grid on;
