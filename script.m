%% 2026 한국 BCI 학회 워크샵 (2026-09-30)
%% Decoding Tutorial: Tuning, PVA, Kalman Filter, OLE (Salinas & Abbott, 1994)
% 데이터 차원
%   X         : [2 x T x N]  cursor position (x, y)
%   Z         : [U x T x N]  binned data (spike count or rate)
%   TargetPos : [N x 2]      target position
%   HomePos   : [N x 2] or  [1 x 2]  home position
%
% Examples
%   0. Preprocessing  : Calculating kinematic parameters, elimiation of silent neurons, epoching, train/test separation
%   1. Example 1  : Cosine tuning analysis (train trial, 움직임 구간 평균 발화)
%   2. Example 2  : PVA (baseline correction + modulation depth normalization, effective neuron only)
%   3. Example 3  : Kalman Filter vs. PVA
%   4. Example 4  : Salinas & Abbott OLE vs. Kalman Filter
%  
% Author
%   Min-Ki Kim
%   mkworks@bu.ac.kr
%

clear; clc; close all;
load sample_data.mat

%% =========================================================================
%% Preset parameters & preprocessing
%% =========================================================================
dt                = 0.05;   % bin size (50 ms)
max_silent_trials = 10;     % 발화가 0인 trial이 이 수보다 많으면 뉴런 제거
alpha             = 0.05;   % significance for neuronal selection (PVA 뉴런 선택)
train_frac        = 0.75;   % validation set ratio
X_is_relative     = false;  % if X equals to the home position, it should be true.

set(0, 'DefaultAxesFontSize', 10, 'DefaultLineLineWidth', 1.5);

% velocity
V   = diff(X, 1, 2) / dt;           % [2 x T-1 x N]
Z_v = Z(:, 1:end-1, :);             % [U x T-1 x N]

% eliminate silent neurons 
spk_per_trial = reshape(sum(Z_v, 2), size(Z_v, 1), []);   % [U x N]
rm = sum(spk_per_trial == 0, 2) > max_silent_trials;
Z_v(rm, :, :) = [];
fprintf('제거된 뉴런: %d개 / 사용 뉴런: %d개\n', nnz(rm), nnz(~rm));

[n_units, n_bins, n_trials] = size(Z_v);

% home/target position
if size(HomePos, 1) == 1
    HomeMat = repmat(HomePos, n_trials, 1);
else
    HomeMat = HomePos;
end
T_target = TargetPos - HomeMat;     % [N x 2], Home 기준 타깃 위치

% speed, true direction, movement mask
speed     = reshape(sqrt(sum(V.^2, 1)), n_bins, n_trials);            % [bins x N]
true_dir  = reshape(atan2(V(2, :, :), V(1, :, :)), n_bins, n_trials);  % [bins x N]
move_mask = speed > 0;       

% train and test set separation 
n_train   = round(train_frac * n_trials);
perm      = randperm(n_trials);
train_idx = sort(perm(1:n_train));
test_idx  = sort(perm(n_train+1:end));
n_test    = numel(test_idx);

% angle difference for measuring directional decoding performance (wrap-around)
ang_err = @(a, b) abs(atan2(sin(a - b), cos(a - b)));


%% =========================================================================
%% Example 1: Cosine tuning curve (train trial only)
%% =========================================================================
fprintf('\n--- [Example 1] Cosine tuning curve ---\n');

trial_dirs = atan2(T_target(:, 2), T_target(:, 1));   % [N x 1]

% mean firing rates of movement period
mean_firing = nan(n_trials, n_units);
for tr = 1:n_trials
    if any(move_mask(:, tr))
        mean_firing(tr, :) = mean(Z_v(:, move_mask(:, tr), tr), 2)';
    end
end

% f_u(theta) = b0 + b1*cos(theta) + b2*sin(theta) = b0 + m*cos(theta - PD)
b0 = zeros(n_units, 1);  b1 = zeros(n_units, 1);  b2 = zeros(n_units, 1);
pref_dirs  = zeros(n_units, 1);
mod_depths = zeros(n_units, 1);
p_values   = zeros(n_units, 1);
r_squared  = zeros(n_units, 1);

Xd = [cos(trial_dirs(train_idx)), sin(trial_dirs(train_idx))];
for u = 1:n_units
    lm = fitlm(Xd, mean_firing(train_idx, u));    % NaN trial은 자동 제외
    b  = lm.Coefficients.Estimate;
    b0(u) = b(1);  b1(u) = b(2);  b2(u) = b(3);
    pref_dirs(u)  = atan2(b(3), b(2));
    mod_depths(u) = hypot(b(2), b(3));
    p_values(u)   = lm.ModelFitVsNullModel.Pvalue;
    r_squared(u)  = lm.Rsquared.Ordinary;
end
sig_units = p_values < alpha;
fprintf('유의하게 튜닝된 뉴런: %d / %d (p < %.2f)\n', nnz(sig_units), n_units, alpha);

% [for visualization] R^2 상위 2개 뉴런
[~, order]  = sort(r_squared, 'descend');
show_units  = order(1:min(2, n_units));
th_dense    = linspace(-pi, pi, 200)';
th_polar    = linspace(0, 2*pi, 200);

figure('Name', 'Example 1: Cosine Tuning Curves & Polar Plots', 'Position', [50, 50, 1100, 700]);
for k = 1:numel(show_units)
    u = show_units(k);

    % Cartesian fit
    subplot(2, 3, (k-1)*3 + 1);
    scatter(trial_dirs(train_idx), mean_firing(train_idx, u), 20, [0.3 0.3 0.8], ...
            'filled', 'MarkerFaceAlpha', 0.5); hold on;
    plot(th_dense, b0(u) + b1(u)*cos(th_dense) + b2(u)*sin(th_dense), 'r-', 'LineWidth', 2);
    xline(pref_dirs(u), 'k--', sprintf('PD: %.1f°', rad2deg(pref_dirs(u))));
    title(sprintf('Unit %d (R^2=%.2f, p=%.2e)', u, r_squared(u), p_values(u)));
    xlabel('Target direction (rad)'); ylabel('Mean firing (per bin)');
    xlim([-pi, pi]); grid on;

    % Polar tuning curve (음수 반지름은 반대 방향으로 그려지므로 0에서 자름)
    subplot(2, 3, (k-1)*3 + 2);
    r_polar = max(0, b0(u) + mod_depths(u) * cos(th_polar - pref_dirs(u)));
    polarplot(th_polar, r_polar, 'r-', 'LineWidth', 2); hold on;
    polarplot([pref_dirs(u), pref_dirs(u)], [0, max(r_polar)], 'k--', 'LineWidth', 1.5);
    title(sprintf('Unit %d Polar Tuning', u));
end

% population PD distribution (significant neurons)
subplot(2, 3, [3, 6]);
if any(sig_units)
    polarhistogram(pref_dirs(sig_units), 12, 'FaceColor', [0.2 0.7 0.4], 'FaceAlpha', 0.7);
    title(sprintf('PD Distribution (significant units, n=%d)', nnz(sig_units)));
else
    polarhistogram(pref_dirs, 12, 'FaceColor', [0.6 0.6 0.6], 'FaceAlpha', 0.7);
    title('PD Distribution (all units; none significant)');
end


%% =========================================================================
%% Example 2: Population Vector Analysis (PVA)
%% =========================================================================
% pop_vec(t) = sum_u [(r_u(t) - b0_u) / m_u] * PD_u
%   - baseline 아래 발화(음수 가중치)도 반대 방향 정보로 사용
%   - modulation depth로 정규화하여 발화율이 큰 뉴런이 지배하지 않도록 함
fprintf('\n--- [Example 2] PVA decoding ---\n');

use_pva = sig_units;
if nnz(use_pva) < 2
    warning('유의 뉴런이 2개 미만이므로 PVA에 전체 뉴런을 사용합니다.');
    use_pva = true(n_units, 1);
end
n_pva = nnz(use_pva);

P_vec = [cos(pref_dirs(use_pva)), sin(pref_dirs(use_pva))];                % [Up x 2]
Zn    = (Z_v(use_pva, :, :) - b0(use_pva)) ./ mod_depths(use_pva);         % [Up x bins x N]
pva_vec = reshape(P_vec' * reshape(Zn, n_pva, []), 2, n_bins, n_trials);   % [2 x bins x N]
pva_dir = reshape(atan2(pva_vec(2, :, :), pva_vec(1, :, :)), n_bins, n_trials);

% [visualization]
sample_tr = test_idx(1);
[~, sample_t] = max(speed(:, sample_tr));     % 속도가 가장 큰 시점

figure('Name', 'Example 2: Population Vector Analysis (PVA)', 'Position', [100, 100, 1100, 450]);

% 단일 시점의 개별 뉴런 기여 벡터와 합성 벡터
subplot(1, 2, 1); hold on;
w_sample = Zn(:, sample_t, sample_tr);
contrib  = w_sample .* P_vec;                  % [Up x 2]
for u = 1:n_pva
    h_units = quiver(0, 0, contrib(u, 1), contrib(u, 2), 0, ...
                     'Color', [0.7 0.7 0.7], 'LineWidth', 1, 'MaxHeadSize', 0.5);
end
pv_now = pva_vec(:, sample_t, sample_tr);
h_pv   = quiver(0, 0, pv_now(1), pv_now(2), 0, 'r', 'LineWidth', 2.5, 'MaxHeadSize', 0.8);
v_act  = V(:, sample_t, sample_tr);
v_disp = v_act / norm(v_act) * norm(pv_now);   % 방향 비교용으로 길이를 PV에 맞춤
h_act  = quiver(0, 0, v_disp(1), v_disp(2), 0, 'k--', 'LineWidth', 2, 'MaxHeadSize', 0.8);
grid on; axis equal;
title(sprintf('Single Timebin (Trial %d, t=%d): Vector Sum', sample_tr, sample_t));
legend([h_units, h_pv, h_act], {'Individual Units', 'Population Vector (PVA)', ...
        'Actual Direction (scaled)'}, 'Location', 'best');
xlabel('X component'); ylabel('Y component');

% 궤적 위의 PVA 방향 (움직임 구간만)
subplot(1, 2, 2);
pos = X(:, 1:end-1, sample_tr);
if ~X_is_relative
    pos = pos - HomeMat(sample_tr, :)';        % 타깃과 같은 Home 기준 좌표로 변환
end
plot(pos(1, :), pos(2, :), 'k-', 'LineWidth', 2); hold on;
qi = find(move_mask(:, sample_tr))';
qi = qi(1:2:end);
quiver(pos(1, qi), pos(2, qi), cos(pva_dir(qi, sample_tr))', sin(pva_dir(qi, sample_tr))', ...
       0.5, 'r', 'LineWidth', 1.2);
scatter(T_target(sample_tr, 1), T_target(sample_tr, 2), 100, 'p', 'filled', 'MarkerFaceColor', 'm');
title(sprintf('PVA Decoded Direction on Trajectory (Trial %d)', sample_tr));
xlabel('X Position'); ylabel('Y Position');
legend({'Actual Trajectory', 'PVA Heading', 'Target'}, 'Location', 'best');
grid on; axis equal;


%% =========================================================================
%% Example 3: Kalman Filter decoding, comparison with PVA 
%% =========================================================================
fprintf('\n--- [Example 3] Kalman Filter vs. PVA ---\n');

Z_train = reshape(Z_v(:, :, train_idx), n_units, [])';   % [samples x U]
V_train = reshape(V(:, :, train_idx), 2, [])';           % [samples x 2]
mu_Z    = mean(Z_train, 1);                              % [1 x U]
Zc_train = Z_train - mu_Z;


S1 = reshape(V(:, 1:end-1, train_idx), 2, [])';
S2 = reshape(V(:, 2:end,   train_idx), 2, [])';
A  = (S2' * S1) / (S1' * S1);
W  = (S2 - S1*A')' * (S2 - S1*A') / size(S2, 1);
H  = (Zc_train' * V_train) / (V_train' * V_train);                          % [U x 2]
Q  = (Zc_train - V_train*H')' * (Zc_train - V_train*H') / size(V_train, 1); % [U x U]

kf_vel = zeros(2, n_bins, n_test);
for i = 1:n_test
    tr = test_idx(i);
    x  = zeros(2, 1);
    P  = W;
    for t = 1:n_bins
        % Prediction
        x_pred = A * x;
        P_pred = A * P * A' + W;
        % Update
        z = Z_v(:, t, tr) - mu_Z';
        K = P_pred * H' / (H * P_pred * H' + Q);
        x = x_pred + K * (z - H * x_pred);
        P = (eye(2) - K * H) * P_pred;
        kf_vel(:, t, i) = x;
    end
end
kf_dir = reshape(atan2(kf_vel(2, :, :), kf_vel(1, :, :)), n_bins, n_test);

% angle difference
mask_test     = move_mask(:, test_idx);
true_test_dir = true_dir(:, test_idx);
pva_test_dir  = pva_dir(:, test_idx);
err_pva = ang_err(pva_test_dir, true_test_dir);
err_kf  = ang_err(kf_dir,       true_test_dir);

% visualization
figure('Name', 'Example 3: Kalman Filter vs. PVA Performance', 'Position', [120, 120, 1200, 420]);

subplot(1, 2, 1);
boxplot([rad2deg(err_pva(mask_test)), rad2deg(err_kf(mask_test))], 'Labels', {'PVA', 'Kalman Filter'});
ylabel('Absolute Angular Error (deg)');
title(sprintf('Moving bins only | Mean Err: PVA=%.1f°, KF=%.1f°', ...
      rad2deg(mean(err_pva(mask_test))), rad2deg(mean(err_kf(mask_test)))));
grid on;

subplot(1, 2, 2);
m1 = double(mask_test(:, 1)); m1(m1 == 0) = NaN;   % 정지 구간은 표시하지 않음
time_axis = (1:n_bins) * dt;
plot(time_axis, rad2deg(true_test_dir(:, 1)) .* m1, 'k-', 'LineWidth', 2); hold on;
plot(time_axis, rad2deg(pva_test_dir(:, 1))  .* m1, 'b--', 'LineWidth', 1.2);
plot(time_axis, rad2deg(kf_dir(:, 1))        .* m1, 'r-', 'LineWidth', 1.5);
legend({'Actual', 'PVA', 'Kalman'}, 'Location', 'best');
xlabel('Time (s)'); ylabel('Direction (deg)');
title(sprintf('Heading Direction over Time (Trial %d)', test_idx(1))); grid on;


%% =========================================================================
%% Example 4: Salinas & Abbott (1994) Optimal Linear Estimator vs. Kalman Filter
%% =========================================================================
fprintf('\n--- [Example 4] Salinas & Abbott OLE vs. Kalman Filter ---\n');

% noise variance: train 움직임 구간 bin에서 튜닝 예측 대비 잔차의 분산
pred_trial = b0 + b1 .* cos(trial_dirs') + b2 .* sin(trial_dirs');        % [U x N]
R_train    = Z_v(:, :, train_idx) - reshape(pred_trial(:, train_idx), n_units, 1, []);
M_train    = move_mask(:, train_idx);
R_flat     = reshape(R_train, n_units, []);
noise_var  = var(R_flat(:, M_train(:)), 0, 2);                             % [U x 1]

% find L, Q directional grid (uniform prior distribution) 
n_grid = 360;
th_g   = (0:n_grid-1)' * (2*pi / n_grid);                                  % [G x 1]
F      = b1 .* cos(th_g') + b2 .* sin(th_g');                              % [U x G]
E      = [cos(th_g'); sin(th_g')];                                         % [2 x G]
L_ole  = F * E' / n_grid;                                                  % [U x 2]
Q_ole  = F * F' / n_grid + diag(noise_var);                                % [U x U]
D_ole  = Q_ole \ L_ole;                                                    % [U x 2]

% directional decoding 
ole_vec = reshape(D_ole' * reshape(Z_v - b0, n_units, []), 2, n_bins, n_trials);
ole_dir = reshape(atan2(ole_vec(2, :, :), ole_vec(1, :, :)), n_bins, n_trials);

% for speed comparision  (tip. g는 방향을 바꾸지 않으므로 각도 성능에는 영향 없음)
O_tr  = reshape(ole_vec(:, :, train_idx), 2, []);
V_tr  = reshape(V(:, :, train_idx), 2, []);
g_ole = sum(O_tr(:) .* V_tr(:)) / sum(O_tr(:).^2);

% test data
S_test_actual = reshape(V(:, :, test_idx), 2, []);        % [2 x samples]
V_ole = g_ole * reshape(ole_vec(:, :, test_idx), 2, []);
V_kf  = reshape(kf_vel, 2, []);
err_ole = ang_err(ole_dir(:, test_idx), true_test_dir);

% performance
rmse_ole = sqrt(mean((V_ole - S_test_actual).^2, 2));
rmse_kf  = sqrt(mean((V_kf  - S_test_actual).^2, 2));
r_ole    = diag(corr(V_ole', S_test_actual'));
r_kf     = diag(corr(V_kf',  S_test_actual'));

fprintf('\n%-8s | %8s %8s | %6s %6s | %s\n', 'Decoder', 'RMSE Vx', 'RMSE Vy', 'r Vx', 'r Vy', 'Ang.Err (deg, moving)');
fprintf('%-8s | %8s %8s | %6s %6s | %.1f\n', 'PVA', '-', '-', '-', '-', rad2deg(mean(err_pva(mask_test))));
fprintf('%-8s | %8.2f %8.2f | %6.2f %6.2f | %.1f\n', 'KF',  rmse_kf,  r_kf,  rad2deg(mean(err_kf(mask_test))));
fprintf('%-8s | %8.2f %8.2f | %6.2f %6.2f | %.1f\n', 'OLE', rmse_ole, r_ole, rad2deg(mean(err_ole(mask_test))));

% visualizaiton
figure('Name', 'Example 4: Salinas & Abbott OLE vs. Kalman Filter', 'Position', [150, 150, 1300, 650]);
sample_bins = 1:min(180, size(S_test_actual, 2));

% Vx tracking
subplot(2, 3, [1 2]);
plot(sample_bins*dt, S_test_actual(1, sample_bins), 'k-', 'LineWidth', 1.8); hold on;
plot(sample_bins*dt, V_ole(1, sample_bins), 'Color', [0 0.45 0.74], 'LineStyle', '--');
plot(sample_bins*dt, V_kf(1, sample_bins),  'Color', [0.85 0.33 0.1], 'LineStyle', '-');
ylabel('V_x'); title(sprintf('X-Velocity (r: OLE=%.2f, KF=%.2f)', r_ole(1), r_kf(1)));
legend({'Actual', 'OLE (scaled)', 'Kalman'}, 'Location', 'northeast'); grid on;

% Vy tracking
subplot(2, 3, [4 5]);
plot(sample_bins*dt, S_test_actual(2, sample_bins), 'k-', 'LineWidth', 1.8); hold on;
plot(sample_bins*dt, V_ole(2, sample_bins), 'Color', [0 0.45 0.74], 'LineStyle', '--');
plot(sample_bins*dt, V_kf(2, sample_bins),  'Color', [0.85 0.33 0.1], 'LineStyle', '-');
xlabel('Time (s, concatenated test trials)'); ylabel('V_y');
title(sprintf('Y-Velocity (r: OLE=%.2f, KF=%.2f)', r_ole(2), r_kf(2)));
legend({'Actual', 'OLE (scaled)', 'Kalman'}, 'Location', 'northeast'); grid on;

% angle difference between three decoders
subplot(2, 3, 3);
boxplot([rad2deg(err_pva(mask_test)), rad2deg(err_kf(mask_test)), rad2deg(err_ole(mask_test))], ...
        'Labels', {'PVA', 'KF', 'OLE'});
ylabel('Absolute Angular Error (deg)'); title('Angular Error (moving bins)'); grid on;

% OLE weights vs. PD: 대각선에서 벗어난 정도 ~ Q^{-1}에 의한 보정량
subplot(2, 3, 6);
ole_w_dir = atan2(D_ole(:, 2), D_ole(:, 1));
scatter(rad2deg(pref_dirs), rad2deg(ole_w_dir), 30, mod_depths, 'filled'); hold on;
plot([-180 180], [-180 180], 'k--', 'LineWidth', 1);
cb = colorbar; cb.Label.String = 'Modulation depth';
xlabel('Preferred direction (deg)'); ylabel('OLE weight direction (deg)');
title('OLE Weights vs. PD'); xlim([-180 180]); ylim([-180 180]); axis square; grid on;