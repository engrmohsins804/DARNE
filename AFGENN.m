%% =========================================================================
%  AFGENN — Adaptive Fast Geometric Extraction for Nearest Neighbors
%  MATLAB Implementation (Reference + Comparison)
%
%  Based on: "Adaptive Fast Geometric Extraction for Nearest Neighbors"
%             PhD Proposal, Version 2
%
%  Sections:
%   1. AFGENN Preprocessing  — dense ranks, global density
%   2. AFGENN Query          — adaptive hyper-window, density-ratio update
%   3. Classification rule   — majority vote on returned neighbor set
%   4. Baselines             — KNN, Weighted KNN, Fixed-Diff FGENN
%   5. Synthetic dataset     — density-gradient 2-D experiment
%   6. Results comparison    — accuracy, neighbor count, runtime tables
%
%  Tested with MATLAB R2022b+.  No external toolboxes required.
%  Random seed is fixed (rng(42)) for reproducibility.
% =========================================================================

clear; clc; close all;
rng(42);

%% =========================================================================
%  SECTION 1 — AFGENN Preprocessing
% =========================================================================

function [R, U, alpha, rho_global] = afgenn_preprocess(X)
% AFGENN_PREPROCESS  Build rank matrix and global density.
%
%   [R, U, alpha, rho_global] = afgenn_preprocess(X)
%
%   Input:
%     X        N x M  training feature matrix (doubles)
%
%   Outputs:
%     R          N x M  integer dense-rank matrix
%     U          1 x M  cell array of sorted unique value vectors
%     alpha      1 x M  number of unique values per feature
%     rho_global scalar  global rank-space density  (N / prod(alpha))

    [N, M] = size(X);
    R      = zeros(N, M, 'int32');
    U      = cell(1, M);
    alpha  = zeros(1, M);

    for m = 1:M
        u_m       = unique(X(:,m));   % sorted unique values
        U{m}      = u_m;
        alpha(m)  = numel(u_m);
        % Dense rank: rank(i,m) = 1 + number of unique values strictly less than x(i,m)
        for i = 1:N
            R(i,m) = int32(1 + sum(u_m < X(i,m)));
        end
    end

    V_global   = prod(double(alpha));
    rho_global = N / V_global;
end

%% =========================================================================
%  SECTION 2 — AFGENN Query-Time Extraction
% =========================================================================

function [neighbors, diff_final, lambda_trace] = afgenn_query( ...
        xq, R, U, alpha, rho_global, ...
        beta, epsilon, T_max, diff_min, diff_max)
% AFGENN_QUERY  Adaptive rank-space neighbor extraction for one query.
%
%   Inputs:
%     xq         1 x M  query feature vector
%     R          N x M  integer rank matrix from preprocessing
%     U          1 x M  cell: sorted unique value lists
%     alpha      1 x M  unique-count per feature
%     rho_global scalar  global density
%     beta       smoothing coefficient in [0,1] (default 0.7)
%     epsilon    convergence tolerance (default 0.1)
%     T_max      maximum iterations (default 20)
%     diff_min   minimum Diff (default 1)
%     diff_max   maximum Diff (default floor(min(alpha)/2))
%
%   Outputs:
%     neighbors    indices (into training set) of extracted neighbors
%     diff_final   final Diff value selected
%     lambda_trace lambda values recorded per iteration (diagnostic)

    if nargin < 6,  beta      = 0.7;              end
    if nargin < 7,  epsilon   = 0.1;              end
    if nargin < 8,  T_max     = 20;               end
    if nargin < 9,  diff_min  = 1;                end
    if nargin < 10, diff_max  = floor(min(alpha)/2); end
    diff_max = max(diff_min, diff_max);

    [N, M] = size(R);

    % --- Rank insertion for query xq (clipped to valid range) ---
    rq = zeros(1, M, 'int32');
    for m = 1:M
        rq(m) = int32(1 + sum(U{m} < xq(m)));
        rq(m) = int32(max(1, min(alpha(m), rq(m))));
    end

    diff      = diff_min;
    lambda_trace = zeros(T_max, 1);
    best_diff = diff;
    best_lam_dist = Inf;

    for t = 1:T_max
        % --- Extract hyper-window W_q(diff) ---
        in_window = true(N, 1);
        for m = 1:M
            lo = max(1,        int32(rq(m)) - int32(diff));
            hi = min(alpha(m), int32(rq(m)) + int32(diff));
            in_window = in_window & (R(:,m) >= lo) & (R(:,m) <= hi);
        end
        % Exclude self (if query is a training point — checked by rank match)
        rank_match = all(bsxfun(@eq, R, rq), 2);
        in_window  = in_window & ~rank_match;

        W_size = sum(in_window);

        % --- Boundary-corrected volume V_q(diff) ---
        V_q = 1;
        for m = 1:M
            L_qm = max(1,        double(rq(m)) - diff);
            U_qm = min(alpha(m), double(rq(m)) + diff);
            V_q  = V_q * (U_qm - L_qm + 1);
        end

        % --- Local density and ratio ---
        if V_q > 0
            rho_q  = W_size / V_q;
        else
            rho_q  = 0;
        end
        lambda = rho_q / max(rho_global, 1e-12);
        lambda_trace(t) = lambda;

        % Track best Diff by |log(lambda)| criterion (bounded-search fallback)
        if W_size > 0 && abs(log(max(lambda, 1e-9))) < best_lam_dist
            best_lam_dist = abs(log(max(lambda, 1e-9)));
            best_diff     = diff;
        end

        % --- Convergence check ---
        if abs(lambda - 1) <= epsilon && W_size > 0
            break;
        end

        % --- Adaptive update ---
        tau = 1e-6;  % division guard
        diff_new = beta * diff + (1 - beta) * diff / max(lambda, tau);
        diff_new = round(diff_new);
        diff_new = max(diff_min, min(diff_max, diff_new));

        % Stagnation: apply bounded-search fallback
        if diff_new == diff
            diff = best_diff;
            break;
        end
        diff = diff_new;
    end

    % Final neighbor extraction at converged/best diff
    in_window = true(N, 1);
    for m = 1:M
        lo = max(1,        int32(rq(m)) - int32(diff));
        hi = min(alpha(m), int32(rq(m)) + int32(diff));
        in_window = in_window & (R(:,m) >= lo) & (R(:,m) <= hi);
    end
    in_window   = in_window & ~rank_match;
    neighbors   = find(in_window);
    diff_final  = diff;
    lambda_trace = lambda_trace(1:t);
end

%% =========================================================================
%  SECTION 3 — Classification Decision Rule
% =========================================================================

function y_pred = afgenn_classify(Xtest, Xtrain, Ytrain, R, U, alpha, rho_global, varargin)
% AFGENN_CLASSIFY  Classify all test points using AFGENN neighbor extraction.
%
%   y_pred = afgenn_classify(Xtest, Xtrain, Ytrain, R, U, alpha, rho_global)
%
%   Classification rule (Eq. 19-20):
%     P(c | xq) = |{i in N(q) : yi = c}| / |N(q)|
%     y_hat     = argmax_c  P(c | xq)
%   Fallback: 1-NN by rank-L∞ distance when N(q) is empty.

    n_test  = size(Xtest, 1);
    y_pred  = zeros(n_test, 1);
    classes = unique(Ytrain);

    for i = 1:n_test
        neighbors = afgenn_query(Xtest(i,:), R, U, alpha, rho_global, varargin{:});

        if isempty(neighbors)
            % Fallback: nearest point by L-inf rank distance
            [~, idx] = min(max(abs(bsxfun(@minus, double(R), double(afgenn_rank_q(Xtest(i,:), U, alpha)))), [], 2));
            y_pred(i) = Ytrain(idx);
        else
            % Majority vote
            votes = arrayfun(@(c) sum(Ytrain(neighbors) == c), classes);
            [~, best] = max(votes);
            y_pred(i) = classes(best);
        end
    end
end

function rq = afgenn_rank_q(xq, U, alpha)
% Helper: compute rank coordinates of a query point.
    M  = numel(U);
    rq = zeros(1, M, 'int32');
    for m = 1:M
        rq(m) = int32(max(1, min(alpha(m), 1 + sum(U{m} < xq(m)))));
    end
end

%% =========================================================================
%  SECTION 4 — Baselines
% =========================================================================

function y_pred = knn_classify(Xtest, Xtrain, Ytrain, k)
% Standard KNN (Euclidean distance, majority vote).
    n_test  = size(Xtest, 1);
    y_pred  = zeros(n_test, 1);
    classes = unique(Ytrain);
    for i = 1:n_test
        dists      = sum(bsxfun(@minus, Xtrain, Xtest(i,:)).^2, 2);
        [~, idx]   = sort(dists, 'ascend');
        nn_labels  = Ytrain(idx(1:k));
        votes      = arrayfun(@(c) sum(nn_labels == c), classes);
        [~, best]  = max(votes);
        y_pred(i)  = classes(best);
    end
end

function y_pred = weighted_knn_classify(Xtest, Xtrain, Ytrain, k)
% Weighted KNN: weight = 1/dist^2 (Hechenbichler & Schliep 2004).
    n_test  = size(Xtest, 1);
    y_pred  = zeros(n_test, 1);
    classes = unique(Ytrain);
    for i = 1:n_test
        dists      = sum(bsxfun(@minus, Xtrain, Xtest(i,:)).^2, 2) + 1e-10;
        [~, idx]   = sort(dists, 'ascend');
        idx_k      = idx(1:k);
        w          = 1 ./ dists(idx_k).^2;
        votes      = arrayfun(@(c) sum(w(Ytrain(idx_k)==c)), classes);
        [~, best]  = max(votes);
        y_pred(i)  = classes(best);
    end
end

function y_pred = fgenn_fixed_classify(Xtest, Xtrain, Ytrain, R, U, alpha, diff_fixed)
% Fixed-Diff FGENN (no density adaptation) — baseline per FGENN [13].
    n_test  = size(Xtest, 1);
    y_pred  = zeros(n_test, 1);
    N       = size(Xtrain, 1);
    classes = unique(Ytrain);

    for i = 1:n_test
        rq = afgenn_rank_q(Xtest(i,:), U, alpha);
        in_window = true(N, 1);
        for m = 1:numel(U)
            lo = max(1,        int32(rq(m)) - int32(diff_fixed));
            hi = min(alpha(m), int32(rq(m)) + int32(diff_fixed));
            in_window = in_window & (R(:,m) >= lo) & (R(:,m) <= hi);
        end
        rank_match = all(bsxfun(@eq, R, rq), 2);
        in_window  = in_window & ~rank_match;
        neighbors  = find(in_window);

        if isempty(neighbors)
            dists     = sum(bsxfun(@minus, Xtrain, Xtest(i,:)).^2, 2);
            [~,idx]   = min(dists);
            y_pred(i) = Ytrain(idx);
        else
            votes      = arrayfun(@(c) sum(Ytrain(neighbors)==c), classes);
            [~, best]  = max(votes);
            y_pred(i)  = classes(best);
        end
    end
end

%% =========================================================================
%  SECTION 5 — Synthetic Dataset Generation
%             Two Gaussians: dense cluster + sparse cluster
% =========================================================================

fprintf('=== AFGENN Synthetic Density-Gradient Experiment ===\n\n');

N_dense  = 200;   % Dense cluster (class 1)
N_sparse =  80;   % Sparse cluster (class 2)
N_total  = N_dense + N_sparse;

% Dense cluster: tight, near origin
X_dense  = 0.4 * randn(N_dense,  2) + [1.5, 1.5];
% Sparse cluster: wide, offset
X_sparse = 1.8 * randn(N_sparse, 2) + [5.0, 1.5];

X_all = [X_dense;  X_sparse];
Y_all = [ones(N_dense, 1); 2*ones(N_sparse, 1)];

% Shuffle
perm  = randperm(N_total);
X_all = X_all(perm, :);
Y_all = Y_all(perm);

% 70 / 30 train-test split
n_train = round(0.7 * N_total);
Xtrain  = X_all(1:n_train,   :);
Ytrain  = Y_all(1:n_train);
Xtest   = X_all(n_train+1:end, :);
Ytest   = Y_all(n_train+1:end);

fprintf('Dataset: %d train, %d test samples; 2 features, 2 classes\n', n_train, numel(Ytest));
fprintf('Class distribution (train): class 1 = %d, class 2 = %d\n\n', sum(Ytrain==1), sum(Ytrain==2));

% Preprocess
[R, U, alpha, rho_global] = afgenn_preprocess(Xtrain);
fprintf('Rank matrix built. alpha = [%d, %d], rho_global = %.4f\n\n', alpha(1), alpha(2), rho_global);

%% =========================================================================
%  SECTION 6 — Classification and Results Comparison
% =========================================================================

k_values    = [3, 5, 7];      % for KNN and WKNN
diff_values = [1, 2, 3, 4];   % for fixed-Diff FGENN

%% --- AFGENN ---
fprintf('--- Running AFGENN (adaptive) ---\n');
tic;
y_afgenn = afgenn_classify(Xtest, Xtrain, Ytrain, R, U, alpha, rho_global, 0.7, 0.1, 20, 1);
t_afgenn = toc;
acc_afgenn    = mean(y_afgenn == Ytest) * 100;
class1_idx    = Ytest == 1;
class2_idx    = Ytest == 2;
recall1_afgenn = mean(y_afgenn(class1_idx) == 1) * 100;
recall2_afgenn = mean(y_afgenn(class2_idx) == 2) * 100;

% Compute average neighbor count across test queries (diagnostic)
neighbor_counts = zeros(numel(Ytest), 1);
for i = 1:numel(Ytest)
    nb = afgenn_query(Xtest(i,:), R, U, alpha, rho_global, 0.7, 0.1, 20, 1);
    neighbor_counts(i) = numel(nb);
end
avg_nb_afgenn = mean(neighbor_counts);

fprintf('  Accuracy: %.1f%%, Recall C1: %.1f%%, Recall C2: %.1f%%, Avg |N(q)|: %.1f, Time: %.3fs\n\n', ...
    acc_afgenn, recall1_afgenn, recall2_afgenn, avg_nb_afgenn, t_afgenn);

%% --- KNN ---
acc_knn = zeros(1, numel(k_values));
time_knn = zeros(1, numel(k_values));
for ki = 1:numel(k_values)
    k = k_values(ki);
    tic;
    y_knn = knn_classify(Xtest, Xtrain, Ytrain, k);
    time_knn(ki) = toc;
    acc_knn(ki)  = mean(y_knn == Ytest) * 100;
end

%% --- Weighted KNN ---
acc_wknn = zeros(1, numel(k_values));
time_wknn = zeros(1, numel(k_values));
for ki = 1:numel(k_values)
    k = k_values(ki);
    tic;
    y_wknn = weighted_knn_classify(Xtest, Xtrain, Ytrain, k);
    time_wknn(ki) = toc;
    acc_wknn(ki)  = mean(y_wknn == Ytest) * 100;
end

%% --- Fixed-Diff FGENN ---
acc_fgenn = zeros(1, numel(diff_values));
time_fgenn = zeros(1, numel(diff_values));
for di = 1:numel(diff_values)
    d = diff_values(di);
    tic;
    y_fgenn = fgenn_fixed_classify(Xtest, Xtrain, Ytrain, R, U, alpha, d);
    time_fgenn(di) = toc;
    acc_fgenn(di)  = mean(y_fgenn == Ytest) * 100;
end

%% =========================================================================
%  Print Results Table
% =========================================================================

fprintf('============================================================\n');
fprintf('  Method                   | Accuracy (%%) | Time (s)\n');
fprintf('------------------------------------------------------------\n');
fprintf('  AFGENN (adaptive)        |   %5.1f      |  %.4f\n', acc_afgenn, t_afgenn);
fprintf('------------------------------------------------------------\n');
for ki = 1:numel(k_values)
    fprintf('  KNN (k=%d)               |   %5.1f      |  %.4f\n', k_values(ki), acc_knn(ki), time_knn(ki));
end
fprintf('------------------------------------------------------------\n');
for ki = 1:numel(k_values)
    fprintf('  Weighted KNN (k=%d)      |   %5.1f      |  %.4f\n', k_values(ki), acc_wknn(ki), time_wknn(ki));
end
fprintf('------------------------------------------------------------\n');
for di = 1:numel(diff_values)
    fprintf('  FGENN Fixed (Diff=%d)    |   %5.1f      |  %.4f\n', diff_values(di), acc_fgenn(di), time_fgenn(di));
end
fprintf('============================================================\n\n');

%% =========================================================================
%  FIGURE 1 — Decision boundaries (2-D)
% =========================================================================

figure('Name', 'AFGENN vs KNN — Decision Boundaries', 'NumberTitle', 'off');
res = 60;
x1_range = linspace(min(X_all(:,1))-0.5, max(X_all(:,1))+0.5, res);
x2_range = linspace(min(X_all(:,2))-0.5, max(X_all(:,2))+0.5, res);
[G1, G2] = meshgrid(x1_range, x2_range);
Xgrid = [G1(:), G2(:)];

% AFGENN decision map
Z_afgenn = afgenn_classify(Xgrid, Xtrain, Ytrain, R, U, alpha, rho_global, 0.7, 0.1, 20, 1);
Z_afgenn = reshape(Z_afgenn, res, res);

% KNN k=5 decision map
Z_knn = knn_classify(Xgrid, Xtrain, Ytrain, 5);
Z_knn = reshape(Z_knn, res, res);

subplot(1,2,1);
contourf(G1, G2, Z_afgenn, 'LineColor', 'none'); colormap(gca, parula); hold on;
gscatter(Xtest(:,1), Xtest(:,2), Ytest, 'rb', 'o^', 7);
title('AFGENN (adaptive)'); xlabel('Feature 1'); ylabel('Feature 2');
legend({'Class 1', 'Class 2'}, 'Location', 'best');
grid on; set(gca, 'FontSize', 10);

subplot(1,2,2);
contourf(G1, G2, Z_knn, 'LineColor', 'none'); colormap(gca, parula); hold on;
gscatter(Xtest(:,1), Xtest(:,2), Ytest, 'rb', 'o^', 7);
title('KNN (k=5)'); xlabel('Feature 1'); ylabel('Feature 2');
legend({'Class 1', 'Class 2'}, 'Location', 'best');
grid on; set(gca, 'FontSize', 10);

sgtitle('Decision Boundaries: AFGENN vs KNN', 'FontWeight', 'bold');

%% =========================================================================
%  FIGURE 2 — Accuracy bar chart comparison
% =========================================================================

figure('Name', 'Accuracy Comparison', 'NumberTitle', 'off');
labels  = {'AFGENN', 'KNN-3', 'KNN-5', 'KNN-7', 'WKNN-3', 'WKNN-5', 'WKNN-7', ...
           'FGENN-D1', 'FGENN-D2', 'FGENN-D3', 'FGENN-D4'};
accs    = [acc_afgenn, acc_knn, acc_wknn, acc_fgenn];
colors  = [repmat([0.2 0.6 0.9], 1, 1); ...
           repmat([0.9 0.5 0.2], 3, 1); ...
           repmat([0.4 0.8 0.4], 3, 1); ...
           repmat([0.8 0.3 0.3], 4, 1)];

bar_h = bar(accs, 'FaceColor', 'flat');
bar_h.CData = colors;
set(gca, 'XTickLabel', labels, 'XTickLabelRotation', 40, 'FontSize', 9);
ylabel('Accuracy (%)'); title('Classification Accuracy — All Methods');
ylim([max(0, min(accs)-5), 100]);
yline(acc_afgenn, '--b', 'AFGENN', 'LabelHorizontalAlignment', 'left');
grid on;

%% =========================================================================
%  FIGURE 3 — Neighbor count distribution (AFGENN adaptive vs fixed Diff)
% =========================================================================

figure('Name', 'Neighbor Count Distribution', 'NumberTitle', 'off');
nb_fixed2 = zeros(numel(Ytest), 1);
for i = 1:numel(Ytest)
    rq = afgenn_rank_q(Xtest(i,:), U, alpha);
    in_window = true(n_train, 1);
    for m = 1:2
        lo = max(1, int32(rq(m)) - 2);
        hi = min(alpha(m), int32(rq(m)) + 2);
        in_window = in_window & (R(:,m) >= lo) & (R(:,m) <= hi);
    end
    rm = all(bsxfun(@eq, R, rq), 2);
    in_window = in_window & ~rm;
    nb_fixed2(i) = sum(in_window);
end

histogram(neighbor_counts, 'BinWidth', 2, 'FaceColor', [0.2 0.6 0.9], 'FaceAlpha', 0.7); hold on;
histogram(nb_fixed2,       'BinWidth', 2, 'FaceColor', [0.8 0.3 0.3], 'FaceAlpha', 0.7);
legend('AFGENN (adaptive)', 'FGENN Fixed (Diff=2)');
xlabel('Number of extracted neighbors'); ylabel('Frequency');
title('Neighbor Count Distribution per Query');
grid on;

%% =========================================================================
%  FIGURE 4 — λ (density ratio) trace for a sample query
% =========================================================================

figure('Name', 'AFGENN Lambda Convergence Trace', 'NumberTitle', 'off');
[~, ~, lambda_trace] = afgenn_query(Xtest(1,:), R, U, alpha, rho_global, 0.7, 0.1, 20, 1);
plot(1:numel(lambda_trace), lambda_trace, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 6);
yline(1, '--k', '\lambda = 1 (target)');
yline(1.1, ':r', '\lambda = 1+\epsilon');
yline(0.9, ':g', '\lambda = 1-\epsilon');
xlabel('Iteration t'); ylabel('\lambda_q(Diff)');
title('Density Ratio \lambda Convergence (Sample Query)');
grid on;

fprintf('All figures generated. AFGENN MATLAB implementation complete.\n');
