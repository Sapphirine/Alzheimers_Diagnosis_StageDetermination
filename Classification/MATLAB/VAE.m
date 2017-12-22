clear all;

% Define fixed parameters
a_0 = 10e-16;
b_0 = 10e-16;

e_0 = 1;
f_0 = 1;

T = 500;

% Read in the files
FILE = 'ADNIMERGE_FILTERED_NOHEAD.csv';

DATA = csvread(FILE);
XDATA = DATA(:, 2:end);
XDATA = XDATA';				% Want each row to be one feature
YDATA = DATA(:, 1);

N = length( YDATA );
d = size( XDATA, 1 );

[trainInd, ~, testInd] = dividerand(N, 0.8, 0, 0.2);

XDATA_test = XDATA(:, testInd);
YDATA_test = YDATA(testInd);

XDATA = XDATA(:, trainInd);
YDATA = YDATA(trainInd);
N = length( YDATA );

% Function for computing L:
L = @(X, Y, a, b, e, f, sig, mu) ...
	(N/2) * (psi(e) - log(f)) - 0.5 * (e/f) * sum( (Y - X' * mu).^2+ diag(X' * sig * X)) ...
	+ (e_0 - 1) * (psi(e) - log(f)) - f_0 * (e/f) ...
	+ sum( (a_0 - 1) * (psi(a) - log(b)) - b_0 * (a ./ b) ) ...
	+ 0.5 * sum(psi(a) - log(b)) - 0.5 * sum( (a ./ b) .* (diag(sig) + mu.^2) ) ...
	+ e - log(f) + gammaln(e) + (1-e) * psi(e) ...
	+ sum( a - log(b) + gammaln(a) + (1 - a) .* psi(a) );

% Initialize the parameters
a_var = abs( randn(d, 1) );
b_var = abs( randn(d, 1) );
e_var = abs( randn(1, 1) );
f_var = abs( randn(1, 1) );
sig_var = randn(d, d);
mu_var =  randn(d, 1);


% Precompute some fixed values
XY_SUM = zeros(d, 1);
X_OUTER_SUM = zeros(d, d);
for iter = 1:N
	XY_SUM = XY_SUM + YDATA(iter) * XDATA(:, iter);
	X_OUTER_SUM = X_OUTER_SUM + XDATA(:, iter) * XDATA(:, iter)';
end

L_hist = zeros(1, T);
for iter = 1:T
	% Update lambda
	e_var = e_0 + N/2;
	f_var = f_0 + 0.5 * sum( (YDATA - XDATA' * mu_var).^2 + diag(XDATA' * sig_var * XDATA));

	% Update w
	sig_var = inv( diag( a_var ./ b_var ) + (e_var/f_var) * X_OUTER_SUM );
	mu_var = sig_var * (e_var / f_var) * XY_SUM;

	% Update alphas
	for k = 1:d
		a_var(k) = a_0 + 0.5;
	end

	b_var = b_0 + 0.5 * (diag(sig_var) + mu_var.^2);

	L_hist(iter) = L(XDATA, YDATA, a_var, b_var, e_var, f_var, sig_var, mu_var);
end

figure(1); clf;

subplot(2, 1, 1);
plot( real(L_hist) )
xlabel(' Number of Iterations ')
ylabel(' Variational Objective Value')

subplot(2, 1, 2);
plot( b_var ./ a_var)
axis tight;
xlabel('k')
ylabel('1/E_q[\alpha_k]')


disp(['1 / E_q[\lambda]: ',  num2str(f_var / e_var)])
y_pred = zeros(length(trainInd), 1);
for iter = 1:length(trainInd)
	y_pred(iter) = XDATA(:,iter)' * mu_var;
end

y_pred(y_pred > 0.5) = 1;
y_pred(y_pred <= 0.5) = 0;

disp(['Train Accuracy: ', num2str( sum(abs(y_pred == YDATA)) / length(trainInd) )] )

y_pred = zeros(length(testInd), 1);
for iter = 1:length(testInd)
	y_pred(iter) = XDATA_test(:,iter)' * mu_var;
end

y_pred(y_pred > 0.5) = 1;
y_pred(y_pred <= 0.5) = 0;
disp(['Test Accuracy: ', num2str( sum(abs(y_pred == YDATA_test)) / length(trainInd) )] )