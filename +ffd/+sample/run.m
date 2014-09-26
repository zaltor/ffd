rng(42);
K = randn(32,420)+1j*randn(32,420);
KH = linops.Matrix(K');
X = randn(32,5) + 1j*randn(32,5);
y = ffd.forward(KH, X);
[Xs, iterations] = ffd(y,KH,'Xthe',X,'R',5);

figure(1); semilogy(iterations.fvals); 
xlabel('iteration'); ylabel('merit function value');

figure(2); semilogy(iterations.Jerrs);
xlabel('iteration'); ylabel('mutual intensity RMS error');