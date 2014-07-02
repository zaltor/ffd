rng(42);
K = randn(32,420)+1j*randn(32,420);
sys = ffd.sys.matrix(K);
X = randn(32,5) + 1j*randn(32,5);
y = ffd.forward(sys, X);
[Xs, fvals, yerrs, Jerrs] = ffd(y,sys,'Xthe',X,'R',5);

figure(1); semilogy(fvals); 
xlabel('iteration'); ylabel('merit function value');

figure(2); semilogy(Jerrs);
xlabel('iteration'); ylabel('mutual intensity RMS error');