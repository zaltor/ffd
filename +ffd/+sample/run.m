rng(42);
K = randn(32,512)+1j*randn(32,512);
KH = linops.Matrix(K');
X = randn(32,4) + 1j*randn(32,4);
y = ffd.forward(KH, X);
[Xs, iterations] = ffd(y,KH,'Jthe',X*X','R',5,'precond',ffd.precond.Equalize);
[Xs2, iterations2] = ffd(y,KH,'Jthe',X*X','R',5,'precond',ffd.precond.None);
[Xs3, iterations3] = ffd(y,KH,'Jthe',X*X','R',4,'precond',ffd.precond.Equalize);
[Xs4, iterations4] = ffd(y,KH,'Jthe',X*X','R',4,'precond',ffd.precond.None);

figure(1); 
semilogy([iterations.ts, iterations2.ts, iterations3.ts, iterations4.ts], ...
         [iterations.fvals, iterations2.fvals, iterations3.fvals, iterations4.fvals]);
xlabel('time (seconds)'); ylabel('merit function value');
legend('preconditioned R=5','not preconditioned R=5','preconditioned R=4','not preconditioned R=4');

figure(2);
semilogy([iterations.ts, iterations2.ts, iterations3.ts, iterations4.ts], ...
         [iterations.Jerrs, iterations2.Jerrs, iterations3.Jerrs, iterations4.Jerrs]);
xlabel('time (seconds)'); ylabel('mutual intensity RMS error');
legend('preconditioned R=5','not preconditioned R=5','preconditioned R=4','not preconditioned R=4');