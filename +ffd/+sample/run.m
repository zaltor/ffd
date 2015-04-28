rng(42);
K = randn(32,512)+1j*randn(32,512);
KH = linops.Matrix(K');
X = randn(32,4) + 1j*randn(32,4);
y = ffd.forward(KH, X);
[Xs, iterations] = ffd(y,KH,'Jthe',X*X','R',32,'descent',ffd.descent.Equalized);
[Xs2, iterations2] = ffd(y,KH,'Jthe',X*X','R',32,'descent',ffd.descent.Steepest);
[Xs3, iterations3] = ffd(y,KH,'Jthe',X*X','R',4,'descent',ffd.descent.Equalized);
[Xs4, iterations4] = ffd(y,KH,'Jthe',X*X','R',4,'descent',ffd.descent.Steepest);

figure(1); 
semilogy([iterations.ts, iterations2.ts, iterations3.ts, iterations4.ts], ...
         [iterations.fvals, iterations2.fvals, iterations3.fvals, iterations4.fvals]);
xlabel('time (seconds)'); ylabel('merit function value');
legend('equalized R=32','steepest R=32','equalized R=4','steepest R=4');

figure(2);
semilogy([iterations.ts, iterations2.ts, iterations3.ts, iterations4.ts], ...
         [iterations.Jerrs, iterations2.Jerrs, iterations3.Jerrs, iterations4.Jerrs]);
xlabel('time (seconds)'); ylabel('mutual intensity RMS error');
legend('equalized R=32','steepest R=32','equalized R=4','steepest R=4');
