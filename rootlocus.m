# plot root locus of spring-mass-damper

m = 1.3
b = 0.8
k = 1.3
omegan = sqrt(k/m)  # undamped natural frequency
cc = 2*m*omegan  # critical damping
zeta = b/cc  # damping factor

###---root locus plotting---###
num = [1];
den = [m 2*zeta*m*omegan omegan^2*m];

sys = tf(num,den)
roots(den)
rlocus(sys, 0.01, 0, 100);
v = [-2 2 -2 2];
axis(v);
axis('square');
title('Root-Locus Plot of G(s) = 1/[ms^2+2*z*m*omega_n*s+omega_n^2*m]');
print("smd_rootlocus.png");


