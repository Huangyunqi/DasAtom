OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
swap q[6],q[2];
swap q[14],q[9];
cx q[6],q[9];
swap q[11],q[3];
cx q[4],q[9];
cx q[5],q[9];
cx q[1],q[9];
cx q[14],q[9];
cx q[12],q[9];
cx q[11],q[9];
cx q[13],q[9];
swap q[2],q[1];
cx q[1],q[9];
