OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
cx q[0],q[3];
cx q[6],q[3];
cx q[4],q[3];
cx q[5],q[3];
cx q[1],q[3];
swap q[2],q[0];
cx q[0],q[3];
swap q[8],q[7];
cx q[7],q[3];
