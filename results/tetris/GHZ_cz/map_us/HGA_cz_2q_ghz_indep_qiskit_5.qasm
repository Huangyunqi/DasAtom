OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
cx q[0],q[3];
cx q[3],q[6];
cx q[6],q[4];
cx q[4],q[1];
