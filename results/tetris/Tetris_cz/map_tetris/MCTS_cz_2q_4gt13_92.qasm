OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
cx q[7],q[1];
cx q[4],q[2];
cx q[4],q[5];
cx q[2],q[4];
cx q[5],q[2];
cx q[5],q[4];
cx q[2],q[4];
cx q[5],q[2];
cx q[4],q[5];
cx q[4],q[7];
cx q[1],q[4];
cx q[1],q[7];
cx q[4],q[7];
cx q[1],q[4];
cx q[7],q[1];
cx q[7],q[1];
cx q[4],q[5];
cx q[2],q[4];
cx q[5],q[2];
cx q[5],q[4];
cx q[2],q[4];
cx q[5],q[2];
cx q[4],q[5];
cx q[4],q[7];
cx q[1],q[4];
cx q[1],q[7];
cx q[4],q[7];
cx q[1],q[4];
cx q[2],q[4];
cx q[7],q[1];
