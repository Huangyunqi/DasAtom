OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
cx q[0],q[4];
cx q[4],q[8];
cx q[8],q[12];
cx q[4],q[0];
cx q[12],q[9];
cx q[8],q[4];
cx q[9],q[1];
cx q[12],q[8];
cx q[1],q[5];
cx q[9],q[12];
cx q[5],q[13];
cx q[1],q[9];
cx q[13],q[10];
cx q[5],q[1];
cx q[10],q[2];
cx q[13],q[5];
cx q[2],q[6];
cx q[10],q[13];
cx q[6],q[14];
cx q[2],q[10];
cx q[6],q[2];
cx q[14],q[6];
