OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
cx q[1],q[9];
swap q[15],q[7];
cx q[9],q[8];
swap q[9],q[4];
cx q[4],q[0];
cx q[6],q[10];
swap q[7],q[5];
swap q[14],q[13];
cx q[9],q[5];
swap q[15],q[14];
swap q[7],q[2];
cx q[9],q[11];
cx q[9],q[14];
swap q[11],q[7];
cx q[5],q[13];
cx q[13],q[8];
swap q[1],q[0];
swap q[14],q[9];
swap q[7],q[6];
swap q[4],q[0];
swap q[13],q[10];
swap q[2],q[0];
cx q[10],q[11];
swap q[13],q[8];
cx q[5],q[6];
swap q[15],q[11];
cx q[4],q[9];
cx q[1],q[6];
cx q[13],q[15];
cx q[4],q[0];
cx q[9],q[12];
swap q[7],q[6];
cx q[1],q[0];
cx q[8],q[0];
swap q[9],q[6];
swap q[15],q[14];
cx q[9],q[12];
cx q[9],q[14];
cx q[8],q[12];
