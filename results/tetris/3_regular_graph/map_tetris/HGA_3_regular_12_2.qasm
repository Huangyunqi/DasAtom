OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
cx q[11],q[7];
swap q[8],q[0];
cx q[11],q[15];
swap q[6],q[2];
cx q[12],q[8];
cx q[12],q[14];
cx q[7],q[5];
cx q[8],q[4];
cx q[14],q[6];
cx q[8],q[9];
swap q[6],q[1];
swap q[15],q[13];
cx q[1],q[5];
cx q[7],q[10];
swap q[12],q[4];
cx q[14],q[10];
swap q[14],q[11];
swap q[8],q[5];
swap q[11],q[7];
cx q[13],q[9];
swap q[14],q[13];
cx q[4],q[6];
cx q[4],q[5];
cx q[14],q[6];
cx q[10],q[9];
cx q[13],q[12];
cx q[1],q[6];
cx q[9],q[11];
cx q[12],q[8];
cx q[11],q[7];
cx q[9],q[13];
cx q[9],q[6];
swap q[15],q[11];
swap q[9],q[4];
cx q[9],q[14];
swap q[14],q[12];
swap q[5],q[4];
cx q[4],q[12];
cx q[15],q[14];
cx q[9],q[14];
cx q[13],q[14];
swap q[6],q[4];
cx q[6],q[10];
swap q[12],q[4];
swap q[14],q[10];
cx q[4],q[1];
cx q[14],q[12];
swap q[5],q[1];
cx q[12],q[8];
swap q[10],q[7];
cx q[14],q[10];
cx q[10],q[8];
cx q[13],q[5];
cx q[8],q[5];
