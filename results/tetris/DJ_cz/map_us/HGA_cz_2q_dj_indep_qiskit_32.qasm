OPENQASM 2.0;
include "qelib1.inc";
qreg q[36];
creg c[36];
cx q[12],q[14];
swap q[31],q[27];
cx q[7],q[14];
cx q[13],q[14];
cx q[19],q[14];
cx q[2],q[14];
cx q[8],q[14];
swap q[32],q[31];
cx q[20],q[14];
swap q[33],q[22];
swap q[31],q[30];
cx q[26],q[14];
cx q[9],q[14];
swap q[31],q[26];
cx q[15],q[14];
cx q[21],q[14];
cx q[16],q[14];
swap q[31],q[30];
cx q[0],q[14];
cx q[6],q[14];
cx q[18],q[14];
cx q[24],q[14];
cx q[1],q[14];
cx q[3],q[14];
cx q[25],q[14];
cx q[27],q[14];
swap q[31],q[25];
cx q[4],q[14];
swap q[33],q[27];
swap q[9],q[5];
cx q[32],q[14];
cx q[22],q[14];
cx q[10],q[14];
cx q[26],q[14];
swap q[16],q[11];
cx q[25],q[14];
cx q[27],q[14];
cx q[9],q[14];
cx q[16],q[14];
cx q[17],q[14];
cx q[28],q[14];
