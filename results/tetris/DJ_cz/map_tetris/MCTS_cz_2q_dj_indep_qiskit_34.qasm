OPENQASM 2.0;
include "qelib1.inc";
qreg q[36];
creg c[36];
cx q[13],q[3];
cx q[2],q[3];
cx q[7],q[3];
cx q[4],q[3];
swap q[14],q[3];
swap q[31],q[24];
cx q[24],q[14];
swap q[10],q[5];
swap q[29],q[28];
swap q[29],q[26];
swap q[11],q[9];
cx q[10],q[14];
swap q[29],q[22];
cx q[32],q[14];
cx q[16],q[14];
cx q[8],q[14];
cx q[15],q[14];
swap q[5],q[4];
swap q[30],q[24];
swap q[34],q[32];
cx q[28],q[14];
cx q[0],q[14];
cx q[27],q[14];
cx q[20],q[14];
cx q[12],q[14];
cx q[26],q[14];
cx q[3],q[14];
cx q[18],q[14];
cx q[6],q[14];
swap q[31],q[26];
swap q[29],q[23];
swap q[23],q[16];
cx q[9],q[14];
swap q[29],q[28];
cx q[22],q[14];
cx q[4],q[14];
cx q[19],q[14];
cx q[25],q[14];
cx q[24],q[14];
cx q[32],q[14];
cx q[26],q[14];
swap q[35],q[22];
cx q[16],q[14];
swap q[33],q[26];
cx q[28],q[14];
cx q[17],q[14];
cx q[22],q[14];
cx q[26],q[14];
cx q[21],q[14];
