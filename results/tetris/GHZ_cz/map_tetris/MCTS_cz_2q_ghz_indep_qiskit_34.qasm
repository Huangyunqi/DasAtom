OPENQASM 2.0;
include "qelib1.inc";
qreg q[36];
creg c[36];
swap q[26],q[24];
swap q[31],q[30];
cx q[3],q[21];
cx q[21],q[33];
cx q[33],q[35];
cx q[35],q[17];
cx q[17],q[23];
cx q[23],q[22];
swap q[15],q[10];
swap q[24],q[12];
cx q[22],q[26];
cx q[26],q[34];
cx q[34],q[31];
cx q[31],q[25];
swap q[11],q[8];
swap q[24],q[20];
cx q[25],q[19];
swap q[7],q[0];
swap q[15],q[7];
cx q[19],q[7];
cx q[7],q[12];
cx q[12],q[8];
cx q[8],q[6];
cx q[6],q[18];
cx q[18],q[14];
swap q[29],q[23];
swap q[33],q[32];
swap q[33],q[29];
swap q[17],q[5];
swap q[22],q[17];
cx q[14],q[28];
cx q[28],q[20];
swap q[31],q[30];
swap q[32],q[31];
cx q[20],q[24];
cx q[24],q[27];
cx q[27],q[15];
cx q[15],q[23];
swap q[4],q[2];
cx q[23],q[10];
cx q[10],q[11];
cx q[11],q[16];
cx q[16],q[29];
cx q[29],q[22];
cx q[22],q[32];
swap q[7],q[2];
swap q[14],q[7];
cx q[32],q[14];
cx q[14],q[0];
swap q[4],q[3];
cx q[0],q[3];
cx q[3],q[13];
