OPENQASM 2.0;
include "qelib1.inc";
qreg q[36];
creg c[36];
swap q[26],q[15];
cx q[6],q[13];
swap q[3],q[2];
swap q[11],q[5];
swap q[13],q[7];
cx q[29],q[11];
swap q[15],q[14];
cx q[11],q[23];
cx q[29],q[35];
swap q[24],q[18];
swap q[12],q[0];
swap q[30],q[26];
cx q[28],q[26];
cx q[28],q[27];
cx q[26],q[24];
swap q[6],q[1];
swap q[14],q[13];
swap q[17],q[4];
cx q[27],q[17];
swap q[26],q[14];
swap q[10],q[3];
swap q[18],q[6];
swap q[35],q[22];
cx q[6],q[2];
cx q[24],q[18];
cx q[26],q[33];
cx q[18],q[25];
cx q[17],q[15];
cx q[1],q[4];
swap q[34],q[27];
cx q[2],q[9];
cx q[12],q[13];
cx q[15],q[19];
swap q[31],q[30];
cx q[33],q[20];
cx q[25],q[27];
cx q[10],q[4];
cx q[10],q[23];
cx q[9],q[19];
swap q[6],q[1];
cx q[12],q[30];
cx q[20],q[21];
cx q[26],q[22];
cx q[27],q[31];
cx q[21],q[31];
swap q[5],q[3];
cx q[1],q[3];
swap q[18],q[13];
cx q[7],q[3];
cx q[18],q[30];
