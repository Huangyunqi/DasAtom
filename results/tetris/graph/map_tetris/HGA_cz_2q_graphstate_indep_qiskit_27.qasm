OPENQASM 2.0;
include "qelib1.inc";
qreg q[36];
creg c[36];
cx q[9],q[27];
swap q[13],q[12];
cx q[27],q[35];
swap q[1],q[0];
swap q[30],q[24];
swap q[11],q[3];
swap q[29],q[21];
swap q[30],q[12];
swap q[3],q[2];
swap q[33],q[22];
cx q[13],q[2];
swap q[25],q[24];
cx q[13],q[21];
swap q[9],q[3];
cx q[31],q[33];
cx q[21],q[28];
swap q[23],q[17];
swap q[13],q[12];
cx q[3],q[1];
swap q[32],q[26];
cx q[2],q[13];
swap q[22],q[15];
swap q[6],q[2];
cx q[22],q[35];
swap q[26],q[20];
swap q[34],q[23];
cx q[13],q[15];
cx q[15],q[10];
swap q[3],q[2];
cx q[31],q[34];
swap q[17],q[15];
swap q[25],q[12];
cx q[10],q[3];
swap q[33],q[28];
cx q[20],q[9];
swap q[28],q[23];
cx q[8],q[12];
cx q[3],q[15];
swap q[25],q[24];
cx q[9],q[11];
swap q[13],q[12];
cx q[23],q[11];
swap q[20],q[15];
swap q[33],q[27];
swap q[17],q[4];
cx q[1],q[19];
swap q[28],q[27];
cx q[15],q[17];
cx q[8],q[26];
cx q[13],q[27];
cx q[20],q[25];
cx q[28],q[17];
cx q[22],q[19];
cx q[27],q[34];
cx q[25],q[26];
