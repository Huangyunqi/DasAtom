OPENQASM 2.0;
include "qelib1.inc";
qreg q[36];
creg c[36];
swap q[26],q[14];
cx q[23],q[16];
swap q[32],q[22];
cx q[12],q[7];
swap q[11],q[5];
swap q[5],q[3];
cx q[9],q[21];
swap q[16],q[5];
cx q[23],q[20];
swap q[6],q[0];
swap q[31],q[30];
swap q[6],q[2];
swap q[10],q[9];
cx q[26],q[33];
cx q[26],q[24];
swap q[28],q[22];
cx q[18],q[24];
cx q[7],q[3];
swap q[17],q[11];
cx q[22],q[17];
cx q[12],q[14];
cx q[20],q[13];
cx q[13],q[9];
swap q[31],q[24];
swap q[33],q[27];
swap q[7],q[3];
swap q[34],q[22];
swap q[30],q[18];
swap q[18],q[6];
cx q[29],q[15];
cx q[27],q[16];
cx q[18],q[24];
cx q[29],q[35];
cx q[15],q[2];
cx q[17],q[22];
cx q[10],q[4];
cx q[30],q[32];
cx q[14],q[1];
cx q[9],q[6];
swap q[10],q[4];
cx q[18],q[19];
cx q[34],q[32];
cx q[5],q[16];
cx q[2],q[6];
cx q[7],q[25];
cx q[25],q[33];
cx q[24],q[19];
cx q[35],q[33];
cx q[21],q[10];
swap q[8],q[1];
cx q[22],q[8];
