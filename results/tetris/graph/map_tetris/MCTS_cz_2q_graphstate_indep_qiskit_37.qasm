OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
swap q[34],q[18];
cx q[29],q[38];
swap q[19],q[17];
cx q[34],q[41];
swap q[7],q[2];
swap q[22],q[15];
swap q[4],q[2];
swap q[45],q[42];
swap q[14],q[8];
cx q[17],q[10];
cx q[17],q[30];
cx q[30],q[37];
swap q[12],q[4];
swap q[34],q[33];
swap q[42],q[35];
swap q[45],q[39];
swap q[39],q[33];
cx q[16],q[9];
swap q[31],q[28];
cx q[27],q[12];
cx q[12],q[19];
cx q[9],q[3];
cx q[19],q[26];
swap q[3],q[1];
cx q[33],q[20];
swap q[44],q[43];
swap q[43],q[22];
cx q[20],q[25];
cx q[27],q[48];
swap q[11],q[3];
cx q[8],q[23];
cx q[48],q[40];
swap q[20],q[6];
cx q[23],q[24];
cx q[35],q[14];
cx q[38],q[43];
cx q[35],q[28];
cx q[14],q[1];
cx q[11],q[25];
cx q[10],q[31];
swap q[47],q[41];
swap q[22],q[15];
cx q[33],q[20];
cx q[11],q[13];
cx q[15],q[3];
cx q[15],q[22];
cx q[3],q[5];
cx q[39],q[44];
cx q[47],q[44];
swap q[46],q[25];
cx q[29],q[31];
cx q[28],q[37];
cx q[20],q[13];
swap q[9],q[8];
cx q[9],q[25];
cx q[25],q[26];
swap q[5],q[4];
cx q[16],q[4];
cx q[24],q[22];
swap q[46],q[40];
cx q[43],q[46];
