OPENQASM 2.0;
include "qelib1.inc";
qreg q[64];
creg c[64];
swap q[26],q[12];
swap q[39],q[23];
swap q[45],q[44];
swap q[38],q[29];
cx q[12],q[14];
cx q[14],q[23];
swap q[44],q[27];
cx q[23],q[37];
cx q[37],q[30];
swap q[57],q[50];
swap q[40],q[16];
cx q[30],q[21];
swap q[17],q[11];
cx q[21],q[31];
cx q[31],q[45];
cx q[45],q[38];
cx q[38],q[62];
swap q[24],q[17];
swap q[11],q[1];
cx q[62],q[44];
cx q[44],q[50];
cx q[50],q[40];
swap q[9],q[1];
cx q[40],q[24];
swap q[12],q[6];
swap q[18],q[12];
cx q[24],q[9];
swap q[7],q[4];
cx q[9],q[18];
cx q[18],q[16];
swap q[39],q[31];
cx q[16],q[10];
cx q[10],q[4];
swap q[31],q[29];
cx q[4],q[11];
swap q[29],q[27];
cx q[11],q[27];
cx q[27],q[42];
swap q[60],q[53];
swap q[53],q[44];
cx q[42],q[43];
cx q[43],q[44];
swap q[16],q[8];
swap q[25],q[16];
swap q[21],q[5];
swap q[27],q[21];
swap q[54],q[38];
swap q[38],q[37];
cx q[44],q[28];
cx q[28],q[35];
swap q[48],q[40];
swap q[40],q[32];
swap q[32],q[8];
swap q[10],q[8];
cx q[35],q[25];
cx q[25],q[33];
swap q[31],q[28];
swap q[40],q[25];
swap q[63],q[61];
cx q[33],q[27];
swap q[61],q[60];
cx q[27],q[37];
swap q[60],q[52];
swap q[52],q[50];
swap q[7],q[6];
swap q[6],q[4];
swap q[4],q[2];
swap q[9],q[2];
cx q[37],q[13];
swap q[50],q[32];
cx q[13],q[10];
cx q[10],q[28];
cx q[28],q[19];
cx q[19],q[25];
swap q[17],q[9];
swap q[26],q[19];
swap q[57],q[43];
swap q[38],q[22];
swap q[59],q[56];
cx q[25],q[41];
cx q[41],q[32];
cx q[32],q[17];
cx q[17],q[1];
swap q[38],q[36];
swap q[59],q[44];
swap q[20],q[12];
swap q[27],q[20];
cx q[1],q[19];
swap q[60],q[42];
cx q[19],q[43];
cx q[43],q[29];
cx q[29],q[36];
cx q[36],q[44];
cx q[44],q[27];
cx q[27],q[42];
swap q[3],q[1];
swap q[56],q[48];
swap q[48],q[32];
swap q[32],q[25];
cx q[42],q[25];
swap q[12],q[10];
cx q[25],q[1];
cx q[1],q[10];
cx q[10],q[16];
