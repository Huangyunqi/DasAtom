OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
swap q[18],q[2];
swap q[18],q[9];
swap q[36],q[22];
swap q[14],q[7];
swap q[29],q[16];
swap q[3],q[1];
swap q[40],q[33];
swap q[26],q[25];
swap q[44],q[43];
swap q[46],q[38];
swap q[48],q[39];
swap q[46],q[44];
swap q[38],q[25];
cx q[9],q[22];
swap q[12],q[10];
cx q[33],q[18];
swap q[8],q[1];
cx q[8],q[21];
swap q[42],q[36];
swap q[38],q[36];
cx q[8],q[1];
cx q[19],q[17];
swap q[22],q[9];
cx q[36],q[28];
cx q[36],q[15];
cx q[33],q[46];
cx q[19],q[34];
swap q[25],q[12];
swap q[13],q[5];
cx q[9],q[10];
cx q[32],q[16];
cx q[32],q[30];
swap q[34],q[33];
cx q[37],q[25];
swap q[8],q[1];
swap q[12],q[4];
swap q[37],q[24];
swap q[47],q[40];
swap q[5],q[3];
swap q[21],q[8];
cx q[8],q[3];
swap q[39],q[33];
cx q[16],q[2];
cx q[30],q[15];
swap q[42],q[35];
swap q[43],q[35];
cx q[22],q[38];
swap q[2],q[0];
cx q[28],q[31];
swap q[20],q[19];
swap q[19],q[18];
cx q[19],q[33];
cx q[21],q[35];
cx q[4],q[2];
cx q[4],q[6];
cx q[17],q[12];
cx q[2],q[5];
cx q[23],q[14];
cx q[38],q[45];
cx q[24],q[12];
cx q[0],q[7];
cx q[45],q[47];
cx q[47],q[46];
swap q[33],q[27];
swap q[43],q[36];
swap q[18],q[16];
cx q[31],q[36];
swap q[48],q[46];
swap q[46],q[43];
cx q[25],q[11];
swap q[13],q[6];
swap q[21],q[14];
swap q[21],q[16];
cx q[35],q[43];
cx q[43],q[42];
swap q[20],q[13];
cx q[36],q[21];
cx q[11],q[27];
swap q[40],q[32];
swap q[42],q[36];
cx q[23],q[32];
cx q[10],q[26];
cx q[26],q[5];
cx q[21],q[7];
swap q[32],q[20];
swap q[11],q[3];
cx q[11],q[20];
swap q[44],q[36];
cx q[32],q[44];
swap q[23],q[16];
cx q[23],q[39];
