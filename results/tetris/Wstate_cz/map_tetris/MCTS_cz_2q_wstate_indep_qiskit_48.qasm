OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
swap q[19],q[6];
swap q[31],q[19];
swap q[8],q[2];
swap q[38],q[30];
swap q[16],q[8];
swap q[14],q[9];
swap q[41],q[25];
cx q[23],q[31];
cx q[31],q[30];
cx q[30],q[16];
cx q[31],q[23];
cx q[16],q[9];
cx q[30],q[31];
cx q[16],q[30];
swap q[27],q[19];
swap q[45],q[43];
swap q[45],q[40];
swap q[18],q[12];
cx q[9],q[25];
cx q[25],q[26];
cx q[26],q[27];
cx q[9],q[16];
cx q[25],q[9];
swap q[35],q[28];
swap q[28],q[21];
cx q[27],q[41];
swap q[21],q[0];
cx q[26],q[25];
cx q[27],q[26];
cx q[41],q[40];
cx q[40],q[32];
cx q[32],q[18];
cx q[18],q[13];
cx q[13],q[6];
cx q[6],q[5];
cx q[41],q[27];
cx q[40],q[41];
swap q[2],q[0];
cx q[5],q[2];
cx q[32],q[40];
swap q[29],q[16];
swap q[47],q[45];
cx q[2],q[16];
swap q[25],q[11];
swap q[31],q[25];
cx q[16],q[15];
cx q[15],q[24];
swap q[27],q[20];
swap q[34],q[27];
swap q[44],q[43];
swap q[47],q[34];
cx q[24],q[45];
cx q[45],q[46];
cx q[18],q[32];
swap q[8],q[0];
cx q[46],q[31];
cx q[31],q[44];
swap q[34],q[33];
cx q[13],q[18];
cx q[6],q[13];
cx q[44],q[47];
swap q[16],q[8];
swap q[25],q[16];
cx q[5],q[6];
swap q[3],q[1];
swap q[34],q[27];
cx q[47],q[33];
cx q[33],q[25];
cx q[2],q[5];
cx q[8],q[2];
swap q[19],q[3];
cx q[15],q[8];
cx q[24],q[15];
swap q[9],q[1];
cx q[25],q[27];
cx q[27],q[19];
swap q[19],q[11];
swap q[39],q[24];
cx q[45],q[39];
swap q[36],q[21];
swap q[3],q[2];
cx q[11],q[9];
cx q[9],q[17];
cx q[46],q[45];
cx q[31],q[46];
cx q[44],q[31];
cx q[17],q[24];
cx q[47],q[44];
cx q[24],q[38];
cx q[33],q[47];
swap q[29],q[28];
cx q[38],q[36];
cx q[36],q[29];
cx q[25],q[33];
cx q[27],q[25];
swap q[15],q[2];
cx q[11],q[27];
cx q[9],q[11];
cx q[17],q[9];
swap q[14],q[0];
swap q[48],q[46];
swap q[46],q[43];
cx q[24],q[17];
swap q[21],q[7];
cx q[38],q[24];
cx q[29],q[15];
cx q[36],q[38];
cx q[29],q[36];
cx q[15],q[7];
cx q[7],q[14];
swap q[46],q[45];
cx q[14],q[35];
cx q[15],q[29];
cx q[35],q[43];
cx q[7],q[15];
cx q[43],q[22];
cx q[14],q[7];
cx q[35],q[14];
cx q[22],q[21];
cx q[43],q[35];
cx q[22],q[43];
swap q[18],q[10];
cx q[21],q[37];
cx q[37],q[42];
cx q[42],q[45];
cx q[21],q[22];
cx q[37],q[21];
swap q[33],q[18];
swap q[11],q[4];
swap q[18],q[11];
swap q[34],q[20];
cx q[45],q[33];
cx q[33],q[18];
cx q[18],q[20];
cx q[20],q[12];
cx q[42],q[37];
cx q[45],q[42];
cx q[33],q[45];
cx q[18],q[33];
cx q[20],q[18];
cx q[12],q[20];
