OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
swap q[43],q[31];
swap q[15],q[10];
cx q[31],q[33];
swap q[18],q[10];
swap q[2],q[1];
swap q[35],q[14];
swap q[34],q[32];
swap q[16],q[14];
swap q[37],q[35];
swap q[13],q[6];
swap q[13],q[12];
swap q[19],q[12];
cx q[33],q[18];
swap q[37],q[23];
swap q[25],q[19];
swap q[47],q[39];
swap q[13],q[6];
cx q[18],q[2];
swap q[48],q[34];
cx q[2],q[16];
swap q[45],q[38];
swap q[19],q[13];
cx q[33],q[31];
cx q[18],q[33];
swap q[38],q[31];
cx q[2],q[18];
swap q[42],q[28];
swap q[29],q[28];
swap q[20],q[18];
swap q[30],q[29];
swap q[26],q[13];
swap q[38],q[29];
swap q[27],q[6];
swap q[38],q[24];
swap q[47],q[40];
cx q[16],q[32];
swap q[45],q[44];
cx q[32],q[23];
cx q[16],q[2];
swap q[11],q[4];
cx q[23],q[25];
swap q[45],q[38];
swap q[11],q[10];
cx q[32],q[16];
cx q[25],q[39];
swap q[16],q[10];
cx q[39],q[34];
cx q[34],q[19];
swap q[38],q[32];
swap q[21],q[7];
swap q[15],q[0];
cx q[19],q[31];
cx q[23],q[38];
cx q[25],q[23];
cx q[39],q[25];
swap q[36],q[15];
cx q[34],q[39];
cx q[19],q[34];
swap q[46],q[45];
cx q[31],q[30];
cx q[30],q[18];
cx q[31],q[19];
swap q[10],q[4];
cx q[30],q[31];
swap q[45],q[31];
cx q[18],q[11];
cx q[11],q[26];
swap q[8],q[0];
swap q[35],q[14];
swap q[15],q[9];
cx q[18],q[30];
cx q[26],q[24];
swap q[47],q[39];
cx q[11],q[18];
cx q[24],q[27];
cx q[26],q[11];
cx q[27],q[40];
cx q[24],q[26];
cx q[40],q[32];
cx q[27],q[24];
cx q[32],q[16];
cx q[40],q[27];
cx q[16],q[21];
cx q[32],q[40];
cx q[21],q[37];
cx q[16],q[32];
cx q[37],q[36];
cx q[21],q[16];
swap q[39],q[25];
cx q[36],q[31];
cx q[37],q[21];
swap q[25],q[24];
cx q[36],q[37];
swap q[28],q[22];
swap q[48],q[34];
swap q[34],q[33];
swap q[38],q[33];
swap q[16],q[7];
swap q[38],q[30];
swap q[3],q[2];
swap q[28],q[21];
swap q[14],q[2];
swap q[43],q[35];
swap q[19],q[13];
swap q[19],q[18];
swap q[25],q[18];
swap q[38],q[25];
cx q[31],q[10];
cx q[10],q[8];
cx q[31],q[36];
cx q[8],q[2];
swap q[43],q[38];
cx q[2],q[17];
cx q[17],q[12];
cx q[10],q[31];
cx q[8],q[10];
cx q[2],q[8];
cx q[12],q[9];
cx q[17],q[2];
cx q[9],q[1];
cx q[12],q[17];
cx q[1],q[0];
cx q[9],q[12];
cx q[0],q[15];
cx q[1],q[9];
cx q[15],q[24];
cx q[0],q[1];
cx q[24],q[22];
cx q[15],q[0];
cx q[22],q[30];
cx q[24],q[15];
cx q[30],q[16];
cx q[22],q[24];
cx q[16],q[14];
cx q[30],q[22];
cx q[14],q[21];
cx q[16],q[30];
cx q[21],q[35];
cx q[14],q[16];
cx q[35],q[42];
cx q[21],q[14];
cx q[42],q[43];
cx q[35],q[21];
cx q[42],q[35];
cx q[43],q[42];
