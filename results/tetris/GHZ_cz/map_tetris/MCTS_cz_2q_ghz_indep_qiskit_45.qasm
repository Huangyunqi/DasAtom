OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
swap q[30],q[9];
cx q[9],q[3];
swap q[41],q[26];
swap q[17],q[3];
swap q[32],q[26];
swap q[36],q[21];
swap q[37],q[36];
cx q[17],q[32];
swap q[40],q[27];
cx q[32],q[37];
cx q[37],q[43];
swap q[27],q[13];
cx q[43],q[38];
cx q[38],q[40];
cx q[40],q[27];
swap q[22],q[17];
cx q[27],q[33];
swap q[39],q[27];
swap q[2],q[0];
swap q[4],q[2];
swap q[6],q[4];
swap q[1],q[0];
swap q[2],q[1];
swap q[11],q[6];
swap q[28],q[15];
cx q[33],q[17];
cx q[17],q[10];
cx q[10],q[12];
swap q[14],q[0];
swap q[42],q[29];
swap q[48],q[32];
cx q[12],q[27];
cx q[27],q[11];
cx q[11],q[2];
swap q[22],q[14];
cx q[2],q[15];
cx q[15],q[21];
cx q[21],q[28];
cx q[28],q[29];
cx q[29],q[22];
swap q[32],q[24];
swap q[34],q[25];
swap q[44],q[42];
swap q[20],q[13];
swap q[46],q[44];
swap q[34],q[20];
cx q[22],q[24];
swap q[39],q[34];
cx q[24],q[25];
cx q[25],q[32];
cx q[32],q[46];
cx q[46],q[39];
swap q[23],q[16];
swap q[42],q[35];
swap q[44],q[37];
swap q[37],q[23];
swap q[4],q[1];
cx q[39],q[37];
swap q[42],q[28];
swap q[28],q[14];
cx q[37],q[35];
cx q[35],q[30];
cx q[30],q[23];
swap q[15],q[7];
cx q[23],q[8];
cx q[8],q[1];
cx q[1],q[0];
cx q[0],q[14];
swap q[19],q[11];
cx q[14],q[15];
swap q[11],q[3];
cx q[15],q[3];
cx q[3],q[16];
swap q[12],q[6];
swap q[18],q[12];
swap q[47],q[32];
swap q[19],q[12];
cx q[16],q[18];
cx q[18],q[26];
swap q[41],q[34];
swap q[13],q[5];
cx q[26],q[11];
cx q[11],q[32];
cx q[32],q[45];
cx q[45],q[31];
cx q[31],q[19];
cx q[19],q[34];
cx q[34],q[13];
