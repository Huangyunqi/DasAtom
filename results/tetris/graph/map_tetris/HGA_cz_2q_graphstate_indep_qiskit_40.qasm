OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
cx q[41],q[40];
cx q[44],q[30];
swap q[19],q[3];
cx q[40],q[38];
cx q[36],q[42];
cx q[44],q[39];
swap q[28],q[21];
swap q[16],q[15];
swap q[30],q[24];
cx q[19],q[33];
swap q[43],q[28];
cx q[32],q[16];
swap q[6],q[4];
cx q[45],q[43];
cx q[24],q[12];
swap q[14],q[2];
swap q[41],q[34];
swap q[24],q[19];
cx q[14],q[28];
swap q[45],q[43];
swap q[48],q[40];
cx q[24],q[23];
cx q[23],q[8];
swap q[19],q[5];
cx q[28],q[15];
swap q[46],q[45];
swap q[43],q[42];
swap q[24],q[10];
cx q[8],q[7];
cx q[34],q[13];
swap q[36],q[21];
swap q[46],q[39];
swap q[27],q[6];
cx q[24],q[37];
swap q[8],q[7];
swap q[11],q[10];
cx q[32],q[27];
swap q[42],q[28];
cx q[24],q[10];
swap q[47],q[46];
swap q[14],q[8];
cx q[37],q[25];
swap q[6],q[5];
swap q[47],q[41];
cx q[15],q[3];
swap q[38],q[25];
swap q[20],q[12];
swap q[21],q[9];
cx q[25],q[18];
swap q[41],q[34];
cx q[38],q[29];
cx q[9],q[4];
swap q[24],q[18];
swap q[29],q[14];
cx q[34],q[13];
swap q[45],q[38];
swap q[10],q[9];
cx q[24],q[22];
swap q[39],q[33];
cx q[14],q[1];
cx q[1],q[4];
swap q[43],q[37];
swap q[27],q[20];
swap q[17],q[16];
cx q[33],q[19];
swap q[28],q[21];
cx q[17],q[12];
swap q[8],q[2];
cx q[39],q[36];
swap q[27],q[26];
cx q[2],q[5];
cx q[22],q[38];
swap q[47],q[41];
cx q[21],q[9];
cx q[3],q[5];
cx q[38],q[36];
swap q[27],q[19];
swap q[38],q[29];
cx q[41],q[27];
cx q[41],q[20];
cx q[40],q[38];
swap q[25],q[12];
swap q[38],q[37];
cx q[40],q[25];
cx q[38],q[26];
