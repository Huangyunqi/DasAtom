OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
swap q[36],q[31];
cx q[14],q[16];
cx q[8],q[16];
cx q[15],q[16];
cx q[22],q[16];
cx q[2],q[16];
cx q[9],q[16];
cx q[23],q[16];
cx q[30],q[16];
cx q[10],q[16];
cx q[17],q[16];
cx q[24],q[16];
cx q[18],q[16];
cx q[0],q[16];
cx q[7],q[16];
cx q[21],q[16];
swap q[36],q[30];
cx q[28],q[16];
swap q[38],q[24];
swap q[35],q[14];
swap q[18],q[12];
cx q[1],q[16];
cx q[3],q[16];
cx q[29],q[16];
cx q[31],q[16];
cx q[4],q[16];
swap q[43],q[31];
cx q[30],q[16];
cx q[24],q[16];
cx q[11],q[16];
cx q[14],q[16];
swap q[24],q[16];
swap q[45],q[42];
swap q[5],q[4];
swap q[10],q[4];
swap q[44],q[39];
cx q[18],q[24];
cx q[31],q[24];
swap q[33],q[20];
cx q[45],q[24];
swap q[42],q[28];
cx q[10],q[24];
swap q[28],q[22];
cx q[37],q[24];
cx q[25],q[24];
cx q[39],q[24];
cx q[32],q[24];
cx q[19],q[24];
cx q[26],q[24];
cx q[33],q[24];
cx q[22],q[24];
swap q[6],q[4];
swap q[11],q[4];
swap q[44],q[36];
cx q[11],q[24];
cx q[36],q[24];
swap q[32],q[24];
cx q[46],q[32];
swap q[13],q[11];
cx q[11],q[32];
cx q[20],q[32];
cx q[27],q[32];
cx q[40],q[32];
cx q[34],q[32];
cx q[47],q[32];
cx q[41],q[32];
cx q[48],q[32];
