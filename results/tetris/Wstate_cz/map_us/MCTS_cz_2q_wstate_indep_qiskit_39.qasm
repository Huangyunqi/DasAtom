OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
cx q[0],q[7];
cx q[7],q[14];
cx q[14],q[21];
cx q[7],q[0];
cx q[21],q[28];
cx q[14],q[7];
cx q[28],q[35];
cx q[35],q[42];
cx q[42],q[36];
cx q[21],q[14];
cx q[28],q[21];
cx q[35],q[28];
cx q[42],q[35];
cx q[36],q[22];
cx q[36],q[42];
cx q[22],q[8];
cx q[22],q[36];
cx q[8],q[1];
cx q[8],q[22];
cx q[1],q[15];
cx q[15],q[29];
cx q[29],q[43];
cx q[43],q[37];
cx q[1],q[8];
cx q[15],q[1];
cx q[37],q[23];
cx q[23],q[9];
cx q[29],q[15];
cx q[9],q[2];
cx q[2],q[16];
cx q[43],q[29];
cx q[37],q[43];
cx q[16],q[30];
cx q[30],q[44];
cx q[23],q[37];
cx q[9],q[23];
cx q[2],q[9];
cx q[44],q[38];
cx q[38],q[24];
cx q[24],q[10];
cx q[10],q[3];
cx q[3],q[17];
cx q[17],q[31];
cx q[31],q[45];
cx q[45],q[39];
cx q[39],q[25];
cx q[16],q[2];
cx q[25],q[11];
cx q[11],q[4];
cx q[30],q[16];
cx q[44],q[30];
cx q[38],q[44];
cx q[24],q[38];
cx q[4],q[18];
cx q[18],q[32];
cx q[10],q[24];
cx q[32],q[46];
cx q[3],q[10];
cx q[17],q[3];
cx q[31],q[17];
cx q[46],q[40];
cx q[40],q[26];
cx q[26],q[12];
cx q[12],q[5];
cx q[45],q[31];
cx q[39],q[45];
cx q[25],q[39];
cx q[11],q[25];
cx q[4],q[11];
cx q[18],q[4];
cx q[32],q[18];
cx q[46],q[32];
cx q[40],q[46];
cx q[26],q[40];
cx q[12],q[26];
cx q[5],q[12];
