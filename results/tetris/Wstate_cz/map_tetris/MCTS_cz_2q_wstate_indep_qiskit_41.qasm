OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
swap q[18],q[10];
cx q[7],q[10];
swap q[23],q[22];
cx q[10],q[23];
cx q[10],q[7];
swap q[26],q[13];
cx q[23],q[32];
cx q[32],q[39];
cx q[39],q[27];
cx q[27],q[11];
cx q[23],q[10];
cx q[11],q[20];
cx q[20],q[13];
cx q[13],q[4];
cx q[32],q[23];
cx q[39],q[32];
swap q[8],q[2];
swap q[44],q[36];
swap q[36],q[28];
swap q[28],q[21];
swap q[21],q[14];
swap q[37],q[23];
cx q[27],q[39];
cx q[4],q[2];
cx q[2],q[14];
swap q[21],q[9];
swap q[37],q[31];
swap q[24],q[3];
cx q[14],q[28];
cx q[28],q[29];
cx q[11],q[27];
cx q[29],q[23];
cx q[23],q[21];
cx q[21],q[35];
cx q[35],q[37];
cx q[37],q[46];
cx q[20],q[11];
cx q[13],q[20];
cx q[4],q[13];
cx q[2],q[4];
swap q[46],q[30];
cx q[14],q[2];
cx q[30],q[24];
cx q[24],q[9];
cx q[28],q[14];
cx q[9],q[8];
cx q[29],q[28];
cx q[23],q[29];
swap q[46],q[39];
swap q[39],q[25];
swap q[25],q[17];
swap q[12],q[6];
swap q[44],q[42];
cx q[21],q[23];
swap q[45],q[44];
swap q[45],q[32];
cx q[8],q[17];
cx q[35],q[21];
cx q[37],q[35];
swap q[20],q[6];
swap q[27],q[20];
swap q[41],q[27];
cx q[30],q[37];
cx q[17],q[19];
cx q[24],q[30];
cx q[19],q[12];
cx q[12],q[26];
swap q[41],q[39];
swap q[21],q[15];
cx q[9],q[24];
swap q[44],q[43];
cx q[26],q[32];
cx q[32],q[39];
cx q[8],q[9];
cx q[17],q[8];
swap q[29],q[21];
swap q[22],q[1];
swap q[47],q[38];
swap q[15],q[0];
cx q[39],q[36];
cx q[36],q[43];
swap q[38],q[31];
cx q[43],q[29];
cx q[19],q[17];
swap q[18],q[15];
cx q[29],q[22];
cx q[12],q[19];
swap q[47],q[34];
swap q[45],q[42];
swap q[45],q[40];
swap q[24],q[16];
cx q[22],q[31];
cx q[26],q[12];
cx q[32],q[26];
cx q[39],q[32];
swap q[44],q[36];
cx q[44],q[39];
cx q[31],q[18];
cx q[18],q[27];
cx q[27],q[34];
cx q[43],q[44];
cx q[29],q[43];
cx q[22],q[29];
swap q[48],q[46];
cx q[34],q[40];
cx q[31],q[22];
cx q[40],q[24];
cx q[18],q[31];
cx q[27],q[18];
cx q[24],q[15];
cx q[15],q[36];
cx q[34],q[27];
cx q[40],q[34];
swap q[46],q[45];
cx q[24],q[40];
cx q[36],q[45];
cx q[15],q[24];
cx q[36],q[15];
cx q[45],q[36];
