OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
swap q[46],q[34];
swap q[21],q[8];
swap q[35],q[21];
cx q[40],q[25];
cx q[25],q[20];
cx q[25],q[40];
cx q[20],q[26];
cx q[26],q[32];
swap q[45],q[37];
cx q[32],q[46];
swap q[10],q[4];
swap q[3],q[1];
swap q[17],q[3];
cx q[46],q[43];
cx q[43],q[35];
cx q[20],q[25];
cx q[26],q[20];
cx q[32],q[26];
cx q[35],q[37];
swap q[41],q[40];
cx q[37],q[23];
swap q[40],q[24];
swap q[12],q[11];
swap q[11],q[3];
swap q[21],q[15];
cx q[23],q[10];
cx q[10],q[17];
cx q[46],q[32];
cx q[17],q[24];
cx q[43],q[46];
cx q[35],q[43];
cx q[37],q[35];
cx q[23],q[37];
swap q[3],q[1];
swap q[7],q[2];
swap q[40],q[32];
swap q[32],q[18];
swap q[13],q[6];
swap q[20],q[13];
swap q[45],q[36];
cx q[24],q[15];
cx q[10],q[23];
cx q[15],q[1];
cx q[1],q[2];
cx q[17],q[10];
cx q[2],q[18];
cx q[24],q[17];
cx q[15],q[24];
swap q[48],q[45];
swap q[48],q[34];
swap q[45],q[44];
swap q[13],q[5];
swap q[19],q[13];
cx q[18],q[20];
cx q[1],q[15];
cx q[2],q[1];
swap q[45],q[40];
cx q[20],q[34];
cx q[34],q[39];
cx q[18],q[2];
swap q[24],q[19];
cx q[39],q[48];
cx q[48],q[40];
cx q[20],q[18];
cx q[34],q[20];
swap q[36],q[29];
swap q[4],q[3];
swap q[29],q[22];
swap q[8],q[3];
swap q[28],q[21];
swap q[23],q[21];
swap q[32],q[25];
cx q[39],q[34];
cx q[48],q[39];
swap q[29],q[8];
cx q[40],q[24];
cx q[24],q[22];
cx q[22],q[29];
cx q[29],q[23];
cx q[23],q[25];
cx q[40],q[48];
swap q[10],q[3];
swap q[31],q[18];
swap q[26],q[12];
swap q[47],q[38];
swap q[17],q[16];
cx q[24],q[40];
cx q[22],q[24];
cx q[25],q[9];
cx q[9],q[8];
cx q[29],q[22];
cx q[23],q[29];
swap q[23],q[17];
swap q[44],q[32];
cx q[25],q[17];
swap q[14],q[2];
cx q[8],q[10];
cx q[10],q[18];
swap q[36],q[21];
cx q[18],q[26];
cx q[26],q[38];
swap q[21],q[7];
cx q[38],q[23];
cx q[23],q[32];
swap q[13],q[5];
swap q[5],q[3];
cx q[32],q[11];
swap q[7],q[0];
cx q[9],q[25];
cx q[11],q[2];
cx q[2],q[0];
cx q[8],q[9];
cx q[10],q[8];
swap q[30],q[24];
cx q[18],q[10];
cx q[26],q[18];
swap q[22],q[21];
swap q[30],q[22];
cx q[0],q[3];
cx q[38],q[26];
cx q[3],q[24];
cx q[23],q[38];
cx q[32],q[23];
cx q[11],q[32];
cx q[24],q[30];
cx q[30],q[33];
cx q[2],q[11];
cx q[0],q[2];
cx q[3],q[0];
cx q[24],q[3];
cx q[30],q[24];
cx q[33],q[30];
