OPENQASM 2.0;
include "qelib1.inc";
qreg q[36];
creg c[36];
cx q[5],q[10];
swap q[29],q[22];
swap q[14],q[2];
cx q[10],q[22];
swap q[26],q[14];
cx q[22],q[29];
cx q[29],q[26];
swap q[13],q[7];
cx q[26],q[13];
cx q[13],q[18];
swap q[12],q[1];
swap q[9],q[3];
swap q[15],q[9];
swap q[22],q[17];
swap q[22],q[20];
cx q[18],q[12];
swap q[8],q[1];
swap q[27],q[24];
swap q[7],q[0];
swap q[34],q[21];
swap q[13],q[7];
swap q[33],q[26];
swap q[3],q[1];
swap q[28],q[16];
swap q[32],q[24];
swap q[7],q[2];
cx q[12],q[15];
cx q[15],q[25];
swap q[12],q[7];
cx q[25],q[19];
cx q[19],q[20];
cx q[20],q[8];
swap q[24],q[12];
cx q[8],q[9];
cx q[9],q[27];
cx q[27],q[21];
swap q[25],q[12];
swap q[23],q[15];
cx q[21],q[13];
cx q[13],q[26];
swap q[35],q[29];
swap q[1],q[0];
swap q[2],q[1];
swap q[9],q[2];
cx q[26],q[34];
swap q[17],q[9];
cx q[34],q[22];
cx q[22],q[14];
cx q[14],q[3];
cx q[3],q[16];
cx q[16],q[28];
cx q[28],q[32];
cx q[32],q[24];
cx q[24],q[31];
cx q[31],q[25];
cx q[25],q[15];
cx q[15],q[29];
cx q[29],q[17];
