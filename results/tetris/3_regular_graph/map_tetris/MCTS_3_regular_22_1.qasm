OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
swap q[18],q[14];
swap q[19],q[17];
cx q[19],q[17];
swap q[2],q[0];
swap q[8],q[6];
cx q[17],q[7];
cx q[22],q[24];
cx q[7],q[1];
swap q[9],q[4];
swap q[20],q[15];
swap q[10],q[5];
swap q[23],q[21];
cx q[7],q[8];
cx q[8],q[13];
cx q[13],q[9];
swap q[5],q[1];
cx q[17],q[11];
cx q[13],q[12];
swap q[3],q[1];
cx q[19],q[18];
cx q[3],q[9];
swap q[15],q[5];
swap q[12],q[6];
cx q[15],q[21];
swap q[9],q[3];
swap q[3],q[2];
cx q[8],q[12];
cx q[5],q[6];
swap q[5],q[1];
swap q[22],q[20];
swap q[10],q[5];
swap q[12],q[7];
cx q[2],q[0];
cx q[15],q[10];
swap q[19],q[14];
swap q[14],q[9];
swap q[22],q[21];
cx q[24],q[22];
swap q[24],q[22];
swap q[9],q[4];
cx q[20],q[10];
cx q[20],q[21];
cx q[4],q[3];
cx q[14],q[24];
cx q[14],q[18];
cx q[1],q[7];
cx q[1],q[5];
cx q[6],q[5];
cx q[0],q[5];
cx q[11],q[10];
cx q[22],q[21];
swap q[3],q[2];
swap q[13],q[9];
swap q[23],q[21];
cx q[0],q[2];
cx q[2],q[7];
cx q[18],q[13];
cx q[11],q[13];
cx q[23],q[13];
