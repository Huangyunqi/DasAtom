OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
swap q[18],q[16];
swap q[6],q[5];
cx q[16],q[6];
swap q[17],q[12];
cx q[10],q[6];
cx q[12],q[6];
cx q[11],q[6];
swap q[9],q[3];
swap q[3],q[2];
cx q[1],q[6];
swap q[14],q[9];
cx q[2],q[6];
swap q[9],q[8];
swap q[20],q[10];
cx q[8],q[6];
swap q[13],q[11];
swap q[9],q[3];
cx q[10],q[6];
cx q[0],q[6];
cx q[11],q[6];
cx q[7],q[6];
swap q[8],q[6];
cx q[3],q[8];
swap q[13],q[8];
swap q[21],q[17];
swap q[21],q[11];
cx q[17],q[13];
cx q[23],q[13];
cx q[14],q[13];
swap q[4],q[3];
cx q[9],q[13];
cx q[19],q[13];
swap q[22],q[17];
cx q[11],q[13];
cx q[3],q[13];
cx q[17],q[13];
swap q[24],q[23];
cx q[23],q[13];
cx q[18],q[13];
