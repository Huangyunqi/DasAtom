OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
swap q[14],q[12];
swap q[20],q[15];
swap q[12],q[10];
cx q[15],q[10];
swap q[12],q[8];
cx q[12],q[10];
swap q[9],q[7];
swap q[10],q[5];
swap q[18],q[17];
cx q[7],q[5];
swap q[17],q[16];
swap q[10],q[5];
swap q[13],q[7];
cx q[16],q[10];
swap q[7],q[6];
cx q[6],q[10];
swap q[7],q[5];
swap q[18],q[17];
swap q[17],q[16];
cx q[5],q[10];
cx q[16],q[10];
swap q[6],q[1];
swap q[23],q[22];
swap q[22],q[16];
cx q[6],q[10];
swap q[23],q[17];
swap q[11],q[10];
swap q[14],q[13];
swap q[4],q[3];
swap q[8],q[3];
cx q[16],q[11];
cx q[17],q[11];
cx q[7],q[11];
cx q[21],q[11];
cx q[13],q[11];
swap q[8],q[6];
swap q[20],q[16];
cx q[6],q[11];
swap q[18],q[17];
swap q[3],q[2];
cx q[16],q[11];
swap q[7],q[2];
cx q[10],q[11];
cx q[17],q[11];
swap q[14],q[9];
swap q[14],q[13];
cx q[7],q[11];
cx q[13],q[11];
