OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
cx q[10],q[12];
cx q[6],q[12];
cx q[11],q[12];
cx q[16],q[12];
cx q[2],q[12];
cx q[7],q[12];
cx q[17],q[12];
swap q[10],q[0];
cx q[22],q[12];
swap q[6],q[5];
cx q[8],q[12];
cx q[13],q[12];
cx q[18],q[12];
swap q[16],q[15];
cx q[14],q[12];
swap q[12],q[11];
swap q[20],q[15];
cx q[10],q[11];
swap q[3],q[2];
swap q[7],q[2];
cx q[6],q[11];
swap q[13],q[9];
swap q[23],q[22];
swap q[22],q[17];
cx q[16],q[11];
cx q[1],q[11];
cx q[21],q[11];
cx q[15],q[11];
cx q[7],q[11];
cx q[13],q[11];
cx q[17],q[11];
