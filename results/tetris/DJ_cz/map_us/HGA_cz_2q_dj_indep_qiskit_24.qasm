OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
cx q[10],q[12];
cx q[6],q[12];
cx q[11],q[12];
cx q[16],q[12];
swap q[6],q[5];
cx q[2],q[12];
swap q[5],q[0];
cx q[7],q[12];
cx q[17],q[12];
cx q[22],q[12];
cx q[8],q[12];
cx q[13],q[12];
swap q[10],q[5];
cx q[18],q[12];
swap q[16],q[15];
swap q[2],q[1];
cx q[14],q[12];
swap q[20],q[15];
cx q[6],q[12];
cx q[10],q[12];
cx q[16],q[12];
cx q[2],q[12];
swap q[12],q[10];
swap q[21],q[20];
swap q[23],q[21];
cx q[15],q[10];
cx q[20],q[10];
swap q[16],q[10];
swap q[3],q[1];
swap q[7],q[1];
swap q[12],q[7];
swap q[14],q[4];
swap q[18],q[14];
swap q[19],q[13];
cx q[21],q[16];
swap q[17],q[16];
swap q[9],q[7];
cx q[12],q[17];
cx q[18],q[17];
cx q[13],q[17];
cx q[7],q[17];
