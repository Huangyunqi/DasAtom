OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
cx q[5],q[15];
swap q[10],q[0];
swap q[17],q[13];
swap q[3],q[2];
swap q[7],q[2];
cx q[10],q[15];
cx q[17],q[15];
swap q[11],q[7];
swap q[18],q[14];
cx q[11],q[15];
swap q[18],q[17];
swap q[21],q[15];
swap q[24],q[23];
cx q[17],q[21];
swap q[17],q[13];
cx q[23],q[21];
swap q[13],q[9];
swap q[21],q[11];
cx q[17],q[11];
swap q[8],q[7];
swap q[19],q[14];
swap q[18],q[14];
swap q[12],q[11];
swap q[20],q[15];
cx q[13],q[12];
swap q[16],q[15];
cx q[7],q[12];
cx q[18],q[12];
cx q[16],q[12];
swap q[24],q[14];
swap q[17],q[15];
cx q[14],q[12];
cx q[11],q[12];
swap q[19],q[18];
cx q[8],q[12];
swap q[1],q[0];
swap q[2],q[1];
swap q[1],q[0];
swap q[6],q[1];
cx q[17],q[12];
swap q[20],q[15];
cx q[18],q[12];
swap q[15],q[10];
swap q[1],q[0];
cx q[2],q[12];
cx q[6],q[12];
cx q[10],q[12];
swap q[7],q[1];
cx q[7],q[12];
swap q[10],q[0];
cx q[10],q[12];
