OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
swap q[10],q[0];
cx q[9],q[13];
cx q[23],q[18];
cx q[12],q[17];
swap q[8],q[4];
cx q[12],q[16];
cx q[16],q[10];
swap q[9],q[3];
swap q[12],q[7];
cx q[3],q[2];
swap q[14],q[9];
swap q[24],q[23];
swap q[23],q[22];
swap q[20],q[10];
cx q[20],q[22];
cx q[24],q[14];
swap q[11],q[1];
cx q[12],q[10];
cx q[10],q[6];
swap q[4],q[3];
cx q[3],q[2];
swap q[22],q[21];
swap q[17],q[13];
cx q[3],q[1];
cx q[1],q[0];
cx q[13],q[8];
cx q[17],q[22];
swap q[24],q[19];
cx q[22],q[24];
cx q[18],q[24];
cx q[21],q[11];
swap q[1],q[0];
cx q[6],q[11];
swap q[8],q[3];
cx q[1],q[3];
cx q[12],q[14];
