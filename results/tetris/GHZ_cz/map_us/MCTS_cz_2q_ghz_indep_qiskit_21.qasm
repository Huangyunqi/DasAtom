OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
cx q[0],q[5];
cx q[5],q[10];
cx q[10],q[15];
cx q[15],q[20];
cx q[20],q[16];
cx q[16],q[6];
cx q[6],q[1];
cx q[1],q[11];
cx q[11],q[21];
cx q[21],q[17];
cx q[17],q[7];
cx q[7],q[2];
cx q[2],q[12];
cx q[12],q[22];
cx q[22],q[18];
cx q[18],q[8];
cx q[8],q[3];
cx q[3],q[13];
cx q[13],q[23];
cx q[23],q[19];
