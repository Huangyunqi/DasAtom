OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[0],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[6],q[13];
cz q[11],q[13];
cz q[1],q[14];
cz q[4],q[15];
cz q[5],q[15];
cz q[2],q[16];
cz q[12],q[17];
cz q[7],q[18];
cz q[17],q[18];
cz q[3],q[19];
cz q[10],q[20];
cz q[19],q[20];
cz q[8],q[21];
cz q[9],q[21];
cz q[14],q[22];
cz q[16],q[22];
