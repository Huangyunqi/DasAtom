OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[5];
cz q[4],q[5];
cz q[4],q[7];
cz q[6],q[7];
cz q[0],q[8];
cz q[8],q[9];
cz q[6],q[11];
cz q[10],q[11];
cz q[9],q[12];
cz q[10],q[12];
cz q[2],q[13];
cz q[3],q[14];
cz q[13],q[14];
