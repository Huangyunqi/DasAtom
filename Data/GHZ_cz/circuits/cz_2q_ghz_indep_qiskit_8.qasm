OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
cz q[7],q[6];
cz q[6],q[5];
cz q[5],q[4];
cz q[4],q[3];
cz q[3],q[2];
cz q[2],q[1];
cz q[1],q[0];
