OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
cz q[0],q[1];
cz q[1],q[3];
cz q[2],q[3];
cz q[0],q[4];
cz q[2],q[4];
