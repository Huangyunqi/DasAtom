OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cz q[2],q[1];
cz q[0],q[2];
cz q[1],q[0];
cz q[1],q[2];
cz q[0],q[2];
cz q[1],q[0];
cz q[1],q[2];
cz q[0],q[2];
cz q[2],q[1];
