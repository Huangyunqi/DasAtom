OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
cz q[1],q[0];
cz q[2],q[1];
cz q[3],q[2];
cz q[4],q[3];
cz q[5],q[4];
