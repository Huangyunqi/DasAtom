OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
cz q[3],q[13];
cz q[3],q[6];
cz q[3],q[0];
cz q[13],q[9];
cz q[13],q[6];
cz q[5],q[10];
cz q[5],q[0];
cz q[5],q[4];
cz q[10],q[11];
cz q[10],q[1];
cz q[0],q[7];
cz q[1],q[6];
cz q[1],q[4];
cz q[9],q[11];
cz q[9],q[12];
cz q[11],q[12];
cz q[2],q[8];
cz q[2],q[7];
cz q[2],q[12];
cz q[8],q[4];
cz q[8],q[7];
cz q[6],q[12];
cz q[6],q[0];
cz q[6],q[9];
cz q[12],q[8];
cz q[12],q[5];
cz q[4],q[9];
cz q[4],q[5];
cz q[4],q[13];
cz q[9],q[1];
cz q[0],q[2];
cz q[0],q[5];
cz q[2],q[7];
cz q[2],q[13];
cz q[8],q[3];
cz q[8],q[10];
cz q[1],q[3];
cz q[1],q[13];
cz q[3],q[11];
cz q[7],q[10];
cz q[7],q[11];
cz q[10],q[11];
cz q[4],q[12];
cz q[4],q[7];
cz q[4],q[10];
cz q[12],q[5];
cz q[12],q[9];
cz q[5],q[13];
cz q[5],q[8];
cz q[13],q[3];
cz q[13],q[10];
cz q[3],q[1];
cz q[3],q[11];
cz q[0],q[2];
cz q[0],q[7];
cz q[0],q[6];
cz q[2],q[10];
cz q[2],q[9];
cz q[8],q[9];
cz q[8],q[11];
cz q[1],q[6];
cz q[1],q[7];
cz q[6],q[11];
