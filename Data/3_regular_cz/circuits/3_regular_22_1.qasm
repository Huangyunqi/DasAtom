OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
cz q[1],q[21];
cz q[1],q[11];
cz q[1],q[7];
cz q[21],q[3];
cz q[21],q[19];
cz q[3],q[15];
cz q[3],q[14];
cz q[15],q[9];
cz q[15],q[13];
cz q[14],q[17];
cz q[14],q[10];
cz q[17],q[5];
cz q[17],q[18];
cz q[0],q[20];
cz q[0],q[13];
cz q[0],q[6];
cz q[20],q[9];
cz q[20],q[6];
cz q[4],q[18];
cz q[4],q[10];
cz q[4],q[16];
cz q[18],q[16];
cz q[2],q[5];
cz q[2],q[9];
cz q[2],q[11];
cz q[5],q[8];
cz q[11],q[12];
cz q[8],q[16];
cz q[8],q[7];
cz q[7],q[10];
cz q[19],q[13];
cz q[19],q[12];
cz q[6],q[12];
