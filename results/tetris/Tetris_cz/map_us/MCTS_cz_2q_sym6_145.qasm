OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
cx q[2],q[0];
cx q[3],q[1];
cx q[4],q[3];
cx q[1],q[4];
cx q[1],q[3];
cx q[4],q[3];
cx q[1],q[4];
cx q[3],q[1];
cx q[1],q[7];
cx q[3],q[4];
cx q[5],q[3];
cx q[4],q[5];
cx q[4],q[3];
cx q[5],q[3];
cx q[4],q[5];
cx q[3],q[4];
cx q[3],q[1];
cx q[7],q[3];
cx q[7],q[1];
cx q[3],q[1];
cx q[7],q[3];
cx q[1],q[7];
cx q[1],q[2];
cx q[0],q[1];
cx q[0],q[2];
cx q[1],q[2];
cx q[0],q[1];
cx q[2],q[0];
cx q[2],q[0];
cx q[1],q[7];
cx q[3],q[1];
cx q[7],q[3];
cx q[7],q[1];
cx q[3],q[1];
cx q[7],q[3];
cx q[1],q[7];
cx q[1],q[7];
cx q[3],q[4];
cx q[5],q[3];
cx q[4],q[5];
cx q[4],q[3];
cx q[5],q[3];
cx q[4],q[5];
cx q[3],q[4];
cx q[3],q[1];
cx q[7],q[3];
cx q[7],q[1];
cx q[3],q[1];
cx q[7],q[3];
cx q[1],q[7];
cx q[1],q[2];
cx q[0],q[1];
cx q[0],q[2];
cx q[1],q[2];
cx q[0],q[1];
cx q[2],q[0];
cx q[2],q[0];
cx q[1],q[7];
cx q[3],q[1];
cx q[7],q[3];
cx q[7],q[1];
cx q[3],q[1];
cx q[7],q[3];
cx q[1],q[7];
cx q[3],q[1];
cx q[4],q[3];
cx q[1],q[4];
cx q[1],q[3];
cx q[4],q[3];
cx q[1],q[4];
cx q[3],q[1];
cx q[1],q[7];
cx q[3],q[4];
cx q[5],q[3];
cx q[4],q[5];
cx q[4],q[3];
cx q[5],q[3];
cx q[4],q[5];
cx q[3],q[4];
cx q[3],q[1];
cx q[7],q[3];
cx q[7],q[1];
cx q[3],q[1];
cx q[7],q[3];
cx q[1],q[7];
cx q[1],q[2];
cx q[0],q[1];
cx q[0],q[2];
cx q[1],q[2];
cx q[0],q[1];
cx q[2],q[0];
cx q[2],q[0];
cx q[1],q[7];
cx q[3],q[1];
cx q[7],q[3];
cx q[7],q[1];
cx q[3],q[1];
cx q[7],q[3];
cx q[1],q[7];
cx q[1],q[7];
cx q[3],q[4];
cx q[5],q[3];
cx q[4],q[5];
cx q[4],q[3];
cx q[5],q[3];
cx q[4],q[5];
cx q[3],q[4];
cx q[3],q[1];
cx q[7],q[3];
cx q[7],q[1];
cx q[3],q[1];
cx q[7],q[3];
cx q[1],q[7];
cx q[1],q[2];
cx q[0],q[1];
cx q[0],q[2];
cx q[1],q[2];
cx q[0],q[1];
cx q[2],q[0];
cx q[2],q[0];
cx q[1],q[7];
cx q[3],q[1];
cx q[7],q[3];
cx q[7],q[1];
cx q[3],q[1];
cx q[7],q[3];
cx q[1],q[7];
cx q[3],q[4];
cx q[1],q[3];
cx q[4],q[1];
cx q[4],q[3];
cx q[1],q[3];
cx q[4],q[1];
cx q[3],q[4];
cx q[4],q[7];
cx q[3],q[1];
cx q[5],q[3];
cx q[1],q[5];
cx q[1],q[3];
cx q[5],q[3];
cx q[1],q[5];
cx q[3],q[1];
cx q[3],q[4];
cx q[7],q[3];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[2];
cx q[0],q[4];
cx q[0],q[2];
cx q[4],q[2];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[0];
cx q[4],q[7];
cx q[3],q[4];
cx q[7],q[3];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[7];
cx q[3],q[1];
cx q[5],q[3];
cx q[1],q[5];
cx q[1],q[3];
cx q[5],q[3];
cx q[1],q[5];
cx q[3],q[1];
cx q[3],q[4];
cx q[7],q[3];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[2];
cx q[0],q[4];
cx q[0],q[2];
cx q[4],q[2];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[0];
cx q[4],q[7];
cx q[3],q[4];
cx q[7],q[3];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[3],q[4];
cx q[1],q[3];
cx q[4],q[1];
cx q[4],q[3];
cx q[1],q[3];
cx q[4],q[1];
cx q[3],q[4];
cx q[4],q[7];
cx q[3],q[1];
cx q[5],q[3];
cx q[1],q[5];
cx q[1],q[3];
cx q[5],q[3];
cx q[1],q[5];
cx q[3],q[1];
cx q[3],q[4];
cx q[7],q[3];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[2];
cx q[0],q[4];
cx q[0],q[2];
cx q[4],q[2];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[0];
cx q[4],q[7];
cx q[3],q[4];
cx q[7],q[3];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[7];
cx q[3],q[1];
cx q[5],q[3];
cx q[1],q[5];
cx q[1],q[3];
cx q[5],q[3];
cx q[1],q[5];
cx q[3],q[1];
cx q[3],q[4];
cx q[7],q[3];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[2];
cx q[0],q[4];
cx q[0],q[2];
cx q[4],q[2];
cx q[0],q[4];
cx q[2],q[0];
cx q[4],q[7];
cx q[3],q[4];
cx q[7],q[3];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[2];
cx q[1],q[7];
cx q[5],q[1];
cx q[0],q[3];
cx q[7],q[5];
cx q[7],q[1];
cx q[5],q[1];
cx q[7],q[5];
cx q[1],q[7];
cx q[1],q[4];
cx q[2],q[1];
cx q[2],q[4];
cx q[1],q[4];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[0];
cx q[3],q[4];
cx q[3],q[0];
cx q[4],q[0];
cx q[3],q[4];
cx q[4],q[2];
cx q[1],q[4];
cx q[2],q[1];
cx q[2],q[4];
cx q[1],q[4];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[2];
cx q[1],q[7];
cx q[5],q[1];
cx q[0],q[3];
cx q[7],q[5];
cx q[7],q[1];
cx q[5],q[1];
cx q[0],q[3];
cx q[7],q[5];
cx q[1],q[7];
cx q[1],q[4];
cx q[2],q[1];
cx q[2],q[4];
cx q[1],q[4];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[0];
cx q[3],q[4];
cx q[3],q[0];
cx q[4],q[0];
cx q[3],q[4];
cx q[4],q[2];
cx q[1],q[4];
cx q[0],q[3];
cx q[2],q[1];
cx q[2],q[4];
cx q[1],q[4];
cx q[2],q[1];
cx q[4],q[2];
cx q[1],q[4];
cx q[3],q[1];
cx q[4],q[3];
cx q[4],q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[1],q[4];
cx q[4],q[7];
cx q[1],q[3];
cx q[2],q[0];
cx q[5],q[1];
cx q[3],q[5];
cx q[3],q[1];
cx q[5],q[1];
cx q[3],q[5];
cx q[1],q[3];
cx q[1],q[4];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[2];
cx q[0],q[4];
cx q[0],q[2];
cx q[4],q[2];
cx q[0],q[4];
cx q[4],q[7];
cx q[1],q[4];
cx q[2],q[0];
cx q[2],q[0];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[7];
cx q[1],q[3];
cx q[5],q[1];
cx q[3],q[5];
cx q[3],q[1];
cx q[5],q[1];
cx q[3],q[5];
cx q[1],q[3];
cx q[1],q[4];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[2];
cx q[0],q[4];
cx q[0],q[2];
cx q[4],q[2];
cx q[0],q[4];
cx q[4],q[7];
cx q[1],q[4];
cx q[2],q[0];
cx q[2],q[0];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[1],q[4];
cx q[3],q[1];
cx q[4],q[3];
cx q[4],q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[1],q[4];
cx q[4],q[7];
cx q[1],q[3];
cx q[5],q[1];
cx q[3],q[5];
cx q[3],q[1];
cx q[5],q[1];
cx q[3],q[5];
cx q[1],q[3];
cx q[1],q[4];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[2];
cx q[0],q[4];
cx q[0],q[2];
cx q[4],q[2];
cx q[0],q[4];
cx q[4],q[7];
cx q[1],q[4];
cx q[2],q[0];
cx q[2],q[0];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[7];
cx q[1],q[3];
cx q[5],q[1];
cx q[3],q[5];
cx q[3],q[1];
cx q[5],q[1];
cx q[3],q[5];
cx q[1],q[3];
cx q[1],q[4];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[2];
cx q[0],q[4];
cx q[0],q[2];
cx q[4],q[2];
cx q[0],q[4];
cx q[4],q[7];
cx q[1],q[4];
swap q[5],q[3];
cx q[2],q[0];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[2];
cx q[0],q[1];
cx q[5],q[7];
cx q[3],q[5];
cx q[7],q[3];
cx q[7],q[5];
cx q[3],q[5];
cx q[7],q[3];
cx q[5],q[7];
cx q[5],q[4];
cx q[2],q[5];
cx q[2],q[4];
cx q[5],q[4];
cx q[2],q[5];
cx q[4],q[2];
cx q[4],q[0];
cx q[1],q[4];
cx q[1],q[0];
cx q[4],q[0];
cx q[1],q[4];
cx q[4],q[2];
cx q[5],q[4];
cx q[2],q[5];
cx q[2],q[4];
cx q[5],q[4];
cx q[2],q[5];
cx q[0],q[1];
cx q[5],q[7];
cx q[4],q[2];
cx q[4],q[2];
cx q[3],q[5];
cx q[7],q[3];
cx q[0],q[1];
cx q[7],q[5];
cx q[3],q[5];
cx q[7],q[3];
cx q[5],q[7];
cx q[5],q[4];
cx q[2],q[5];
cx q[2],q[4];
cx q[5],q[4];
cx q[2],q[5];
cx q[4],q[2];
cx q[4],q[0];
cx q[1],q[4];
cx q[1],q[0];
cx q[4],q[0];
cx q[1],q[4];
cx q[4],q[2];
cx q[5],q[4];
cx q[2],q[5];
cx q[2],q[4];
cx q[5],q[4];
cx q[2],q[5];
cx q[4],q[2];
cx q[5],q[7];
cx q[0],q[1];
cx q[0],q[4];
cx q[3],q[5];
cx q[1],q[2];
cx q[7],q[3];
cx q[7],q[5];
cx q[3],q[5];
cx q[7],q[3];
cx q[5],q[7];
cx q[5],q[1];
cx q[2],q[5];
cx q[2],q[1];
cx q[5],q[1];
cx q[2],q[5];
cx q[1],q[2];
cx q[1],q[0];
cx q[4],q[1];
cx q[4],q[0];
cx q[1],q[0];
cx q[4],q[1];
cx q[1],q[2];
cx q[5],q[1];
cx q[0],q[4];
cx q[0],q[4];
cx q[2],q[5];
cx q[2],q[1];
cx q[5],q[1];
cx q[2],q[5];
cx q[5],q[7];
cx q[3],q[5];
cx q[1],q[2];
cx q[7],q[3];
cx q[7],q[5];
cx q[3],q[5];
cx q[7],q[3];
cx q[1],q[2];
cx q[5],q[7];
cx q[5],q[1];
cx q[2],q[5];
cx q[2],q[1];
cx q[5],q[1];
cx q[2],q[5];
cx q[1],q[2];
cx q[1],q[0];
cx q[4],q[1];
cx q[4],q[0];
cx q[1],q[0];
cx q[4],q[1];
cx q[0],q[4];
cx q[1],q[2];
cx q[5],q[1];
cx q[2],q[5];
cx q[2],q[1];
cx q[5],q[1];
cx q[2],q[5];
cx q[1],q[2];
cx q[2],q[5];
cx q[1],q[4];
cx q[0],q[1];
cx q[4],q[0];
cx q[4],q[1];
cx q[0],q[1];
cx q[4],q[0];
cx q[1],q[4];
cx q[4],q[7];
cx q[1],q[0];
cx q[3],q[1];
cx q[0],q[3];
cx q[0],q[1];
cx q[3],q[1];
cx q[0],q[3];
cx q[1],q[0];
cx q[1],q[4];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[2];
cx q[5],q[4];
cx q[5],q[2];
cx q[4],q[2];
cx q[5],q[4];
cx q[4],q[7];
cx q[1],q[4];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[7];
cx q[1],q[0];
cx q[3],q[1];
cx q[0],q[3];
cx q[2],q[5];
cx q[0],q[1];
cx q[3],q[1];
cx q[0],q[3];
cx q[2],q[5];
cx q[1],q[0];
cx q[1],q[4];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[2];
cx q[5],q[4];
cx q[5],q[2];
cx q[4],q[2];
cx q[5],q[4];
cx q[4],q[7];
cx q[1],q[4];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[1],q[4];
cx q[0],q[1];
cx q[4],q[0];
cx q[4],q[1];
cx q[0],q[1];
cx q[4],q[0];
cx q[1],q[4];
cx q[4],q[7];
cx q[1],q[0];
cx q[3],q[1];
cx q[0],q[3];
cx q[2],q[5];
cx q[0],q[1];
cx q[3],q[1];
cx q[2],q[5];
cx q[0],q[3];
cx q[1],q[0];
cx q[1],q[4];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[2];
cx q[5],q[4];
cx q[5],q[2];
cx q[4],q[2];
cx q[5],q[4];
cx q[4],q[7];
cx q[1],q[4];
cx q[7],q[1];
cx q[2],q[5];
cx q[2],q[5];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[7];
cx q[1],q[0];
cx q[3],q[1];
cx q[0],q[3];
cx q[0],q[1];
cx q[3],q[1];
cx q[0],q[3];
cx q[1],q[0];
cx q[1],q[4];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
cx q[4],q[2];
cx q[5],q[4];
swap q[6],q[0];
cx q[5],q[2];
cx q[4],q[2];
cx q[5],q[4];
cx q[4],q[7];
cx q[1],q[4];
cx q[2],q[5];
cx q[7],q[1];
cx q[7],q[4];
cx q[1],q[4];
cx q[7],q[1];
cx q[4],q[7];
swap q[2],q[0];
cx q[5],q[1];
cx q[6],q[7];
cx q[3],q[6];
cx q[4],q[0];
cx q[7],q[3];
cx q[7],q[6];
cx q[3],q[6];
cx q[7],q[3];
cx q[6],q[7];
cx q[6],q[4];
cx q[0],q[6];
cx q[0],q[4];
cx q[6],q[4];
cx q[0],q[6];
cx q[4],q[0];
cx q[4],q[5];
cx q[1],q[4];
cx q[1],q[5];
cx q[4],q[5];
cx q[1],q[4];
cx q[4],q[0];
cx q[6],q[4];
cx q[0],q[6];
cx q[0],q[4];
cx q[6],q[4];
cx q[0],q[6];
cx q[4],q[0];
cx q[4],q[0];
cx q[5],q[1];
cx q[5],q[1];
cx q[6],q[7];
cx q[3],q[6];
cx q[7],q[3];
cx q[7],q[6];
cx q[3],q[6];
cx q[7],q[3];
cx q[6],q[7];
cx q[6],q[4];
cx q[0],q[6];
cx q[0],q[4];
cx q[6],q[4];
cx q[0],q[6];
cx q[4],q[0];
cx q[4],q[5];
cx q[1],q[4];
cx q[1],q[5];
cx q[4],q[5];
cx q[1],q[4];
cx q[5],q[1];
cx q[4],q[0];
cx q[6],q[4];
cx q[0],q[6];
cx q[0],q[4];
cx q[6],q[4];
cx q[0],q[6];
cx q[4],q[0];
cx q[1],q[0];
cx q[6],q[7];
cx q[3],q[6];
cx q[7],q[3];
cx q[5],q[4];
cx q[7],q[6];
cx q[3],q[6];
cx q[7],q[3];
cx q[6],q[7];
swap q[4],q[1];
cx q[6],q[4];
cx q[0],q[6];
cx q[0],q[4];
cx q[6],q[4];
cx q[0],q[6];
cx q[4],q[0];
cx q[4],q[5];
cx q[1],q[4];
cx q[1],q[5];
cx q[4],q[5];
cx q[1],q[4];
cx q[4],q[0];
cx q[6],q[4];
cx q[0],q[6];
cx q[0],q[4];
cx q[6],q[4];
cx q[0],q[6];
swap q[5],q[1];
cx q[6],q[7];
cx q[3],q[6];
cx q[4],q[0];
cx q[4],q[0];
cx q[7],q[3];
cx q[1],q[5];
cx q[1],q[5];
cx q[7],q[6];
cx q[3],q[6];
cx q[7],q[3];
cx q[6],q[7];
cx q[6],q[4];
cx q[0],q[6];
cx q[0],q[4];
cx q[6],q[4];
cx q[0],q[6];
cx q[4],q[0];
cx q[4],q[1];
cx q[5],q[4];
cx q[5],q[1];
cx q[4],q[1];
cx q[5],q[4];
cx q[1],q[5];
cx q[4],q[0];
cx q[6],q[4];
cx q[0],q[6];
cx q[0],q[4];
cx q[6],q[4];
cx q[0],q[6];
cx q[4],q[0];
cx q[1],q[0];
cx q[6],q[7];
cx q[4],q[5];
cx q[3],q[6];
cx q[7],q[3];
cx q[7],q[6];
swap q[4],q[1];
cx q[3],q[6];
cx q[7],q[3];
cx q[6],q[7];
cx q[6],q[4];
cx q[0],q[6];
cx q[0],q[4];
cx q[6],q[4];
cx q[0],q[6];
cx q[4],q[0];
cx q[4],q[1];
cx q[5],q[4];
cx q[5],q[1];
cx q[4],q[1];
cx q[5],q[4];
cx q[4],q[0];
cx q[6],q[4];
cx q[0],q[6];
cx q[0],q[4];
cx q[6],q[4];
cx q[0],q[6];
cx q[6],q[7];
cx q[1],q[5];
cx q[4],q[0];
cx q[4],q[0];
cx q[1],q[5];
cx q[3],q[6];
cx q[7],q[3];
cx q[7],q[6];
cx q[3],q[6];
cx q[7],q[3];
swap q[3],q[0];
cx q[6],q[7];
cx q[6],q[4];
cx q[3],q[6];
cx q[3],q[4];
swap q[7],q[1];
cx q[6],q[4];
cx q[3],q[6];
cx q[4],q[3];
cx q[4],q[7];
cx q[5],q[4];
cx q[5],q[7];
cx q[4],q[7];
cx q[5],q[4];
cx q[4],q[3];
cx q[6],q[4];
cx q[7],q[5];
cx q[7],q[5];
cx q[3],q[6];
cx q[3],q[4];
cx q[6],q[4];
cx q[3],q[6];
cx q[4],q[3];
cx q[1],q[3];
cx q[0],q[1];
cx q[3],q[0];
cx q[3],q[1];
cx q[0],q[1];
cx q[3],q[0];
cx q[1],q[3];
cx q[6],q[4];
cx q[3],q[7];
cx q[5],q[3];
swap q[6],q[0];
swap q[2],q[0];
cx q[5],q[7];
cx q[3],q[7];
cx q[5],q[3];
cx q[7],q[5];
cx q[7],q[3];
cx q[5],q[1];
cx q[6],q[7];
cx q[3],q[6];
cx q[3],q[7];
cx q[6],q[7];
cx q[3],q[6];
cx q[7],q[3];
cx q[7],q[5];
cx q[1],q[7];
cx q[1],q[5];
cx q[7],q[5];
cx q[1],q[7];
cx q[5],q[1];
cx q[5],q[2];
cx q[4],q[5];
cx q[4],q[2];
cx q[5],q[2];
cx q[4],q[5];
cx q[2],q[4];
cx q[2],q[4];
cx q[5],q[1];
cx q[7],q[5];
cx q[1],q[7];
cx q[1],q[5];
cx q[7],q[5];
cx q[1],q[7];
cx q[7],q[3];
cx q[5],q[1];
cx q[5],q[1];
cx q[6],q[7];
cx q[3],q[6];
cx q[3],q[7];
cx q[6],q[7];
cx q[3],q[6];
cx q[7],q[3];
cx q[7],q[5];
cx q[1],q[7];
cx q[1],q[5];
cx q[7],q[5];
cx q[1],q[7];
cx q[5],q[1];
cx q[5],q[2];
cx q[4],q[5];
cx q[4],q[2];
cx q[5],q[2];
cx q[4],q[5];
cx q[2],q[4];
cx q[2],q[4];
cx q[5],q[1];
cx q[7],q[5];
cx q[1],q[7];
cx q[1],q[5];
cx q[7],q[5];
cx q[1],q[7];
cx q[5],q[1];
cx q[7],q[5];
cx q[3],q[7];
cx q[5],q[3];
cx q[5],q[7];
cx q[3],q[7];
cx q[5],q[3];
cx q[7],q[5];
cx q[7],q[3];
cx q[5],q[1];
cx q[6],q[7];
cx q[3],q[6];
cx q[3],q[7];
cx q[6],q[7];
cx q[3],q[6];
cx q[7],q[3];
cx q[7],q[5];
cx q[1],q[7];
cx q[1],q[5];
cx q[7],q[5];
cx q[1],q[7];
cx q[5],q[1];
cx q[5],q[2];
cx q[4],q[5];
cx q[4],q[2];
cx q[5],q[2];
cx q[4],q[5];
cx q[2],q[4];
cx q[2],q[4];
cx q[5],q[1];
cx q[7],q[5];
cx q[1],q[7];
cx q[1],q[5];
cx q[7],q[5];
cx q[1],q[7];
cx q[7],q[3];
cx q[6],q[7];
cx q[3],q[6];
cx q[5],q[1];
cx q[5],q[1];
cx q[3],q[7];
cx q[6],q[7];
swap q[6],q[0];
cx q[3],q[0];
cx q[7],q[3];
cx q[7],q[5];
cx q[1],q[7];
cx q[1],q[5];
cx q[7],q[5];
cx q[1],q[7];
cx q[5],q[1];
cx q[5],q[2];
cx q[4],q[5];
cx q[4],q[2];
cx q[5],q[2];
cx q[4],q[5];
cx q[2],q[4];
cx q[5],q[1];
cx q[7],q[5];
cx q[1],q[7];
cx q[1],q[5];
cx q[7],q[5];
cx q[1],q[7];
cx q[5],q[1];
cx q[3],q[1];
swap q[3],q[1];
cx q[4],q[7];
cx q[0],q[1];
cx q[5],q[2];
cx q[3],q[0];
cx q[3],q[1];
cx q[0],q[1];
cx q[3],q[0];
cx q[1],q[3];
cx q[1],q[5];
cx q[2],q[1];
cx q[2],q[5];
cx q[1],q[5];
cx q[2],q[1];
cx q[5],q[2];
cx q[5],q[4];
cx q[7],q[5];
cx q[7],q[4];
cx q[5],q[4];
cx q[7],q[5];
cx q[5],q[2];
cx q[4],q[7];
cx q[4],q[7];
cx q[1],q[5];
cx q[2],q[1];
cx q[2],q[5];
cx q[1],q[5];
cx q[2],q[1];
cx q[1],q[3];
cx q[0],q[1];
cx q[3],q[0];
cx q[5],q[2];
cx q[3],q[1];
cx q[0],q[1];
cx q[5],q[2];
cx q[3],q[0];
cx q[1],q[3];
cx q[1],q[5];
cx q[2],q[1];
cx q[2],q[5];
cx q[1],q[5];
cx q[2],q[1];
cx q[5],q[2];
cx q[5],q[4];
cx q[7],q[5];
cx q[7],q[4];
cx q[5],q[4];
cx q[7],q[5];
cx q[5],q[2];
cx q[1],q[5];
cx q[4],q[7];
cx q[2],q[1];
cx q[2],q[5];
cx q[1],q[5];
swap q[8],q[7];
cx q[2],q[1];
cx q[1],q[3];
cx q[0],q[1];
cx q[3],q[0];
cx q[5],q[2];
cx q[8],q[2];
cx q[4],q[5];
cx q[3],q[1];
swap q[8],q[5];
cx q[0],q[1];
cx q[3],q[0];
cx q[1],q[3];
cx q[1],q[5];
cx q[2],q[1];
cx q[2],q[5];
cx q[1],q[5];
cx q[2],q[1];
cx q[5],q[2];
cx q[5],q[4];
cx q[8],q[5];
cx q[8],q[4];
cx q[5],q[4];
cx q[8],q[5];
cx q[5],q[2];
cx q[4],q[8];
cx q[4],q[8];
cx q[1],q[5];
cx q[2],q[1];
cx q[2],q[5];
cx q[1],q[5];
cx q[2],q[1];
cx q[1],q[3];
cx q[0],q[1];
cx q[5],q[2];
cx q[3],q[0];
cx q[3],q[1];
cx q[0],q[1];
cx q[5],q[2];
cx q[3],q[0];
cx q[1],q[3];
cx q[1],q[5];
cx q[2],q[1];
cx q[2],q[5];
cx q[1],q[5];
cx q[2],q[1];
cx q[5],q[2];
cx q[5],q[4];
cx q[8],q[5];
cx q[8],q[4];
cx q[5],q[4];
cx q[8],q[5];
cx q[4],q[8];
cx q[5],q[2];
cx q[1],q[5];
cx q[2],q[1];
cx q[2],q[5];
cx q[1],q[5];
cx q[2],q[1];
cx q[1],q[3];
cx q[0],q[1];
cx q[5],q[2];
cx q[5],q[8];
cx q[3],q[0];
cx q[3],q[1];
cx q[4],q[2];
cx q[0],q[1];
cx q[3],q[0];
cx q[1],q[3];
cx q[1],q[4];
cx q[2],q[1];
cx q[2],q[4];
cx q[1],q[4];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[5];
cx q[8],q[4];
cx q[8],q[5];
cx q[4],q[5];
cx q[8],q[4];
cx q[4],q[2];
cx q[1],q[4];
cx q[2],q[1];
cx q[5],q[8];
cx q[5],q[8];
cx q[2],q[4];
cx q[1],q[4];
cx q[2],q[1];
cx q[1],q[3];
cx q[4],q[2];
cx q[4],q[2];
cx q[0],q[1];
cx q[3],q[0];
cx q[3],q[1];
cx q[0],q[1];
swap q[3],q[0];
cx q[0],q[3];
cx q[1],q[0];
cx q[1],q[4];
cx q[2],q[1];
cx q[2],q[4];
cx q[1],q[4];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[5];
cx q[8],q[4];
cx q[8],q[5];
cx q[4],q[5];
cx q[8],q[4];
swap q[4],q[2];
cx q[2],q[4];
cx q[1],q[2];
cx q[4],q[1];
cx q[4],q[2];
cx q[1],q[2];
cx q[4],q[1];
cx q[2],q[4];
cx q[0],q[4];
cx q[5],q[8];
cx q[3],q[0];
cx q[4],q[3];
cx q[4],q[0];
cx q[5],q[8];
cx q[3],q[0];
cx q[4],q[3];
cx q[0],q[4];
cx q[4],q[2];
cx q[1],q[0];
cx q[3],q[1];
cx q[0],q[3];
cx q[0],q[1];
cx q[3],q[1];
cx q[0],q[3];
cx q[1],q[0];
cx q[1],q[4];
cx q[2],q[1];
cx q[2],q[4];
cx q[1],q[4];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[5];
cx q[8],q[4];
cx q[8],q[5];
cx q[4],q[5];
cx q[8],q[4];
cx q[4],q[2];
cx q[1],q[4];
cx q[2],q[1];
cx q[2],q[4];
cx q[1],q[4];
cx q[2],q[1];
cx q[5],q[8];
cx q[5],q[8];
cx q[1],q[0];
cx q[3],q[1];
cx q[4],q[2];
cx q[4],q[2];
cx q[0],q[3];
cx q[0],q[1];
cx q[3],q[1];
cx q[0],q[3];
cx q[1],q[0];
cx q[1],q[4];
cx q[2],q[1];
cx q[2],q[4];
cx q[1],q[4];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[5];
cx q[8],q[4];
cx q[8],q[5];
cx q[4],q[5];
cx q[8],q[4];
cx q[5],q[8];
cx q[4],q[2];
cx q[1],q[4];
cx q[2],q[1];
cx q[2],q[4];
cx q[1],q[4];
cx q[2],q[1];
swap q[3],q[1];
cx q[4],q[2];
cx q[0],q[2];
cx q[1],q[0];
swap q[5],q[4];
cx q[2],q[1];
swap q[8],q[7];
cx q[2],q[0];
cx q[1],q[0];
swap q[7],q[3];
cx q[2],q[1];
cx q[0],q[2];
cx q[0],q[4];
cx q[1],q[0];
cx q[4],q[1];
cx q[4],q[0];
cx q[1],q[0];
cx q[4],q[1];
cx q[0],q[4];
cx q[5],q[2];
cx q[0],q[3];
cx q[1],q[0];
cx q[3],q[1];
cx q[3],q[0];
cx q[1],q[0];
cx q[3],q[1];
cx q[0],q[3];
cx q[4],q[3];
cx q[0],q[4];
cx q[3],q[0];
cx q[3],q[4];
cx q[0],q[4];
cx q[3],q[0];
cx q[4],q[3];
cx q[4],q[0];
cx q[3],q[7];
cx q[1],q[4];
cx q[0],q[1];
cx q[0],q[4];
cx q[1],q[4];
cx q[0],q[1];
cx q[4],q[0];
cx q[4],q[3];
cx q[7],q[4];
cx q[7],q[3];
cx q[4],q[3];
cx q[7],q[4];
swap q[4],q[3];
cx q[4],q[7];
cx q[4],q[5];
cx q[2],q[4];
cx q[2],q[5];
cx q[4],q[5];
cx q[2],q[4];
cx q[4],q[7];
cx q[3],q[4];
cx q[7],q[3];
cx q[5],q[2];
cx q[7],q[4];
cx q[3],q[4];
cx q[5],q[2];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[7];
cx q[3],q[0];
cx q[1],q[3];
cx q[0],q[1];
cx q[0],q[3];
cx q[1],q[3];
cx q[0],q[1];
cx q[3],q[0];
cx q[3],q[4];
cx q[7],q[3];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[5];
cx q[2],q[4];
cx q[2],q[5];
cx q[4],q[5];
cx q[2],q[4];
cx q[4],q[7];
cx q[3],q[4];
cx q[7],q[3];
cx q[5],q[2];
cx q[7],q[4];
cx q[3],q[4];
cx q[5],q[2];
cx q[7],q[3];
cx q[4],q[7];
cx q[3],q[4];
cx q[0],q[3];
cx q[4],q[0];
cx q[4],q[3];
cx q[0],q[3];
cx q[4],q[0];
cx q[3],q[4];
cx q[4],q[7];
cx q[3],q[0];
cx q[1],q[3];
cx q[0],q[1];
cx q[0],q[3];
cx q[1],q[3];
cx q[0],q[1];
cx q[3],q[0];
cx q[3],q[4];
cx q[7],q[3];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[5];
cx q[2],q[4];
cx q[2],q[5];
cx q[4],q[5];
cx q[2],q[4];
cx q[4],q[7];
cx q[3],q[4];
cx q[5],q[2];
cx q[7],q[3];
cx q[5],q[2];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[7];
cx q[3],q[0];
cx q[1],q[3];
cx q[0],q[1];
cx q[0],q[3];
cx q[1],q[3];
cx q[0],q[1];
cx q[3],q[0];
cx q[3],q[4];
cx q[7],q[3];
cx q[7],q[4];
cx q[3],q[4];
cx q[7],q[3];
cx q[4],q[7];
cx q[4],q[5];
cx q[2],q[4];
swap q[7],q[3];
cx q[2],q[5];
swap q[8],q[2];
cx q[4],q[5];
cx q[8],q[4];
cx q[5],q[8];
cx q[4],q[3];
cx q[7],q[4];
cx q[3],q[7];
cx q[3],q[4];
cx q[7],q[4];
cx q[3],q[7];
cx q[4],q[3];
cx q[4],q[5];
swap q[5],q[2];
cx q[8],q[7];
cx q[0],q[3];
cx q[1],q[0];
cx q[3],q[1];
cx q[3],q[0];
cx q[1],q[0];
swap q[7],q[5];
cx q[3],q[1];
cx q[0],q[3];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[4];
cx q[0],q[4];
cx q[2],q[0];
cx q[4],q[2];
cx q[4],q[8];
cx q[5],q[4];
cx q[5],q[8];
cx q[4],q[8];
cx q[5],q[4];
cx q[4],q[2];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[4];
cx q[0],q[4];
cx q[2],q[0];
cx q[4],q[2];
cx q[4],q[2];
cx q[8],q[5];
cx q[8],q[5];
cx q[0],q[3];
cx q[1],q[0];
cx q[3],q[1];
cx q[3],q[0];
cx q[1],q[0];
cx q[3],q[1];
cx q[0],q[3];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[4];
cx q[0],q[4];
cx q[2],q[0];
cx q[4],q[2];
cx q[4],q[8];
cx q[5],q[4];
cx q[5],q[8];
cx q[4],q[8];
cx q[5],q[4];
cx q[8],q[5];
cx q[4],q[2];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[4];
cx q[0],q[4];
cx q[2],q[0];
cx q[4],q[2];
cx q[5],q[2];
cx q[0],q[3];
swap q[5],q[4];
cx q[8],q[5];
cx q[1],q[0];
cx q[3],q[1];
cx q[3],q[0];
cx q[1],q[0];
cx q[3],q[1];
cx q[0],q[3];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[4];
cx q[0],q[4];
cx q[2],q[0];
cx q[4],q[2];
cx q[4],q[8];
cx q[5],q[4];
cx q[5],q[8];
cx q[4],q[8];
cx q[5],q[4];
cx q[4],q[2];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[4];
cx q[0],q[4];
cx q[2],q[0];
cx q[8],q[5];
cx q[8],q[5];
cx q[0],q[3];
cx q[1],q[0];
cx q[4],q[2];
cx q[4],q[2];
cx q[3],q[1];
cx q[3],q[0];
cx q[1],q[0];
cx q[3],q[1];
cx q[0],q[3];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[4];
cx q[0],q[4];
cx q[2],q[0];
cx q[4],q[2];
cx q[4],q[8];
cx q[5],q[4];
cx q[5],q[8];
cx q[4],q[8];
cx q[5],q[4];
cx q[8],q[5];
cx q[4],q[2];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[4];
cx q[0],q[4];
cx q[2],q[0];
cx q[4],q[2];
cx q[8],q[2];
cx q[0],q[3];
swap q[8],q[4];
swap q[3],q[1];
cx q[8],q[5];
cx q[3],q[0];
cx q[1],q[3];
cx q[1],q[0];
cx q[3],q[0];
cx q[1],q[3];
cx q[0],q[1];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[4];
cx q[0],q[4];
cx q[2],q[0];
cx q[4],q[2];
cx q[4],q[8];
cx q[5],q[4];
cx q[5],q[8];
cx q[4],q[8];
cx q[5],q[4];
cx q[4],q[2];
cx q[0],q[4];
cx q[2],q[0];
cx q[2],q[4];
cx q[0],q[4];
cx q[2],q[0];
cx q[0],q[1];
cx q[8],q[5];
cx q[3],q[0];
cx q[1],q[3];
cx q[8],q[5];
cx q[4],q[2];
cx q[4],q[2];
cx q[1],q[0];
cx q[3],q[0];
cx q[1],q[3];
cx q[0],q[1];
cx q[0],q[4];
cx q[2],q[0];
swap q[2],q[0];
cx q[0],q[4];
cx q[2],q[4];
cx q[0],q[2];
cx q[4],q[0];
cx q[4],q[8];
cx q[5],q[4];
cx q[5],q[8];
cx q[4],q[8];
cx q[5],q[4];
cx q[4],q[0];
cx q[2],q[4];
cx q[0],q[2];
cx q[0],q[4];
cx q[2],q[4];
cx q[0],q[2];
cx q[4],q[0];
cx q[1],q[0];
cx q[8],q[5];
cx q[8],q[5];
cx q[3],q[1];
cx q[0],q[3];
cx q[0],q[1];
cx q[3],q[1];
cx q[0],q[3];
swap q[2],q[0];
cx q[1],q[2];
cx q[2],q[4];
cx q[0],q[1];
cx q[3],q[0];
cx q[1],q[3];
cx q[1],q[0];
cx q[3],q[0];
cx q[1],q[3];
cx q[0],q[1];
cx q[0],q[2];
cx q[4],q[0];
cx q[4],q[2];
cx q[0],q[2];
cx q[4],q[0];
cx q[2],q[4];
cx q[2],q[8];
cx q[5],q[2];
cx q[5],q[8];
cx q[2],q[8];
cx q[5],q[2];
cx q[2],q[4];
cx q[0],q[2];
cx q[4],q[0];
cx q[4],q[2];
cx q[0],q[2];
cx q[4],q[0];
cx q[8],q[5];
cx q[0],q[1];
cx q[3],q[0];
cx q[8],q[5];
cx q[2],q[4];
cx q[2],q[4];
cx q[1],q[3];
cx q[1],q[0];
cx q[3],q[0];
cx q[1],q[3];
cx q[0],q[1];
cx q[0],q[2];
cx q[4],q[0];
cx q[4],q[2];
cx q[0],q[2];
cx q[4],q[0];
cx q[2],q[4];
cx q[2],q[8];
cx q[5],q[2];
cx q[5],q[8];
cx q[2],q[8];
cx q[5],q[2];
cx q[2],q[4];
cx q[8],q[5];
cx q[0],q[2];
swap q[8],q[7];
cx q[4],q[0];
cx q[4],q[2];
cx q[0],q[2];
cx q[4],q[0];
cx q[2],q[4];
cx q[0],q[2];
cx q[1],q[4];
cx q[3],q[1];
cx q[4],q[3];
cx q[4],q[1];
cx q[3],q[1];
cx q[4],q[3];
swap q[4],q[2];
cx q[1],q[2];
cx q[1],q[7];
cx q[3],q[1];
cx q[7],q[3];
cx q[7],q[1];
cx q[3],q[1];
cx q[7],q[3];
cx q[1],q[7];
cx q[1],q[5];
cx q[3],q[1];
cx q[5],q[3];
cx q[5],q[1];
cx q[3],q[1];
cx q[5],q[3];
cx q[1],q[5];
cx q[1],q[2];
cx q[7],q[5];
cx q[3],q[0];
cx q[4],q[3];
cx q[4],q[0];
cx q[3],q[0];
cx q[4],q[3];
cx q[0],q[4];
cx q[0],q[1];
cx q[2],q[0];
cx q[2],q[1];
cx q[0],q[1];
cx q[2],q[0];
cx q[1],q[2];
cx q[1],q[7];
cx q[5],q[1];
cx q[5],q[7];
cx q[1],q[7];
cx q[5],q[1];
cx q[1],q[2];
cx q[0],q[1];
cx q[7],q[5];
cx q[2],q[0];
cx q[2],q[1];
cx q[7],q[5];
cx q[0],q[1];
cx q[2],q[0];
cx q[1],q[2];
cx q[1],q[2];
cx q[0],q[4];
cx q[3],q[0];
cx q[4],q[3];
cx q[4],q[0];
cx q[3],q[0];
cx q[4],q[3];
cx q[0],q[4];
swap q[5],q[3];
cx q[0],q[1];
cx q[2],q[0];
cx q[2],q[1];
cx q[0],q[1];
cx q[2],q[0];
cx q[1],q[2];
cx q[1],q[7];
cx q[3],q[1];
cx q[3],q[7];
cx q[1],q[7];
cx q[3],q[1];
cx q[1],q[2];
cx q[7],q[3];
cx q[0],q[1];
cx q[2],q[0];
cx q[2],q[1];
cx q[0],q[1];
cx q[2],q[0];
cx q[1],q[2];
swap q[7],q[1];
cx q[4],q[2];
cx q[5],q[4];
cx q[2],q[5];
cx q[2],q[4];
cx q[5],q[4];
cx q[2],q[5];
cx q[4],q[2];
cx q[4],q[1];
cx q[5],q[4];
cx q[1],q[5];
cx q[1],q[4];
cx q[5],q[4];
cx q[1],q[5];
cx q[4],q[1];
cx q[4],q[3];
cx q[5],q[4];
cx q[3],q[5];
cx q[3],q[4];
cx q[5],q[4];
swap q[4],q[3];
cx q[2],q[1];
cx q[4],q[5];
cx q[3],q[4];
cx q[5],q[2];
cx q[1],q[5];
cx q[1],q[2];
cx q[5],q[2];
cx q[1],q[5];
cx q[2],q[1];
cx q[2],q[4];
cx q[5],q[2];
cx q[4],q[5];
cx q[4],q[2];
cx q[5],q[2];
cx q[4],q[5];
cx q[2],q[4];
cx q[1],q[4];
cx q[5],q[1];
cx q[4],q[5];
cx q[4],q[1];
cx q[5],q[1];
cx q[4],q[5];
cx q[1],q[4];
