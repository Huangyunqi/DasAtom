OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
cx q[7],q[3];
cx q[9],q[5];
swap q[13],q[5];
swap q[11],q[7];
cx q[13],q[14];
cx q[6],q[5];
swap q[10],q[8];
swap q[3],q[1];
cx q[11],q[10];
cx q[1],q[0];
swap q[13],q[8];
cx q[11],q[7];
swap q[5],q[1];
swap q[11],q[7];
cx q[8],q[4];
swap q[14],q[13];
swap q[2],q[1];
cx q[9],q[11];
cx q[11],q[15];
cx q[9],q[1];
cx q[5],q[10];
swap q[8],q[0];
swap q[11],q[6];
cx q[8],q[13];
cx q[11],q[15];
swap q[10],q[8];
cx q[10],q[14];
cx q[4],q[8];
cx q[13],q[14];
cx q[4],q[1];
cx q[11],q[14];
cx q[2],q[1];
swap q[9],q[8];
swap q[15],q[7];
swap q[5],q[1];
cx q[9],q[14];
cx q[2],q[7];
cx q[9],q[6];
cx q[9],q[10];
cx q[5],q[10];
cx q[6],q[11];
cx q[11],q[7];
swap q[9],q[4];
swap q[6],q[2];
swap q[15],q[11];
swap q[13],q[8];
swap q[15],q[7];
swap q[4],q[0];
cx q[14],q[6];
cx q[14],q[13];
cx q[6],q[11];
cx q[5],q[13];
swap q[7],q[3];
cx q[10],q[9];
cx q[5],q[1];
swap q[15],q[10];
cx q[3],q[1];
cx q[9],q[11];
cx q[6],q[4];
cx q[9],q[1];
swap q[10],q[8];
swap q[2],q[1];
cx q[11],q[10];
cx q[8],q[4];
cx q[8],q[10];
swap q[13],q[5];
cx q[1],q[5];
cx q[13],q[14];
cx q[13],q[8];
cx q[1],q[3];
swap q[10],q[5];
swap q[13],q[12];
cx q[14],q[10];
cx q[4],q[5];
swap q[7],q[2];
cx q[12],q[4];
cx q[14],q[15];
swap q[8],q[0];
cx q[10],q[7];
cx q[7],q[11];
cx q[11],q[9];
cx q[1],q[0];
cx q[10],q[6];
swap q[4],q[1];
swap q[7],q[6];
cx q[4],q[8];
swap q[14],q[13];
cx q[6],q[1];
cx q[9],q[8];
cx q[3],q[1];
swap q[15],q[11];
swap q[13],q[8];
cx q[3],q[11];
swap q[1],q[0];
cx q[7],q[11];
cx q[9],q[1];
swap q[15],q[7];
cx q[0],q[8];
swap q[12],q[4];
swap q[15],q[10];
cx q[7],q[5];
swap q[14],q[12];
cx q[7],q[6];
cx q[6],q[1];
cx q[10],q[5];
cx q[7],q[10];
cx q[0],q[1];
cx q[13],q[5];
swap q[11],q[7];
cx q[0],q[4];
swap q[13],q[9];
swap q[7],q[2];
swap q[15],q[13];
swap q[2],q[0];
cx q[11],q[14];
cx q[14],q[13];
cx q[14],q[10];
cx q[8],q[0];
swap q[7],q[3];
cx q[10],q[5];
swap q[15],q[10];
swap q[8],q[0];
swap q[7],q[3];
swap q[1],q[0];
cx q[6],q[9];
cx q[13],q[9];
swap q[11],q[7];
cx q[9],q[5];
cx q[5],q[4];
cx q[13],q[10];
cx q[1],q[3];
cx q[10],q[8];
swap q[3],q[1];
cx q[3],q[6];
cx q[0],q[4];
cx q[3],q[7];
cx q[4],q[9];
cx q[9],q[13];
swap q[2],q[0];
swap q[13],q[10];
cx q[3],q[2];
swap q[4],q[0];
cx q[6],q[2];
cx q[10],q[2];
cx q[6],q[5];
swap q[13],q[8];
swap q[6],q[3];
swap q[1],q[0];
cx q[8],q[0];
cx q[10],q[15];
cx q[9],q[4];
cx q[9],q[6];
swap q[4],q[0];
cx q[6],q[10];
swap q[13],q[9];
swap q[1],q[0];
swap q[14],q[12];
cx q[9],q[4];
swap q[15],q[14];
cx q[0],q[4];
cx q[12],q[4];
cx q[0],q[8];
swap q[14],q[6];
swap q[12],q[8];
swap q[10],q[7];
swap q[13],q[12];
cx q[4],q[6];
cx q[8],q[5];
cx q[8],q[9];
swap q[3],q[1];
swap q[14],q[12];
swap q[7],q[3];
swap q[5],q[4];
swap q[15],q[13];
cx q[6],q[9];
swap q[15],q[11];
swap q[9],q[6];
cx q[6],q[10];
swap q[8],q[4];
cx q[11],q[10];
cx q[11],q[7];
swap q[14],q[13];
swap q[7],q[6];
cx q[13],q[9];
cx q[10],q[2];
cx q[10],q[11];
cx q[2],q[1];
cx q[11],q[7];
cx q[13],q[5];
cx q[2],q[0];
cx q[10],q[9];
cx q[7],q[3];
cx q[1],q[4];
cx q[1],q[5];
swap q[2],q[0];
swap q[9],q[8];
cx q[9],q[6];
cx q[2],q[3];
cx q[12],q[9];
swap q[4],q[0];
cx q[13],q[12];
cx q[7],q[6];
cx q[2],q[0];
cx q[11],q[9];
cx q[8],q[9];
swap q[6],q[2];
swap q[12],q[4];
swap q[10],q[6];
cx q[13],q[10];
cx q[2],q[0];
cx q[13],q[12];
cx q[2],q[5];
cx q[4],q[0];
swap q[10],q[6];
swap q[13],q[12];
cx q[10],q[13];
swap q[5],q[1];
swap q[13],q[8];
cx q[6],q[3];
cx q[10],q[5];
swap q[3],q[2];
cx q[4],q[9];
cx q[8],q[0];
cx q[10],q[7];
cx q[2],q[5];
cx q[2],q[1];
swap q[13],q[10];
cx q[5],q[1];
swap q[2],q[0];
swap q[14],q[12];
swap q[5],q[0];
cx q[6],q[10];
cx q[10],q[7];
cx q[6],q[14];
cx q[7],q[11];
cx q[11],q[9];
cx q[6],q[5];
cx q[11],q[3];
cx q[5],q[0];
cx q[2],q[3];
cx q[14],q[11];
cx q[5],q[2];
swap q[9],q[1];
cx q[10],q[9];
cx q[3],q[1];
cx q[6],q[10];
swap q[8],q[0];
swap q[14],q[6];
cx q[8],q[10];
swap q[3],q[1];
swap q[10],q[8];
cx q[10],q[11];
cx q[6],q[3];
cx q[3],q[1];
cx q[0],q[1];
swap q[11],q[3];
cx q[0],q[4];
cx q[0],q[8];
cx q[2],q[3];
swap q[13],q[10];
swap q[5],q[4];
cx q[2],q[10];
cx q[11],q[10];
cx q[10],q[7];
cx q[7],q[5];
cx q[1],q[9];
swap q[7],q[5];
cx q[5],q[9];
swap q[7],q[5];
cx q[9],q[5];
swap q[15],q[7];
swap q[9],q[1];
swap q[7],q[3];
cx q[5],q[13];
cx q[5],q[9];
swap q[15],q[14];
cx q[5],q[2];
cx q[2],q[1];
cx q[10],q[9];
swap q[2],q[0];
cx q[13],q[14];
swap q[14],q[13];
cx q[0],q[4];
swap q[7],q[5];
swap q[13],q[12];
cx q[10],q[5];
cx q[10],q[8];
swap q[3],q[1];
cx q[8],q[12];
cx q[14],q[11];
swap q[15],q[10];
cx q[12],q[4];
swap q[7],q[3];
cx q[8],q[5];
cx q[9],q[6];
swap q[1],q[0];
cx q[7],q[11];
cx q[7],q[10];
cx q[5],q[6];
cx q[6],q[2];
swap q[9],q[4];
swap q[7],q[2];
cx q[9],q[10];
cx q[11],q[7];
cx q[1],q[2];
swap q[14],q[9];
swap q[6],q[2];
cx q[10],q[7];
cx q[1],q[4];
cx q[1],q[9];
swap q[7],q[3];
swap q[12],q[9];
swap q[3],q[1];
swap q[13],q[12];
cx q[10],q[7];
swap q[4],q[1];
swap q[10],q[9];
cx q[9],q[8];
cx q[9],q[5];
cx q[7],q[5];
cx q[5],q[4];
cx q[7],q[10];
cx q[10],q[11];
swap q[13],q[12];
swap q[5],q[2];
cx q[11],q[15];
swap q[15],q[13];
cx q[4],q[12];
cx q[4],q[8];
cx q[6],q[14];
cx q[8],q[13];
cx q[6],q[1];
cx q[10],q[14];
swap q[8],q[5];
cx q[11],q[14];
cx q[12],q[8];
swap q[5],q[1];
cx q[13],q[8];
cx q[8],q[5];
