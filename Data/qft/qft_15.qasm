OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
u2(0,pi) q[14];
p(pi/4) q[14];
cx q[14],q[13];
p(-pi/4) q[13];
cx q[14],q[13];
p(pi/4) q[13];
u2(0,pi) q[13];
p(pi/4) q[13];
p(pi/8) q[14];
cx q[14],q[12];
p(-pi/8) q[12];
cx q[14],q[12];
p(pi/8) q[12];
cx q[13],q[12];
p(-pi/4) q[12];
cx q[13],q[12];
p(pi/4) q[12];
u2(0,pi) q[12];
p(pi/4) q[12];
p(pi/8) q[13];
p(pi/16) q[14];
cx q[14],q[11];
p(-pi/16) q[11];
cx q[14],q[11];
p(pi/16) q[11];
cx q[13],q[11];
p(-pi/8) q[11];
cx q[13],q[11];
p(pi/8) q[11];
cx q[12],q[11];
p(-pi/4) q[11];
cx q[12],q[11];
p(pi/4) q[11];
u2(0,pi) q[11];
p(pi/4) q[11];
p(pi/8) q[12];
p(pi/16) q[13];
p(pi/32) q[14];
cx q[14],q[10];
p(-pi/32) q[10];
cx q[14],q[10];
p(pi/32) q[10];
cx q[13],q[10];
p(-pi/16) q[10];
cx q[13],q[10];
p(pi/16) q[10];
cx q[12],q[10];
p(-pi/8) q[10];
cx q[12],q[10];
p(pi/8) q[10];
cx q[11],q[10];
p(-pi/4) q[10];
cx q[11],q[10];
p(pi/4) q[10];
u2(0,pi) q[10];
p(pi/4) q[10];
p(pi/8) q[11];
p(pi/16) q[12];
p(pi/32) q[13];
p(pi/64) q[14];
cx q[14],q[9];
p(-pi/64) q[9];
cx q[14],q[9];
p(pi/128) q[14];
cx q[14],q[8];
p(-pi/128) q[8];
cx q[14],q[8];
p(pi/256) q[14];
cx q[14],q[7];
p(-pi/256) q[7];
cx q[14],q[7];
p(pi/512) q[14];
cx q[14],q[6];
p(-pi/512) q[6];
cx q[14],q[6];
p(pi/1024) q[14];
cx q[14],q[5];
p(-pi/1024) q[5];
cx q[14],q[5];
p(pi/2048) q[14];
cx q[14],q[4];
p(-pi/2048) q[4];
cx q[14],q[4];
p(pi/4096) q[14];
cx q[14],q[3];
p(-pi/4096) q[3];
cx q[14],q[3];
p(pi/8192) q[14];
cx q[14],q[2];
p(-pi/8192) q[2];
cx q[14],q[2];
p(pi/16384) q[14];
cx q[14],q[1];
p(-pi/16384) q[1];
cx q[14],q[1];
p(pi/16384) q[1];
p(pi/32768) q[14];
cx q[14],q[0];
p(-pi/32768) q[0];
cx q[14],q[0];
p(pi/32768) q[0];
p(pi/8192) q[2];
p(pi/4096) q[3];
p(pi/2048) q[4];
p(pi/1024) q[5];
p(pi/512) q[6];
p(pi/256) q[7];
p(pi/128) q[8];
p(pi/64) q[9];
cx q[13],q[9];
p(-pi/32) q[9];
cx q[13],q[9];
p(pi/64) q[13];
cx q[13],q[8];
p(-pi/64) q[8];
cx q[13],q[8];
p(pi/128) q[13];
cx q[13],q[7];
p(-pi/128) q[7];
cx q[13],q[7];
p(pi/256) q[13];
cx q[13],q[6];
p(-pi/256) q[6];
cx q[13],q[6];
p(pi/512) q[13];
cx q[13],q[5];
p(-pi/512) q[5];
cx q[13],q[5];
p(pi/1024) q[13];
cx q[13],q[4];
p(-pi/1024) q[4];
cx q[13],q[4];
p(pi/2048) q[13];
cx q[13],q[3];
p(-pi/2048) q[3];
cx q[13],q[3];
p(pi/4096) q[13];
cx q[13],q[2];
p(-pi/4096) q[2];
cx q[13],q[2];
p(pi/8192) q[13];
cx q[13],q[1];
p(-pi/8192) q[1];
cx q[13],q[1];
p(pi/8192) q[1];
p(pi/16384) q[13];
cx q[13],q[0];
p(-pi/16384) q[0];
cx q[13],q[0];
p(pi/16384) q[0];
p(pi/4096) q[2];
p(pi/2048) q[3];
p(pi/1024) q[4];
p(pi/512) q[5];
p(pi/256) q[6];
p(pi/128) q[7];
p(pi/64) q[8];
p(pi/32) q[9];
cx q[12],q[9];
p(-pi/16) q[9];
cx q[12],q[9];
p(pi/32) q[12];
cx q[12],q[8];
p(-pi/32) q[8];
cx q[12],q[8];
p(pi/64) q[12];
cx q[12],q[7];
p(-pi/64) q[7];
cx q[12],q[7];
p(pi/128) q[12];
cx q[12],q[6];
p(-pi/128) q[6];
cx q[12],q[6];
p(pi/256) q[12];
cx q[12],q[5];
p(-pi/256) q[5];
cx q[12],q[5];
p(pi/512) q[12];
cx q[12],q[4];
p(-pi/512) q[4];
cx q[12],q[4];
p(pi/1024) q[12];
cx q[12],q[3];
p(-pi/1024) q[3];
cx q[12],q[3];
p(pi/2048) q[12];
cx q[12],q[2];
p(-pi/2048) q[2];
cx q[12],q[2];
p(pi/4096) q[12];
cx q[12],q[1];
p(-pi/4096) q[1];
cx q[12],q[1];
p(pi/4096) q[1];
p(pi/8192) q[12];
cx q[12],q[0];
p(-pi/8192) q[0];
cx q[12],q[0];
p(pi/8192) q[0];
p(pi/2048) q[2];
p(pi/1024) q[3];
p(pi/512) q[4];
p(pi/256) q[5];
p(pi/128) q[6];
p(pi/64) q[7];
p(pi/32) q[8];
p(pi/16) q[9];
cx q[11],q[9];
p(-pi/8) q[9];
cx q[11],q[9];
p(pi/16) q[11];
cx q[11],q[8];
p(-pi/16) q[8];
cx q[11],q[8];
p(pi/32) q[11];
cx q[11],q[7];
p(-pi/32) q[7];
cx q[11],q[7];
p(pi/64) q[11];
cx q[11],q[6];
p(-pi/64) q[6];
cx q[11],q[6];
p(pi/128) q[11];
cx q[11],q[5];
p(-pi/128) q[5];
cx q[11],q[5];
p(pi/256) q[11];
cx q[11],q[4];
p(-pi/256) q[4];
cx q[11],q[4];
p(pi/512) q[11];
cx q[11],q[3];
p(-pi/512) q[3];
cx q[11],q[3];
p(pi/1024) q[11];
cx q[11],q[2];
p(-pi/1024) q[2];
cx q[11],q[2];
p(pi/2048) q[11];
cx q[11],q[1];
p(-pi/2048) q[1];
cx q[11],q[1];
p(pi/2048) q[1];
p(pi/4096) q[11];
cx q[11],q[0];
p(-pi/4096) q[0];
cx q[11],q[0];
p(pi/4096) q[0];
p(pi/1024) q[2];
p(pi/512) q[3];
p(pi/256) q[4];
p(pi/128) q[5];
p(pi/64) q[6];
p(pi/32) q[7];
p(pi/16) q[8];
p(pi/8) q[9];
cx q[10],q[9];
p(-pi/4) q[9];
cx q[10],q[9];
p(pi/8) q[10];
cx q[10],q[8];
p(-pi/8) q[8];
cx q[10],q[8];
p(pi/16) q[10];
cx q[10],q[7];
p(-pi/16) q[7];
cx q[10],q[7];
p(pi/32) q[10];
cx q[10],q[6];
p(-pi/32) q[6];
cx q[10],q[6];
p(pi/64) q[10];
cx q[10],q[5];
p(-pi/64) q[5];
cx q[10],q[5];
p(pi/128) q[10];
cx q[10],q[4];
p(-pi/128) q[4];
cx q[10],q[4];
p(pi/256) q[10];
cx q[10],q[3];
p(-pi/256) q[3];
cx q[10],q[3];
p(pi/512) q[10];
cx q[10],q[2];
p(-pi/512) q[2];
cx q[10],q[2];
p(pi/1024) q[10];
cx q[10],q[1];
p(-pi/1024) q[1];
cx q[10],q[1];
p(pi/1024) q[1];
p(pi/2048) q[10];
cx q[10],q[0];
p(-pi/2048) q[0];
cx q[10],q[0];
p(pi/2048) q[0];
p(pi/512) q[2];
p(pi/256) q[3];
p(pi/128) q[4];
p(pi/64) q[5];
p(pi/32) q[6];
p(pi/16) q[7];
p(pi/8) q[8];
p(pi/4) q[9];
u2(0,pi) q[9];
p(pi/4) q[9];
cx q[9],q[8];
p(-pi/4) q[8];
cx q[9],q[8];
p(pi/4) q[8];
u2(0,pi) q[8];
p(pi/4) q[8];
p(pi/8) q[9];
cx q[9],q[7];
p(-pi/8) q[7];
cx q[9],q[7];
p(pi/8) q[7];
cx q[8],q[7];
p(-pi/4) q[7];
cx q[8],q[7];
p(pi/4) q[7];
u2(0,pi) q[7];
p(pi/4) q[7];
p(pi/8) q[8];
p(pi/16) q[9];
cx q[9],q[6];
p(-pi/16) q[6];
cx q[9],q[6];
p(pi/16) q[6];
cx q[8],q[6];
p(-pi/8) q[6];
cx q[8],q[6];
p(pi/8) q[6];
cx q[7],q[6];
p(-pi/4) q[6];
cx q[7],q[6];
p(pi/4) q[6];
u2(0,pi) q[6];
p(pi/4) q[6];
p(pi/8) q[7];
p(pi/16) q[8];
p(pi/32) q[9];
cx q[9],q[5];
p(-pi/32) q[5];
cx q[9],q[5];
p(pi/32) q[5];
cx q[8],q[5];
p(-pi/16) q[5];
cx q[8],q[5];
p(pi/16) q[5];
cx q[7],q[5];
p(-pi/8) q[5];
cx q[7],q[5];
p(pi/8) q[5];
cx q[6],q[5];
p(-pi/4) q[5];
cx q[6],q[5];
p(pi/4) q[5];
u2(0,pi) q[5];
p(pi/4) q[5];
p(pi/8) q[6];
p(pi/16) q[7];
p(pi/32) q[8];
p(pi/64) q[9];
cx q[9],q[4];
p(-pi/64) q[4];
cx q[9],q[4];
p(pi/64) q[4];
cx q[8],q[4];
p(-pi/32) q[4];
cx q[8],q[4];
p(pi/32) q[4];
cx q[7],q[4];
p(-pi/16) q[4];
cx q[7],q[4];
p(pi/16) q[4];
cx q[6],q[4];
p(-pi/8) q[4];
cx q[6],q[4];
p(pi/8) q[4];
cx q[5],q[4];
p(-pi/4) q[4];
cx q[5],q[4];
p(pi/4) q[4];
u2(0,pi) q[4];
p(pi/4) q[4];
p(pi/8) q[5];
p(pi/16) q[6];
p(pi/32) q[7];
p(pi/64) q[8];
p(pi/128) q[9];
cx q[9],q[3];
p(-pi/128) q[3];
cx q[9],q[3];
p(pi/128) q[3];
cx q[8],q[3];
p(-pi/64) q[3];
cx q[8],q[3];
p(pi/64) q[3];
cx q[7],q[3];
p(-pi/32) q[3];
cx q[7],q[3];
p(pi/32) q[3];
cx q[6],q[3];
p(-pi/16) q[3];
cx q[6],q[3];
p(pi/16) q[3];
cx q[5],q[3];
p(-pi/8) q[3];
cx q[5],q[3];
p(pi/8) q[3];
cx q[4],q[3];
p(-pi/4) q[3];
cx q[4],q[3];
p(pi/4) q[3];
u2(0,pi) q[3];
p(pi/4) q[3];
p(pi/8) q[4];
p(pi/16) q[5];
p(pi/32) q[6];
p(pi/64) q[7];
p(pi/128) q[8];
p(pi/256) q[9];
cx q[9],q[2];
p(-pi/256) q[2];
cx q[9],q[2];
p(pi/256) q[2];
cx q[8],q[2];
p(-pi/128) q[2];
cx q[8],q[2];
p(pi/128) q[2];
cx q[7],q[2];
p(-pi/64) q[2];
cx q[7],q[2];
p(pi/64) q[2];
cx q[6],q[2];
p(-pi/32) q[2];
cx q[6],q[2];
p(pi/32) q[2];
cx q[5],q[2];
p(-pi/16) q[2];
cx q[5],q[2];
p(pi/16) q[2];
cx q[4],q[2];
p(-pi/8) q[2];
cx q[4],q[2];
p(pi/8) q[2];
cx q[3],q[2];
p(-pi/4) q[2];
cx q[3],q[2];
p(pi/4) q[2];
u2(0,pi) q[2];
p(pi/4) q[2];
p(pi/8) q[3];
p(pi/16) q[4];
p(pi/32) q[5];
p(pi/64) q[6];
p(pi/128) q[7];
p(pi/256) q[8];
p(pi/512) q[9];
cx q[9],q[1];
p(-pi/512) q[1];
cx q[9],q[1];
p(pi/512) q[1];
cx q[8],q[1];
p(-pi/256) q[1];
cx q[8],q[1];
p(pi/256) q[1];
cx q[7],q[1];
p(-pi/128) q[1];
cx q[7],q[1];
p(pi/128) q[1];
cx q[6],q[1];
p(-pi/64) q[1];
cx q[6],q[1];
p(pi/64) q[1];
cx q[5],q[1];
p(-pi/32) q[1];
cx q[5],q[1];
p(pi/32) q[1];
cx q[4],q[1];
p(-pi/16) q[1];
cx q[4],q[1];
p(pi/16) q[1];
cx q[3],q[1];
p(-pi/8) q[1];
cx q[3],q[1];
p(pi/8) q[1];
cx q[2],q[1];
p(-pi/4) q[1];
cx q[2],q[1];
p(pi/4) q[1];
u2(0,pi) q[1];
p(pi/4) q[1];
p(pi/8) q[2];
p(pi/16) q[3];
p(pi/32) q[4];
p(pi/64) q[5];
p(pi/128) q[6];
p(pi/256) q[7];
p(pi/512) q[8];
p(pi/1024) q[9];
cx q[9],q[0];
p(-pi/1024) q[0];
cx q[9],q[0];
p(pi/1024) q[0];
cx q[8],q[0];
p(-pi/512) q[0];
cx q[8],q[0];
p(pi/512) q[0];
cx q[7],q[0];
p(-pi/256) q[0];
cx q[7],q[0];
p(pi/256) q[0];
cx q[6],q[0];
p(-pi/128) q[0];
cx q[6],q[0];
p(pi/128) q[0];
cx q[5],q[0];
p(-pi/64) q[0];
cx q[5],q[0];
p(pi/64) q[0];
cx q[4],q[0];
p(-pi/32) q[0];
cx q[4],q[0];
p(pi/32) q[0];
cx q[3],q[0];
p(-pi/16) q[0];
cx q[3],q[0];
p(pi/16) q[0];
cx q[2],q[0];
p(-pi/8) q[0];
cx q[2],q[0];
p(pi/8) q[0];
cx q[1],q[0];
p(-pi/4) q[0];
cx q[1],q[0];
p(pi/4) q[0];
u2(0,pi) q[0];
cx q[0],q[14];
cx q[1],q[13];
cx q[13],q[1];
cx q[1],q[13];
cx q[14],q[0];
cx q[0],q[14];
cx q[2],q[12];
cx q[12],q[2];
cx q[2],q[12];
cx q[3],q[11];
cx q[11],q[3];
cx q[3],q[11];
cx q[4],q[10];
cx q[10],q[4];
cx q[4],q[10];
cx q[5],q[9];
cx q[6],q[8];
cx q[8],q[6];
cx q[6],q[8];
cx q[9],q[5];
cx q[5],q[9];
