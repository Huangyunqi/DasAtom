OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
u2(0,pi) q[24];
p(pi/4) q[24];
cx q[24],q[23];
p(-pi/4) q[23];
cx q[24],q[23];
p(pi/4) q[23];
u2(0,pi) q[23];
p(pi/4) q[23];
p(pi/8) q[24];
cx q[24],q[22];
p(-pi/8) q[22];
cx q[24],q[22];
p(pi/8) q[22];
cx q[23],q[22];
p(-pi/4) q[22];
cx q[23],q[22];
p(pi/4) q[22];
u2(0,pi) q[22];
p(pi/4) q[22];
p(pi/8) q[23];
p(pi/16) q[24];
cx q[24],q[21];
p(-pi/16) q[21];
cx q[24],q[21];
p(pi/16) q[21];
cx q[23],q[21];
p(-pi/8) q[21];
cx q[23],q[21];
p(pi/8) q[21];
cx q[22],q[21];
p(-pi/4) q[21];
cx q[22],q[21];
p(pi/4) q[21];
u2(0,pi) q[21];
p(pi/4) q[21];
p(pi/8) q[22];
p(pi/16) q[23];
p(pi/32) q[24];
cx q[24],q[20];
p(-pi/32) q[20];
cx q[24],q[20];
p(pi/32) q[20];
cx q[23],q[20];
p(-pi/16) q[20];
cx q[23],q[20];
p(pi/16) q[20];
cx q[22],q[20];
p(-pi/8) q[20];
cx q[22],q[20];
p(pi/8) q[20];
cx q[21],q[20];
p(-pi/4) q[20];
cx q[21],q[20];
p(pi/4) q[20];
u2(0,pi) q[20];
p(pi/4) q[20];
p(pi/8) q[21];
p(pi/16) q[22];
p(pi/32) q[23];
p(pi/64) q[24];
cx q[24],q[19];
p(-pi/64) q[19];
cx q[24],q[19];
p(pi/64) q[19];
cx q[23],q[19];
p(-pi/32) q[19];
cx q[23],q[19];
p(pi/32) q[19];
cx q[22],q[19];
p(-pi/16) q[19];
cx q[22],q[19];
p(pi/16) q[19];
cx q[21],q[19];
p(-pi/8) q[19];
cx q[21],q[19];
p(pi/8) q[19];
cx q[20],q[19];
p(-pi/4) q[19];
cx q[20],q[19];
p(pi/4) q[19];
u2(0,pi) q[19];
p(pi/4) q[19];
p(pi/8) q[20];
p(pi/16) q[21];
p(pi/32) q[22];
p(pi/64) q[23];
p(pi/128) q[24];
cx q[24],q[18];
p(-pi/128) q[18];
cx q[24],q[18];
p(pi/128) q[18];
cx q[23],q[18];
p(-pi/64) q[18];
cx q[23],q[18];
p(pi/64) q[18];
cx q[22],q[18];
p(-pi/32) q[18];
cx q[22],q[18];
p(pi/32) q[18];
cx q[21],q[18];
p(-pi/16) q[18];
cx q[21],q[18];
p(pi/16) q[18];
cx q[20],q[18];
p(-pi/8) q[18];
cx q[20],q[18];
p(pi/8) q[18];
cx q[19],q[18];
p(-pi/4) q[18];
cx q[19],q[18];
p(pi/4) q[18];
u2(0,pi) q[18];
p(pi/4) q[18];
p(pi/8) q[19];
p(pi/16) q[20];
p(pi/32) q[21];
p(pi/64) q[22];
p(pi/128) q[23];
p(pi/256) q[24];
cx q[24],q[17];
p(-pi/256) q[17];
cx q[24],q[17];
p(pi/256) q[17];
cx q[23],q[17];
p(-pi/128) q[17];
cx q[23],q[17];
p(pi/128) q[17];
cx q[22],q[17];
p(-pi/64) q[17];
cx q[22],q[17];
p(pi/64) q[17];
cx q[21],q[17];
p(-pi/32) q[17];
cx q[21],q[17];
p(pi/32) q[17];
cx q[20],q[17];
p(-pi/16) q[17];
cx q[20],q[17];
p(pi/16) q[17];
cx q[19],q[17];
p(-pi/8) q[17];
cx q[19],q[17];
p(pi/8) q[17];
cx q[18],q[17];
p(-pi/4) q[17];
cx q[18],q[17];
p(pi/4) q[17];
u2(0,pi) q[17];
p(pi/4) q[17];
p(pi/8) q[18];
p(pi/16) q[19];
p(pi/32) q[20];
p(pi/64) q[21];
p(pi/128) q[22];
p(pi/256) q[23];
p(pi/512) q[24];
cx q[24],q[16];
p(-pi/512) q[16];
cx q[24],q[16];
p(pi/512) q[16];
cx q[23],q[16];
p(-pi/256) q[16];
cx q[23],q[16];
p(pi/256) q[16];
cx q[22],q[16];
p(-pi/128) q[16];
cx q[22],q[16];
p(pi/128) q[16];
cx q[21],q[16];
p(-pi/64) q[16];
cx q[21],q[16];
p(pi/64) q[16];
cx q[20],q[16];
p(-pi/32) q[16];
cx q[20],q[16];
p(pi/32) q[16];
cx q[19],q[16];
p(-pi/16) q[16];
cx q[19],q[16];
p(pi/16) q[16];
cx q[18],q[16];
p(-pi/8) q[16];
cx q[18],q[16];
p(pi/8) q[16];
cx q[17],q[16];
p(-pi/4) q[16];
cx q[17],q[16];
p(pi/4) q[16];
u2(0,pi) q[16];
p(pi/4) q[16];
p(pi/8) q[17];
p(pi/16) q[18];
p(pi/32) q[19];
p(pi/64) q[20];
p(pi/128) q[21];
p(pi/256) q[22];
p(pi/512) q[23];
p(pi/1024) q[24];
cx q[24],q[15];
p(-pi/1024) q[15];
cx q[24],q[15];
p(pi/1024) q[15];
cx q[23],q[15];
p(-pi/512) q[15];
cx q[23],q[15];
p(pi/512) q[15];
cx q[22],q[15];
p(-pi/256) q[15];
cx q[22],q[15];
p(pi/256) q[15];
cx q[21],q[15];
p(-pi/128) q[15];
cx q[21],q[15];
p(pi/128) q[15];
cx q[20],q[15];
p(-pi/64) q[15];
cx q[20],q[15];
p(pi/64) q[15];
cx q[19],q[15];
p(-pi/32) q[15];
cx q[19],q[15];
p(pi/32) q[15];
cx q[18],q[15];
p(-pi/16) q[15];
cx q[18],q[15];
p(pi/16) q[15];
cx q[17],q[15];
p(-pi/8) q[15];
cx q[17],q[15];
p(pi/8) q[15];
cx q[16],q[15];
p(-pi/4) q[15];
cx q[16],q[15];
p(pi/4) q[15];
u2(0,pi) q[15];
p(pi/4) q[15];
p(pi/8) q[16];
p(pi/16) q[17];
p(pi/32) q[18];
p(pi/64) q[19];
p(pi/128) q[20];
p(pi/256) q[21];
p(pi/512) q[22];
p(pi/1024) q[23];
p(pi/2048) q[24];
cx q[24],q[14];
p(-pi/2048) q[14];
cx q[24],q[14];
p(pi/2048) q[14];
cx q[23],q[14];
p(-pi/1024) q[14];
cx q[23],q[14];
p(pi/1024) q[14];
cx q[22],q[14];
p(-pi/512) q[14];
cx q[22],q[14];
p(pi/512) q[14];
cx q[21],q[14];
p(-pi/256) q[14];
cx q[21],q[14];
p(pi/256) q[14];
cx q[20],q[14];
p(-pi/128) q[14];
cx q[20],q[14];
p(pi/128) q[14];
cx q[19],q[14];
p(-pi/64) q[14];
cx q[19],q[14];
p(pi/64) q[14];
cx q[18],q[14];
p(-pi/32) q[14];
cx q[18],q[14];
p(pi/32) q[14];
cx q[17],q[14];
p(-pi/16) q[14];
cx q[17],q[14];
p(pi/16) q[14];
cx q[16],q[14];
p(-pi/8) q[14];
cx q[16],q[14];
p(pi/8) q[14];
cx q[15],q[14];
p(-pi/4) q[14];
cx q[15],q[14];
p(pi/4) q[14];
u2(0,pi) q[14];
p(pi/4) q[14];
p(pi/8) q[15];
p(pi/16) q[16];
p(pi/32) q[17];
p(pi/64) q[18];
p(pi/128) q[19];
p(pi/256) q[20];
p(pi/512) q[21];
p(pi/1024) q[22];
p(pi/2048) q[23];
p(pi/4096) q[24];
cx q[24],q[13];
p(-pi/4096) q[13];
cx q[24],q[13];
p(pi/4096) q[13];
cx q[23],q[13];
p(-pi/2048) q[13];
cx q[23],q[13];
p(pi/2048) q[13];
cx q[22],q[13];
p(-pi/1024) q[13];
cx q[22],q[13];
p(pi/1024) q[13];
cx q[21],q[13];
p(-pi/512) q[13];
cx q[21],q[13];
p(pi/512) q[13];
cx q[20],q[13];
p(-pi/256) q[13];
cx q[20],q[13];
p(pi/256) q[13];
cx q[19],q[13];
p(-pi/128) q[13];
cx q[19],q[13];
p(pi/128) q[13];
cx q[18],q[13];
p(-pi/64) q[13];
cx q[18],q[13];
p(pi/64) q[13];
cx q[17],q[13];
p(-pi/32) q[13];
cx q[17],q[13];
p(pi/32) q[13];
cx q[16],q[13];
p(-pi/16) q[13];
cx q[16],q[13];
p(pi/16) q[13];
cx q[15],q[13];
p(-pi/8) q[13];
cx q[15],q[13];
p(pi/8) q[13];
cx q[14],q[13];
p(-pi/4) q[13];
cx q[14],q[13];
p(pi/4) q[13];
u2(0,pi) q[13];
p(pi/4) q[13];
p(pi/8) q[14];
p(pi/16) q[15];
p(pi/32) q[16];
p(pi/64) q[17];
p(pi/128) q[18];
p(pi/256) q[19];
p(pi/512) q[20];
p(pi/1024) q[21];
p(pi/2048) q[22];
p(pi/4096) q[23];
p(pi/8192) q[24];
cx q[24],q[12];
p(-pi/8192) q[12];
cx q[24],q[12];
p(pi/8192) q[12];
cx q[23],q[12];
p(-pi/4096) q[12];
cx q[23],q[12];
p(pi/4096) q[12];
cx q[22],q[12];
p(-pi/2048) q[12];
cx q[22],q[12];
p(pi/2048) q[12];
cx q[21],q[12];
p(-pi/1024) q[12];
cx q[21],q[12];
p(pi/1024) q[12];
cx q[20],q[12];
p(-pi/512) q[12];
cx q[20],q[12];
p(pi/512) q[12];
cx q[19],q[12];
p(-pi/256) q[12];
cx q[19],q[12];
p(pi/256) q[12];
cx q[18],q[12];
p(-pi/128) q[12];
cx q[18],q[12];
p(pi/128) q[12];
cx q[17],q[12];
p(-pi/64) q[12];
cx q[17],q[12];
p(pi/64) q[12];
cx q[16],q[12];
p(-pi/32) q[12];
cx q[16],q[12];
p(pi/32) q[12];
cx q[15],q[12];
p(-pi/16) q[12];
cx q[15],q[12];
p(pi/16) q[12];
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
p(pi/32) q[15];
p(pi/64) q[16];
p(pi/128) q[17];
p(pi/256) q[18];
p(pi/512) q[19];
p(pi/1024) q[20];
p(pi/2048) q[21];
p(pi/4096) q[22];
p(pi/8192) q[23];
p(pi/16384) q[24];
cx q[24],q[11];
p(-pi/16384) q[11];
cx q[24],q[11];
p(pi/16384) q[11];
cx q[23],q[11];
p(-pi/8192) q[11];
cx q[23],q[11];
p(pi/8192) q[11];
cx q[22],q[11];
p(-pi/4096) q[11];
cx q[22],q[11];
p(pi/4096) q[11];
cx q[21],q[11];
p(-pi/2048) q[11];
cx q[21],q[11];
p(pi/2048) q[11];
cx q[20],q[11];
p(-pi/1024) q[11];
cx q[20],q[11];
p(pi/1024) q[11];
cx q[19],q[11];
p(-pi/512) q[11];
cx q[19],q[11];
p(pi/512) q[11];
cx q[18],q[11];
p(-pi/256) q[11];
cx q[18],q[11];
p(pi/256) q[11];
cx q[17],q[11];
p(-pi/128) q[11];
cx q[17],q[11];
p(pi/128) q[11];
cx q[16],q[11];
p(-pi/64) q[11];
cx q[16],q[11];
p(pi/64) q[11];
cx q[15],q[11];
p(-pi/32) q[11];
cx q[15],q[11];
p(pi/32) q[11];
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
p(pi/64) q[15];
p(pi/128) q[16];
p(pi/256) q[17];
p(pi/512) q[18];
p(pi/1024) q[19];
p(pi/2048) q[20];
p(pi/4096) q[21];
p(pi/8192) q[22];
p(pi/16384) q[23];
p(pi/32768) q[24];
cx q[24],q[10];
p(-pi/32768) q[10];
cx q[24],q[10];
p(pi/32768) q[10];
cx q[23],q[10];
p(-pi/16384) q[10];
cx q[23],q[10];
p(pi/16384) q[10];
cx q[22],q[10];
p(-pi/8192) q[10];
cx q[22],q[10];
p(pi/8192) q[10];
cx q[21],q[10];
p(-pi/4096) q[10];
cx q[21],q[10];
p(pi/4096) q[10];
cx q[20],q[10];
p(-pi/2048) q[10];
cx q[20],q[10];
p(pi/2048) q[10];
cx q[19],q[10];
p(-pi/1024) q[10];
cx q[19],q[10];
p(pi/1024) q[10];
cx q[18],q[10];
p(-pi/512) q[10];
cx q[18],q[10];
p(pi/512) q[10];
cx q[17],q[10];
p(-pi/256) q[10];
cx q[17],q[10];
p(pi/256) q[10];
cx q[16],q[10];
p(-pi/128) q[10];
cx q[16],q[10];
p(pi/128) q[10];
cx q[15],q[10];
p(-pi/64) q[10];
cx q[15],q[10];
p(pi/64) q[10];
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
p(pi/128) q[15];
p(pi/256) q[16];
p(pi/512) q[17];
p(pi/1024) q[18];
p(pi/2048) q[19];
p(pi/4096) q[20];
p(pi/8192) q[21];
p(pi/16384) q[22];
p(pi/32768) q[23];
p(pi/65536) q[24];
cx q[24],q[9];
p(-pi/65536) q[9];
cx q[24],q[9];
p(pi/65536) q[9];
cx q[23],q[9];
p(-pi/32768) q[9];
cx q[23],q[9];
p(pi/32768) q[9];
cx q[22],q[9];
p(-pi/16384) q[9];
cx q[22],q[9];
p(pi/16384) q[9];
cx q[21],q[9];
p(-pi/8192) q[9];
cx q[21],q[9];
p(pi/8192) q[9];
cx q[20],q[9];
p(-pi/4096) q[9];
cx q[20],q[9];
p(pi/4096) q[9];
cx q[19],q[9];
p(-pi/2048) q[9];
cx q[19],q[9];
p(pi/2048) q[9];
cx q[18],q[9];
p(-pi/1024) q[9];
cx q[18],q[9];
p(pi/1024) q[9];
cx q[17],q[9];
p(-pi/512) q[9];
cx q[17],q[9];
p(pi/512) q[9];
cx q[16],q[9];
p(-pi/256) q[9];
cx q[16],q[9];
p(pi/256) q[9];
cx q[15],q[9];
p(-pi/128) q[9];
cx q[15],q[9];
p(pi/128) q[9];
cx q[14],q[9];
p(-pi/64) q[9];
cx q[14],q[9];
p(pi/64) q[9];
cx q[13],q[9];
p(-pi/32) q[9];
cx q[13],q[9];
p(pi/32) q[9];
cx q[12],q[9];
p(-pi/16) q[9];
cx q[12],q[9];
p(pi/16) q[9];
cx q[11],q[9];
p(-pi/8) q[9];
cx q[11],q[9];
p(pi/8) q[9];
cx q[10],q[9];
p(-pi/4) q[9];
cx q[10],q[9];
p(pi/4) q[9];
u2(0,pi) q[9];
p(pi/4) q[9];
p(pi/8) q[10];
p(pi/16) q[11];
p(pi/32) q[12];
p(pi/64) q[13];
p(pi/128) q[14];
p(pi/256) q[15];
p(pi/512) q[16];
p(pi/1024) q[17];
p(pi/2048) q[18];
p(pi/4096) q[19];
p(pi/8192) q[20];
p(pi/16384) q[21];
p(pi/32768) q[22];
p(pi/65536) q[23];
p(pi/131072) q[24];
cx q[24],q[8];
p(-pi/131072) q[8];
cx q[24],q[8];
p(pi/131072) q[8];
cx q[23],q[8];
p(-pi/65536) q[8];
cx q[23],q[8];
p(pi/65536) q[8];
cx q[22],q[8];
p(-pi/32768) q[8];
cx q[22],q[8];
p(pi/32768) q[8];
cx q[21],q[8];
p(-pi/16384) q[8];
cx q[21],q[8];
p(pi/16384) q[8];
cx q[20],q[8];
p(-pi/8192) q[8];
cx q[20],q[8];
p(pi/8192) q[8];
cx q[19],q[8];
p(-pi/4096) q[8];
cx q[19],q[8];
p(pi/4096) q[8];
cx q[18],q[8];
p(-pi/2048) q[8];
cx q[18],q[8];
p(pi/2048) q[8];
cx q[17],q[8];
p(-pi/1024) q[8];
cx q[17],q[8];
p(pi/1024) q[8];
cx q[16],q[8];
p(-pi/512) q[8];
cx q[16],q[8];
p(pi/512) q[8];
cx q[15],q[8];
p(-pi/256) q[8];
cx q[15],q[8];
p(pi/256) q[8];
cx q[14],q[8];
p(-pi/128) q[8];
cx q[14],q[8];
p(pi/128) q[8];
cx q[13],q[8];
p(-pi/64) q[8];
cx q[13],q[8];
p(pi/64) q[8];
cx q[12],q[8];
p(-pi/32) q[8];
cx q[12],q[8];
p(pi/32) q[8];
cx q[11],q[8];
p(-pi/16) q[8];
cx q[11],q[8];
p(pi/16) q[8];
cx q[10],q[8];
p(-pi/8) q[8];
cx q[10],q[8];
p(pi/8) q[8];
cx q[9],q[8];
p(-pi/4) q[8];
cx q[9],q[8];
p(pi/4) q[8];
u2(0,pi) q[8];
p(pi/4) q[8];
p(pi/8) q[9];
p(pi/16) q[10];
p(pi/32) q[11];
p(pi/64) q[12];
p(pi/128) q[13];
p(pi/256) q[14];
p(pi/512) q[15];
p(pi/1024) q[16];
p(pi/2048) q[17];
p(pi/4096) q[18];
p(pi/8192) q[19];
p(pi/16384) q[20];
p(pi/32768) q[21];
p(pi/65536) q[22];
p(pi/131072) q[23];
p(pi/262144) q[24];
cx q[24],q[7];
p(-pi/262144) q[7];
cx q[24],q[7];
p(pi/262144) q[7];
cx q[23],q[7];
p(-pi/131072) q[7];
cx q[23],q[7];
p(pi/131072) q[7];
cx q[22],q[7];
p(-pi/65536) q[7];
cx q[22],q[7];
p(pi/65536) q[7];
cx q[21],q[7];
p(-pi/32768) q[7];
cx q[21],q[7];
p(pi/32768) q[7];
cx q[20],q[7];
p(-pi/16384) q[7];
cx q[20],q[7];
p(pi/16384) q[7];
cx q[19],q[7];
p(-pi/8192) q[7];
cx q[19],q[7];
p(pi/8192) q[7];
cx q[18],q[7];
p(-pi/4096) q[7];
cx q[18],q[7];
p(pi/4096) q[7];
cx q[17],q[7];
p(-pi/2048) q[7];
cx q[17],q[7];
p(pi/2048) q[7];
cx q[16],q[7];
p(-pi/1024) q[7];
cx q[16],q[7];
p(pi/1024) q[7];
cx q[15],q[7];
p(-pi/512) q[7];
cx q[15],q[7];
p(pi/512) q[7];
cx q[14],q[7];
p(-pi/256) q[7];
cx q[14],q[7];
p(pi/256) q[7];
cx q[13],q[7];
p(-pi/128) q[7];
cx q[13],q[7];
p(pi/128) q[7];
cx q[12],q[7];
p(-pi/64) q[7];
cx q[12],q[7];
p(pi/64) q[7];
cx q[11],q[7];
p(-pi/32) q[7];
cx q[11],q[7];
p(pi/32) q[7];
cx q[10],q[7];
p(-pi/16) q[7];
cx q[10],q[7];
p(pi/16) q[7];
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
p(pi/32) q[10];
p(pi/64) q[11];
p(pi/128) q[12];
p(pi/256) q[13];
p(pi/512) q[14];
p(pi/1024) q[15];
p(pi/2048) q[16];
p(pi/4096) q[17];
p(pi/8192) q[18];
p(pi/16384) q[19];
p(pi/32768) q[20];
p(pi/65536) q[21];
p(pi/131072) q[22];
p(pi/262144) q[23];
p(pi/524288) q[24];
cx q[24],q[6];
p(-pi/524288) q[6];
cx q[24],q[6];
p(pi/524288) q[6];
cx q[23],q[6];
p(-pi/262144) q[6];
cx q[23],q[6];
p(pi/262144) q[6];
cx q[22],q[6];
p(-pi/131072) q[6];
cx q[22],q[6];
p(pi/131072) q[6];
cx q[21],q[6];
p(-pi/65536) q[6];
cx q[21],q[6];
p(pi/65536) q[6];
cx q[20],q[6];
p(-pi/32768) q[6];
cx q[20],q[6];
p(pi/32768) q[6];
cx q[19],q[6];
p(-pi/16384) q[6];
cx q[19],q[6];
p(pi/16384) q[6];
cx q[18],q[6];
p(-pi/8192) q[6];
cx q[18],q[6];
p(pi/8192) q[6];
cx q[17],q[6];
p(-pi/4096) q[6];
cx q[17],q[6];
p(pi/4096) q[6];
cx q[16],q[6];
p(-pi/2048) q[6];
cx q[16],q[6];
p(pi/2048) q[6];
cx q[15],q[6];
p(-pi/1024) q[6];
cx q[15],q[6];
p(pi/1024) q[6];
cx q[14],q[6];
p(-pi/512) q[6];
cx q[14],q[6];
p(pi/512) q[6];
cx q[13],q[6];
p(-pi/256) q[6];
cx q[13],q[6];
p(pi/256) q[6];
cx q[12],q[6];
p(-pi/128) q[6];
cx q[12],q[6];
p(pi/128) q[6];
cx q[11],q[6];
p(-pi/64) q[6];
cx q[11],q[6];
p(pi/64) q[6];
cx q[10],q[6];
p(-pi/32) q[6];
cx q[10],q[6];
p(pi/32) q[6];
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
p(pi/64) q[10];
p(pi/128) q[11];
p(pi/256) q[12];
p(pi/512) q[13];
p(pi/1024) q[14];
p(pi/2048) q[15];
p(pi/4096) q[16];
p(pi/8192) q[17];
p(pi/16384) q[18];
p(pi/32768) q[19];
p(pi/65536) q[20];
p(pi/131072) q[21];
p(pi/262144) q[22];
p(pi/524288) q[23];
p(pi/1048576) q[24];
cx q[24],q[5];
p(-pi/1048576) q[5];
cx q[24],q[5];
p(pi/1048576) q[5];
cx q[23],q[5];
p(-pi/524288) q[5];
cx q[23],q[5];
p(pi/524288) q[5];
cx q[22],q[5];
p(-pi/262144) q[5];
cx q[22],q[5];
p(pi/262144) q[5];
cx q[21],q[5];
p(-pi/131072) q[5];
cx q[21],q[5];
p(pi/131072) q[5];
cx q[20],q[5];
p(-pi/65536) q[5];
cx q[20],q[5];
p(pi/65536) q[5];
cx q[19],q[5];
p(-pi/32768) q[5];
cx q[19],q[5];
p(pi/32768) q[5];
cx q[18],q[5];
p(-pi/16384) q[5];
cx q[18],q[5];
p(pi/16384) q[5];
cx q[17],q[5];
p(-pi/8192) q[5];
cx q[17],q[5];
p(pi/8192) q[5];
cx q[16],q[5];
p(-pi/4096) q[5];
cx q[16],q[5];
p(pi/4096) q[5];
cx q[15],q[5];
p(-pi/2048) q[5];
cx q[15],q[5];
p(pi/2048) q[5];
cx q[14],q[5];
p(-pi/1024) q[5];
cx q[14],q[5];
p(pi/1024) q[5];
cx q[13],q[5];
p(-pi/512) q[5];
cx q[13],q[5];
p(pi/512) q[5];
cx q[12],q[5];
p(-pi/256) q[5];
cx q[12],q[5];
p(pi/256) q[5];
cx q[11],q[5];
p(-pi/128) q[5];
cx q[11],q[5];
p(pi/128) q[5];
cx q[10],q[5];
p(-pi/64) q[5];
cx q[10],q[5];
p(pi/64) q[5];
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
p(pi/128) q[10];
p(pi/256) q[11];
p(pi/512) q[12];
p(pi/1024) q[13];
p(pi/2048) q[14];
p(pi/4096) q[15];
p(pi/8192) q[16];
p(pi/16384) q[17];
p(pi/32768) q[18];
p(pi/65536) q[19];
p(pi/131072) q[20];
p(pi/262144) q[21];
p(pi/524288) q[22];
p(pi/1048576) q[23];
p(pi/2097152) q[24];
cx q[24],q[4];
p(-pi/2097152) q[4];
cx q[24],q[4];
p(pi/2097152) q[4];
cx q[23],q[4];
p(-pi/1048576) q[4];
cx q[23],q[4];
p(pi/1048576) q[4];
cx q[22],q[4];
p(-pi/524288) q[4];
cx q[22],q[4];
p(pi/524288) q[4];
cx q[21],q[4];
p(-pi/262144) q[4];
cx q[21],q[4];
p(pi/262144) q[4];
cx q[20],q[4];
p(-pi/131072) q[4];
cx q[20],q[4];
p(pi/131072) q[4];
cx q[19],q[4];
p(-pi/65536) q[4];
cx q[19],q[4];
p(pi/65536) q[4];
cx q[18],q[4];
p(-pi/32768) q[4];
cx q[18],q[4];
p(pi/32768) q[4];
cx q[17],q[4];
p(-pi/16384) q[4];
cx q[17],q[4];
p(pi/16384) q[4];
cx q[16],q[4];
p(-pi/8192) q[4];
cx q[16],q[4];
p(pi/8192) q[4];
cx q[15],q[4];
p(-pi/4096) q[4];
cx q[15],q[4];
p(pi/4096) q[4];
cx q[14],q[4];
p(-pi/2048) q[4];
cx q[14],q[4];
p(pi/2048) q[4];
cx q[13],q[4];
p(-pi/1024) q[4];
cx q[13],q[4];
p(pi/1024) q[4];
cx q[12],q[4];
p(-pi/512) q[4];
cx q[12],q[4];
p(pi/512) q[4];
cx q[11],q[4];
p(-pi/256) q[4];
cx q[11],q[4];
p(pi/256) q[4];
cx q[10],q[4];
p(-pi/128) q[4];
cx q[10],q[4];
p(pi/128) q[4];
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
p(pi/256) q[10];
p(pi/512) q[11];
p(pi/1024) q[12];
p(pi/2048) q[13];
p(pi/4096) q[14];
p(pi/8192) q[15];
p(pi/16384) q[16];
p(pi/32768) q[17];
p(pi/65536) q[18];
p(pi/131072) q[19];
p(pi/262144) q[20];
p(pi/524288) q[21];
p(pi/1048576) q[22];
p(pi/2097152) q[23];
p(pi/4194304) q[24];
cx q[24],q[3];
p(-pi/4194304) q[3];
cx q[24],q[3];
p(pi/4194304) q[3];
cx q[23],q[3];
p(-pi/2097152) q[3];
cx q[23],q[3];
p(pi/2097152) q[3];
cx q[22],q[3];
p(-pi/1048576) q[3];
cx q[22],q[3];
p(pi/1048576) q[3];
cx q[21],q[3];
p(-pi/524288) q[3];
cx q[21],q[3];
p(pi/524288) q[3];
cx q[20],q[3];
p(-pi/262144) q[3];
cx q[20],q[3];
p(pi/262144) q[3];
cx q[19],q[3];
p(-pi/131072) q[3];
cx q[19],q[3];
p(pi/131072) q[3];
cx q[18],q[3];
p(-pi/65536) q[3];
cx q[18],q[3];
p(pi/65536) q[3];
cx q[17],q[3];
p(-pi/32768) q[3];
cx q[17],q[3];
p(pi/32768) q[3];
cx q[16],q[3];
p(-pi/16384) q[3];
cx q[16],q[3];
p(pi/16384) q[3];
cx q[15],q[3];
p(-pi/8192) q[3];
cx q[15],q[3];
p(pi/8192) q[3];
cx q[14],q[3];
p(-pi/4096) q[3];
cx q[14],q[3];
p(pi/4096) q[3];
cx q[13],q[3];
p(-pi/2048) q[3];
cx q[13],q[3];
p(pi/2048) q[3];
cx q[12],q[3];
p(-pi/1024) q[3];
cx q[12],q[3];
p(pi/1024) q[3];
cx q[11],q[3];
p(-pi/512) q[3];
cx q[11],q[3];
p(pi/512) q[3];
cx q[10],q[3];
p(-pi/256) q[3];
cx q[10],q[3];
p(pi/256) q[3];
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
p(pi/512) q[10];
p(pi/1024) q[11];
p(pi/2048) q[12];
p(pi/4096) q[13];
p(pi/8192) q[14];
p(pi/16384) q[15];
p(pi/32768) q[16];
p(pi/65536) q[17];
p(pi/131072) q[18];
p(pi/262144) q[19];
p(pi/524288) q[20];
p(pi/1048576) q[21];
p(pi/2097152) q[22];
p(pi/4194304) q[23];
p(pi/8388608) q[24];
cx q[24],q[2];
p(-pi/8388608) q[2];
cx q[24],q[2];
p(pi/8388608) q[2];
cx q[23],q[2];
p(-pi/4194304) q[2];
cx q[23],q[2];
p(pi/4194304) q[2];
cx q[22],q[2];
p(-pi/2097152) q[2];
cx q[22],q[2];
p(pi/2097152) q[2];
cx q[21],q[2];
p(-pi/1048576) q[2];
cx q[21],q[2];
p(pi/1048576) q[2];
cx q[20],q[2];
p(-pi/524288) q[2];
cx q[20],q[2];
p(pi/524288) q[2];
cx q[19],q[2];
p(-pi/262144) q[2];
cx q[19],q[2];
p(pi/262144) q[2];
cx q[18],q[2];
p(-pi/131072) q[2];
cx q[18],q[2];
p(pi/131072) q[2];
cx q[17],q[2];
p(-pi/65536) q[2];
cx q[17],q[2];
p(pi/65536) q[2];
cx q[16],q[2];
p(-pi/32768) q[2];
cx q[16],q[2];
p(pi/32768) q[2];
cx q[15],q[2];
p(-pi/16384) q[2];
cx q[15],q[2];
p(pi/16384) q[2];
cx q[14],q[2];
p(-pi/8192) q[2];
cx q[14],q[2];
p(pi/8192) q[2];
cx q[13],q[2];
p(-pi/4096) q[2];
cx q[13],q[2];
p(pi/4096) q[2];
cx q[12],q[2];
p(-pi/2048) q[2];
cx q[12],q[2];
p(pi/2048) q[2];
cx q[11],q[2];
p(-pi/1024) q[2];
cx q[11],q[2];
p(pi/1024) q[2];
cx q[10],q[2];
p(-pi/512) q[2];
cx q[10],q[2];
p(pi/512) q[2];
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
p(pi/1024) q[10];
p(pi/2048) q[11];
p(pi/4096) q[12];
p(pi/8192) q[13];
p(pi/16384) q[14];
p(pi/32768) q[15];
p(pi/65536) q[16];
p(pi/131072) q[17];
p(pi/262144) q[18];
p(pi/524288) q[19];
p(pi/1048576) q[20];
p(pi/2097152) q[21];
p(pi/4194304) q[22];
p(pi/8388608) q[23];
p(pi/16777216) q[24];
cx q[24],q[1];
p(-pi/16777216) q[1];
cx q[24],q[1];
p(pi/16777216) q[1];
cx q[23],q[1];
p(-pi/8388608) q[1];
cx q[23],q[1];
p(pi/8388608) q[1];
cx q[22],q[1];
p(-pi/4194304) q[1];
cx q[22],q[1];
p(pi/4194304) q[1];
cx q[21],q[1];
p(-pi/2097152) q[1];
cx q[21],q[1];
p(pi/2097152) q[1];
cx q[20],q[1];
p(-pi/1048576) q[1];
cx q[20],q[1];
p(pi/1048576) q[1];
cx q[19],q[1];
p(-pi/524288) q[1];
cx q[19],q[1];
p(pi/524288) q[1];
cx q[18],q[1];
p(-pi/262144) q[1];
cx q[18],q[1];
p(pi/262144) q[1];
cx q[17],q[1];
p(-pi/131072) q[1];
cx q[17],q[1];
p(pi/131072) q[1];
cx q[16],q[1];
p(-pi/65536) q[1];
cx q[16],q[1];
p(pi/65536) q[1];
cx q[15],q[1];
p(-pi/32768) q[1];
cx q[15],q[1];
p(pi/32768) q[1];
cx q[14],q[1];
p(-pi/16384) q[1];
cx q[14],q[1];
p(pi/16384) q[1];
cx q[13],q[1];
p(-pi/8192) q[1];
cx q[13],q[1];
p(pi/8192) q[1];
cx q[12],q[1];
p(-pi/4096) q[1];
cx q[12],q[1];
p(pi/4096) q[1];
cx q[11],q[1];
p(-pi/2048) q[1];
cx q[11],q[1];
p(pi/2048) q[1];
cx q[10],q[1];
p(-pi/1024) q[1];
cx q[10],q[1];
p(pi/1024) q[1];
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
p(pi/2048) q[10];
p(pi/4096) q[11];
p(pi/8192) q[12];
p(pi/16384) q[13];
p(pi/32768) q[14];
p(pi/65536) q[15];
p(pi/131072) q[16];
p(pi/262144) q[17];
p(pi/524288) q[18];
p(pi/1048576) q[19];
p(pi/2097152) q[20];
p(pi/4194304) q[21];
p(pi/8388608) q[22];
p(pi/16777216) q[23];
p(pi/33554432) q[24];
cx q[24],q[0];
p(-pi/33554432) q[0];
cx q[24],q[0];
p(pi/33554432) q[0];
cx q[23],q[0];
p(-pi/16777216) q[0];
cx q[23],q[0];
p(pi/16777216) q[0];
cx q[22],q[0];
p(-pi/8388608) q[0];
cx q[22],q[0];
p(pi/8388608) q[0];
cx q[21],q[0];
p(-pi/4194304) q[0];
cx q[21],q[0];
p(pi/4194304) q[0];
cx q[20],q[0];
p(-pi/2097152) q[0];
cx q[20],q[0];
p(pi/2097152) q[0];
cx q[19],q[0];
p(-pi/1048576) q[0];
cx q[19],q[0];
p(pi/1048576) q[0];
cx q[18],q[0];
p(-pi/524288) q[0];
cx q[18],q[0];
p(pi/524288) q[0];
cx q[17],q[0];
p(-pi/262144) q[0];
cx q[17],q[0];
p(pi/262144) q[0];
cx q[16],q[0];
p(-pi/131072) q[0];
cx q[16],q[0];
p(pi/131072) q[0];
cx q[15],q[0];
p(-pi/65536) q[0];
cx q[15],q[0];
p(pi/65536) q[0];
cx q[14],q[0];
p(-pi/32768) q[0];
cx q[14],q[0];
p(pi/32768) q[0];
cx q[13],q[0];
p(-pi/16384) q[0];
cx q[13],q[0];
p(pi/16384) q[0];
cx q[12],q[0];
p(-pi/8192) q[0];
cx q[12],q[0];
p(pi/8192) q[0];
cx q[11],q[0];
p(-pi/4096) q[0];
cx q[11],q[0];
p(pi/4096) q[0];
cx q[10],q[0];
p(-pi/2048) q[0];
cx q[10],q[0];
p(pi/2048) q[0];
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
cx q[0],q[24];
cx q[1],q[23];
cx q[2],q[22];
cx q[3],q[21];
cx q[4],q[20];
cx q[5],q[19];
cx q[6],q[18];
cx q[7],q[17];
cx q[8],q[16];
cx q[9],q[15];
cx q[10],q[14];
cx q[11],q[13];
cx q[13],q[11];
cx q[11],q[13];
cx q[14],q[10];
cx q[10],q[14];
cx q[15],q[9];
cx q[9],q[15];
cx q[16],q[8];
cx q[8],q[16];
cx q[17],q[7];
cx q[7],q[17];
cx q[18],q[6];
cx q[6],q[18];
cx q[19],q[5];
cx q[5],q[19];
cx q[20],q[4];
cx q[4],q[20];
cx q[21],q[3];
cx q[3],q[21];
cx q[22],q[2];
cx q[2],q[22];
cx q[23],q[1];
cx q[1],q[23];
cx q[24],q[0];
cx q[0],q[24];
