OPENQASM 2.0;
include "qelib1.inc";
qreg q[36];
creg c[36];
swap q[21],q[11];
cx q[6],q[7];
cx q[6],q[7];
cx q[6],q[1];
cx q[6],q[1];
cx q[7],q[1];
cx q[7],q[1];
cx q[6],q[13];
cx q[6],q[13];
cx q[6],q[8];
cx q[6],q[8];
cx q[6],q[0];
cx q[6],q[0];
swap q[21],q[20];
swap q[32],q[24];
cx q[6],q[12];
swap q[18],q[6];
swap q[33],q[32];
cx q[18],q[12];
cx q[18],q[6];
cx q[18],q[6];
swap q[15],q[3];
cx q[7],q[13];
swap q[26],q[15];
cx q[7],q[13];
swap q[25],q[18];
cx q[7],q[8];
cx q[7],q[8];
cx q[7],q[0];
cx q[25],q[20];
cx q[7],q[0];
cx q[25],q[20];
cx q[7],q[12];
cx q[25],q[24];
cx q[7],q[12];
cx q[25],q[24];
cx q[25],q[32];
swap q[20],q[14];
cx q[7],q[6];
cx q[7],q[6];
cx q[1],q[13];
cx q[1],q[13];
cx q[1],q[8];
cx q[7],q[14];
cx q[7],q[14];
cx q[25],q[32];
cx q[25],q[26];
cx q[25],q[26];
cx q[25],q[33];
cx q[25],q[33];
cx q[25],q[18];
swap q[12],q[7];
swap q[32],q[31];
cx q[1],q[8];
cx q[1],q[0];
cx q[12],q[24];
cx q[13],q[8];
cx q[13],q[8];
cx q[1],q[0];
swap q[24],q[18];
cx q[1],q[7];
swap q[31],q[26];
cx q[13],q[0];
cx q[13],q[0];
cx q[25],q[24];
cx q[1],q[7];
cx q[13],q[7];
swap q[9],q[2];
cx q[13],q[7];
cx q[1],q[6];
cx q[1],q[6];
cx q[13],q[6];
swap q[21],q[9];
swap q[31],q[24];
swap q[33],q[32];
swap q[32],q[30];
cx q[8],q[0];
cx q[8],q[0];
cx q[1],q[14];
cx q[1],q[14];
cx q[12],q[18];
swap q[18],q[12];
swap q[26],q[25];
cx q[13],q[6];
cx q[8],q[7];
cx q[8],q[7];
cx q[0],q[7];
cx q[8],q[6];
cx q[13],q[14];
cx q[13],q[14];
cx q[18],q[25];
cx q[18],q[25];
cx q[18],q[24];
cx q[18],q[24];
cx q[26],q[21];
cx q[26],q[21];
swap q[26],q[21];
cx q[8],q[6];
swap q[25],q[18];
cx q[0],q[7];
cx q[8],q[14];
cx q[8],q[14];
cx q[0],q[6];
cx q[0],q[6];
swap q[18],q[0];
swap q[22],q[9];
cx q[18],q[14];
swap q[27],q[22];
cx q[7],q[6];
cx q[7],q[6];
cx q[18],q[14];
cx q[7],q[14];
cx q[7],q[14];
cx q[25],q[30];
cx q[25],q[30];
swap q[28],q[16];
cx q[1],q[12];
cx q[25],q[31];
swap q[11],q[9];
cx q[21],q[20];
cx q[21],q[20];
cx q[21],q[27];
cx q[25],q[31];
cx q[25],q[26];
cx q[25],q[26];
cx q[21],q[27];
cx q[1],q[12];
cx q[13],q[12];
cx q[13],q[12];
cx q[1],q[0];
cx q[1],q[0];
cx q[8],q[12];
cx q[8],q[12];
cx q[13],q[0];
cx q[6],q[14];
cx q[13],q[0];
cx q[18],q[12];
cx q[18],q[12];
cx q[8],q[0];
cx q[8],q[0];
cx q[7],q[12];
cx q[7],q[12];
cx q[25],q[20];
cx q[25],q[20];
cx q[6],q[14];
cx q[25],q[27];
cx q[25],q[27];
cx q[18],q[0];
cx q[6],q[12];
cx q[21],q[28];
cx q[6],q[12];
cx q[21],q[28];
cx q[25],q[28];
swap q[8],q[1];
cx q[18],q[0];
cx q[21],q[32];
cx q[21],q[32];
cx q[21],q[19];
cx q[25],q[28];
cx q[21],q[19];
cx q[25],q[32];
cx q[25],q[32];
cx q[21],q[9];
cx q[14],q[12];
cx q[21],q[9];
cx q[21],q[33];
cx q[7],q[0];
cx q[25],q[19];
cx q[25],q[19];
cx q[21],q[33];
cx q[14],q[12];
cx q[21],q[15];
cx q[7],q[0];
swap q[26],q[25];
cx q[6],q[0];
cx q[6],q[0];
cx q[21],q[15];
cx q[21],q[22];
swap q[13],q[8];
cx q[21],q[22];
swap q[21],q[9];
cx q[13],q[24];
cx q[13],q[24];
swap q[24],q[13];
cx q[26],q[21];
cx q[9],q[16];
swap q[31],q[24];
cx q[8],q[13];
cx q[8],q[13];
cx q[1],q[13];
cx q[9],q[16];
swap q[34],q[28];
swap q[27],q[26];
cx q[1],q[13];
cx q[27],q[21];
cx q[18],q[13];
cx q[31],q[30];
cx q[31],q[30];
cx q[9],q[3];
cx q[9],q[3];
cx q[31],q[24];
cx q[14],q[0];
cx q[27],q[33];
swap q[10],q[3];
cx q[31],q[24];
cx q[14],q[0];
cx q[18],q[13];
cx q[31],q[25];
cx q[27],q[33];
swap q[27],q[22];
cx q[31],q[25];
cx q[7],q[13];
cx q[7],q[13];
cx q[9],q[3];
cx q[6],q[13];
cx q[22],q[15];
swap q[30],q[25];
cx q[22],q[15];
cx q[9],q[3];
cx q[9],q[2];
cx q[9],q[2];
cx q[9],q[4];
cx q[9],q[4];
cx q[6],q[13];
cx q[31],q[20];
cx q[31],q[20];
cx q[12],q[0];
cx q[12],q[0];
cx q[14],q[13];
cx q[14],q[13];
cx q[12],q[13];
cx q[31],q[26];
cx q[12],q[13];
cx q[31],q[26];
cx q[31],q[34];
cx q[31],q[34];
swap q[17],q[3];
cx q[0],q[13];
cx q[0],q[13];
cx q[31],q[32];
cx q[31],q[32];
cx q[31],q[19];
cx q[31],q[19];
cx q[22],q[27];
cx q[22],q[27];
cx q[22],q[16];
swap q[22],q[10];
swap q[25],q[13];
cx q[10],q[16];
cx q[8],q[13];
cx q[8],q[13];
swap q[18],q[8];
cx q[31],q[21];
swap q[34],q[31];
cx q[34],q[21];
cx q[34],q[33];
cx q[34],q[33];
cx q[10],q[22];
cx q[10],q[22];
cx q[10],q[17];
cx q[1],q[13];
cx q[1],q[13];
cx q[8],q[13];
cx q[8],q[13];
cx q[7],q[13];
cx q[18],q[24];
cx q[10],q[17];
cx q[18],q[24];
cx q[7],q[13];
cx q[6],q[13];
cx q[6],q[13];
swap q[24],q[6];
swap q[34],q[22];
swap q[25],q[18];
cx q[1],q[6];
cx q[14],q[13];
cx q[14],q[13];
cx q[1],q[6];
cx q[8],q[6];
cx q[25],q[30];
cx q[10],q[2];
cx q[25],q[30];
cx q[12],q[13];
cx q[22],q[15];
cx q[10],q[2];
cx q[12],q[13];
cx q[8],q[6];
cx q[25],q[20];
cx q[10],q[4];
cx q[25],q[20];
cx q[7],q[6];
cx q[25],q[26];
cx q[22],q[15];
cx q[10],q[4];
cx q[25],q[26];
cx q[0],q[13];
swap q[9],q[2];
cx q[22],q[27];
cx q[22],q[27];
cx q[22],q[16];
cx q[22],q[16];
cx q[25],q[31];
cx q[22],q[34];
cx q[22],q[34];
cx q[22],q[17];
cx q[0],q[13];
cx q[7],q[6];
cx q[25],q[31];
cx q[24],q[6];
cx q[24],q[6];
cx q[25],q[32];
cx q[14],q[6];
cx q[25],q[32];
cx q[25],q[19];
swap q[28],q[16];
cx q[25],q[19];
cx q[18],q[13];
cx q[22],q[17];
cx q[18],q[13];
cx q[14],q[6];
swap q[26],q[25];
swap q[17],q[16];
cx q[26],q[21];
cx q[26],q[21];
cx q[12],q[6];
cx q[12],q[6];
cx q[26],q[33];
cx q[26],q[33];
cx q[0],q[6];
cx q[26],q[15];
cx q[0],q[6];
cx q[18],q[6];
cx q[18],q[6];
cx q[13],q[6];
cx q[26],q[15];
cx q[26],q[27];
swap q[30],q[24];
cx q[26],q[27];
cx q[13],q[6];
cx q[26],q[28];
cx q[26],q[28];
cx q[26],q[34];
swap q[12],q[1];
swap q[15],q[9];
cx q[12],q[24];
cx q[22],q[15];
cx q[22],q[15];
swap q[26],q[21];
cx q[12],q[24];
cx q[21],q[34];
swap q[9],q[4];
cx q[21],q[16];
swap q[24],q[12];
cx q[8],q[12];
cx q[21],q[16];
cx q[21],q[15];
cx q[22],q[9];
cx q[22],q[9];
cx q[8],q[12];
swap q[33],q[27];
cx q[7],q[12];
cx q[7],q[12];
swap q[16],q[4];
swap q[25],q[24];
cx q[25],q[20];
cx q[25],q[20];
cx q[25],q[24];
cx q[25],q[24];
cx q[25],q[31];
cx q[30],q[12];
cx q[21],q[15];
cx q[25],q[31];
cx q[8],q[20];
swap q[21],q[8];
cx q[21],q[20];
cx q[25],q[32];
cx q[25],q[32];
cx q[8],q[9];
cx q[30],q[12];
cx q[8],q[9];
cx q[25],q[19];
swap q[22],q[16];
cx q[25],q[19];
cx q[14],q[12];
cx q[14],q[12];
cx q[25],q[26];
cx q[25],q[26];
swap q[25],q[21];
swap q[9],q[4];
swap q[10],q[4];
cx q[21],q[27];
cx q[7],q[20];
cx q[1],q[12];
cx q[21],q[27];
cx q[25],q[24];
cx q[21],q[22];
cx q[21],q[22];
cx q[7],q[20];
cx q[30],q[20];
cx q[30],q[20];
cx q[1],q[12];
cx q[0],q[12];
cx q[25],q[24];
cx q[14],q[20];
cx q[14],q[20];
cx q[0],q[12];
cx q[21],q[33];
cx q[25],q[31];
cx q[25],q[31];
swap q[20],q[7];
cx q[20],q[24];
cx q[20],q[24];
cx q[30],q[24];
cx q[21],q[33];
cx q[20],q[31];
cx q[20],q[31];
cx q[1],q[7];
cx q[21],q[28];
cx q[25],q[32];
cx q[18],q[12];
cx q[1],q[7];
cx q[18],q[12];
cx q[0],q[7];
cx q[0],q[7];
cx q[13],q[12];
swap q[8],q[1];
cx q[25],q[32];
cx q[13],q[12];
cx q[18],q[7];
cx q[30],q[24];
cx q[6],q[12];
cx q[21],q[28];
cx q[20],q[32];
cx q[30],q[31];
cx q[6],q[12];
cx q[18],q[7];
cx q[21],q[34];
cx q[21],q[34];
cx q[30],q[31];
cx q[20],q[32];
cx q[25],q[19];
cx q[21],q[9];
cx q[25],q[19];
cx q[20],q[19];
cx q[30],q[32];
swap q[32],q[25];
cx q[32],q[26];
cx q[13],q[7];
cx q[13],q[7];
cx q[6],q[7];
cx q[6],q[7];
cx q[32],q[26];
cx q[12],q[7];
swap q[24],q[6];
cx q[21],q[9];
cx q[21],q[15];
cx q[21],q[15];
swap q[33],q[32];
cx q[30],q[25];
cx q[14],q[6];
cx q[33],q[27];
swap q[30],q[25];
cx q[12],q[7];
cx q[14],q[6];
cx q[33],q[27];
cx q[21],q[10];
swap q[24],q[14];
cx q[33],q[22];
cx q[20],q[19];
cx q[20],q[26];
cx q[8],q[6];
cx q[24],q[31];
cx q[33],q[22];
cx q[25],q[19];
cx q[25],q[19];
cx q[33],q[32];
cx q[21],q[10];
swap q[16],q[9];
cx q[8],q[6];
cx q[0],q[6];
cx q[20],q[26];
cx q[20],q[27];
cx q[24],q[31];
cx q[20],q[27];
cx q[33],q[32];
cx q[33],q[28];
cx q[24],q[30];
cx q[33],q[28];
cx q[24],q[30];
cx q[0],q[6];
cx q[33],q[34];
cx q[33],q[34];
cx q[25],q[26];
cx q[24],q[19];
cx q[24],q[19];
cx q[18],q[6];
swap q[34],q[16];
cx q[18],q[6];
cx q[13],q[6];
cx q[13],q[6];
cx q[33],q[34];
cx q[25],q[26];
cx q[20],q[22];
swap q[31],q[24];
cx q[20],q[22];
cx q[31],q[26];
cx q[14],q[6];
cx q[20],q[32];
swap q[27],q[25];
cx q[14],q[6];
cx q[12],q[6];
cx q[27],q[25];
cx q[20],q[32];
cx q[12],q[6];
cx q[7],q[6];
cx q[33],q[34];
cx q[7],q[6];
cx q[31],q[26];
swap q[24],q[18];
cx q[20],q[28];
swap q[17],q[10];
cx q[8],q[18];
swap q[33],q[22];
cx q[27],q[25];
cx q[8],q[18];
cx q[0],q[18];
cx q[22],q[15];
cx q[0],q[18];
cx q[24],q[18];
cx q[27],q[33];
cx q[24],q[18];
cx q[27],q[33];
cx q[27],q[32];
cx q[20],q[28];
cx q[13],q[18];
cx q[13],q[18];
cx q[27],q[32];
cx q[31],q[25];
cx q[27],q[28];
cx q[31],q[25];
cx q[27],q[28];
swap q[23],q[17];
cx q[14],q[18];
cx q[14],q[18];
cx q[12],q[18];
cx q[12],q[18];
cx q[31],q[33];
cx q[31],q[33];
cx q[31],q[32];
cx q[31],q[32];
cx q[20],q[16];
cx q[20],q[16];
cx q[7],q[18];
cx q[27],q[16];
cx q[20],q[34];
swap q[30],q[24];
cx q[27],q[16];
cx q[20],q[34];
swap q[33],q[31];
cx q[7],q[18];
cx q[6],q[18];
cx q[27],q[34];
cx q[6],q[18];
swap q[23],q[16];
cx q[27],q[34];
swap q[14],q[8];
cx q[14],q[24];
swap q[6],q[0];
cx q[14],q[24];
cx q[6],q[24];
cx q[33],q[28];
cx q[6],q[24];
cx q[33],q[28];
cx q[33],q[23];
swap q[20],q[14];
cx q[30],q[24];
cx q[33],q[23];
swap q[12],q[6];
cx q[30],q[24];
cx q[20],q[19];
cx q[20],q[19];
cx q[12],q[19];
cx q[12],q[19];
cx q[33],q[34];
cx q[33],q[34];
cx q[22],q[15];
cx q[30],q[19];
cx q[22],q[16];
cx q[14],q[15];
swap q[33],q[23];
cx q[22],q[16];
cx q[20],q[26];
cx q[20],q[26];
swap q[24],q[12];
cx q[20],q[25];
cx q[13],q[12];
cx q[20],q[25];
cx q[13],q[12];
swap q[33],q[28];
cx q[24],q[26];
cx q[14],q[15];
cx q[30],q[19];
cx q[27],q[15];
swap q[19],q[13];
cx q[27],q[15];
cx q[19],q[13];
cx q[23],q[15];
cx q[23],q[15];
cx q[19],q[13];
swap q[31],q[24];
cx q[8],q[12];
cx q[31],q[26];
cx q[31],q[25];
cx q[20],q[24];
cx q[20],q[24];
cx q[31],q[25];
cx q[8],q[12];
swap q[28],q[21];
cx q[8],q[13];
cx q[20],q[32];
cx q[6],q[12];
cx q[20],q[32];
cx q[31],q[24];
cx q[30],q[26];
cx q[6],q[12];
cx q[7],q[12];
cx q[31],q[24];
cx q[8],q[13];
cx q[6],q[13];
cx q[7],q[12];
cx q[20],q[33];
cx q[30],q[26];
cx q[6],q[13];
swap q[15],q[14];
cx q[0],q[12];
cx q[19],q[26];
cx q[19],q[26];
swap q[26],q[8];
cx q[0],q[12];
cx q[18],q[12];
cx q[30],q[25];
cx q[7],q[13];
cx q[15],q[16];
cx q[31],q[32];
cx q[30],q[25];
cx q[18],q[12];
cx q[15],q[16];
cx q[27],q[16];
cx q[26],q[8];
cx q[7],q[13];
cx q[27],q[16];
cx q[20],q[33];
cx q[23],q[16];
cx q[30],q[24];
cx q[23],q[16];
cx q[30],q[24];
cx q[31],q[32];
cx q[20],q[21];
cx q[30],q[32];
cx q[20],q[21];
cx q[20],q[34];
cx q[26],q[8];
swap q[31],q[27];
cx q[0],q[13];
cx q[0],q[13];
cx q[20],q[34];
cx q[27],q[33];
cx q[18],q[13];
cx q[18],q[13];
cx q[27],q[33];
cx q[27],q[21];
cx q[12],q[13];
swap q[31],q[30];
cx q[27],q[21];
cx q[19],q[25];
cx q[6],q[8];
cx q[6],q[8];
cx q[7],q[8];
cx q[19],q[25];
cx q[12],q[13];
cx q[7],q[8];
cx q[0],q[8];
cx q[0],q[8];
cx q[26],q[25];
cx q[26],q[25];
cx q[19],q[24];
cx q[31],q[32];
cx q[19],q[24];
cx q[27],q[34];
cx q[27],q[34];
cx q[26],q[24];
cx q[26],q[24];
swap q[18],q[6];
cx q[31],q[33];
cx q[18],q[25];
cx q[20],q[14];
cx q[6],q[8];
swap q[33],q[31];
cx q[20],q[14];
cx q[20],q[16];
cx q[20],q[16];
swap q[12],q[0];
swap q[25],q[18];
cx q[25],q[18];
cx q[27],q[14];
cx q[27],q[14];
cx q[6],q[8];
cx q[27],q[16];
cx q[33],q[31];
cx q[33],q[21];
cx q[0],q[8];
cx q[7],q[18];
cx q[0],q[8];
cx q[25],q[24];
cx q[7],q[18];
cx q[19],q[32];
swap q[33],q[28];
cx q[13],q[8];
cx q[12],q[18];
cx q[25],q[24];
cx q[27],q[16];
cx q[12],q[18];
swap q[33],q[19];
swap q[22],q[14];
cx q[6],q[18];
cx q[33],q[32];
cx q[28],q[21];
cx q[28],q[34];
cx q[6],q[18];
cx q[26],q[32];
cx q[26],q[32];
cx q[25],q[32];
cx q[0],q[18];
swap q[18],q[7];
cx q[33],q[31];
cx q[33],q[31];
cx q[26],q[31];
cx q[28],q[34];
swap q[23],q[16];
swap q[24],q[12];
cx q[18],q[12];
cx q[13],q[8];
cx q[0],q[7];
cx q[25],q[32];
cx q[18],q[12];
cx q[26],q[31];
cx q[33],q[21];
cx q[13],q[7];
cx q[33],q[21];
cx q[25],q[31];
cx q[25],q[31];
swap q[29],q[23];
cx q[24],q[12];
cx q[26],q[21];
cx q[18],q[32];
cx q[28],q[22];
swap q[32],q[18];
cx q[26],q[21];
cx q[33],q[34];
cx q[33],q[34];
cx q[13],q[7];
cx q[24],q[12];
swap q[25],q[21];
cx q[26],q[34];
cx q[6],q[12];
cx q[6],q[12];
cx q[0],q[12];
cx q[28],q[22];
cx q[0],q[12];
cx q[28],q[29];
cx q[13],q[12];
cx q[32],q[18];
cx q[8],q[7];
swap q[34],q[27];
cx q[8],q[7];
cx q[24],q[18];
cx q[24],q[18];
cx q[32],q[31];
cx q[6],q[18];
cx q[32],q[31];
cx q[21],q[25];
cx q[13],q[12];
cx q[21],q[25];
swap q[31],q[24];
cx q[28],q[29];
cx q[8],q[12];
cx q[26],q[27];
cx q[8],q[12];
cx q[7],q[12];
cx q[32],q[25];
cx q[21],q[27];
cx q[21],q[27];
cx q[7],q[12];
swap q[34],q[29];
swap q[18],q[6];
cx q[31],q[24];
cx q[31],q[24];
cx q[33],q[22];
cx q[33],q[22];
cx q[32],q[25];
swap q[7],q[0];
cx q[33],q[34];
cx q[18],q[6];
swap q[32],q[27];
cx q[31],q[25];
cx q[7],q[6];
cx q[7],q[6];
cx q[13],q[6];
cx q[33],q[34];
cx q[13],q[6];
swap q[22],q[21];
cx q[27],q[32];
cx q[18],q[24];
cx q[27],q[32];
cx q[8],q[6];
cx q[18],q[24];
cx q[8],q[6];
cx q[0],q[6];
cx q[0],q[6];
swap q[24],q[14];
swap q[14],q[7];
cx q[31],q[25];
cx q[18],q[25];
cx q[12],q[6];
cx q[31],q[32];
cx q[26],q[21];
cx q[12],q[6];
cx q[31],q[32];
cx q[26],q[21];
swap q[25],q[18];
cx q[22],q[21];
cx q[26],q[34];
cx q[22],q[21];
cx q[25],q[18];
cx q[14],q[7];
cx q[14],q[7];
cx q[13],q[7];
cx q[13],q[7];
cx q[8],q[7];
cx q[26],q[34];
cx q[22],q[34];
cx q[8],q[7];
cx q[14],q[18];
cx q[22],q[34];
cx q[0],q[7];
cx q[14],q[18];
swap q[33],q[25];
cx q[13],q[18];
cx q[27],q[21];
cx q[13],q[18];
cx q[27],q[21];
swap q[20],q[13];
cx q[33],q[32];
cx q[33],q[32];
cx q[14],q[32];
swap q[22],q[14];
cx q[0],q[7];
cx q[22],q[32];
swap q[32],q[14];
cx q[8],q[18];
cx q[8],q[18];
cx q[12],q[7];
cx q[12],q[7];
cx q[6],q[7];
cx q[27],q[34];
cx q[27],q[34];
cx q[31],q[21];
cx q[27],q[26];
cx q[0],q[18];
cx q[26],q[27];
cx q[6],q[7];
cx q[31],q[21];
cx q[0],q[18];
cx q[33],q[21];
swap q[1],q[0];
cx q[33],q[21];
cx q[31],q[34];
cx q[12],q[18];
cx q[12],q[18];
cx q[31],q[34];
cx q[33],q[34];
cx q[6],q[18];
cx q[33],q[34];
cx q[6],q[18];
cx q[20],q[14];
cx q[20],q[14];
cx q[22],q[21];
cx q[22],q[21];
cx q[8],q[14];
cx q[8],q[14];
swap q[34],q[22];
cx q[33],q[28];
cx q[31],q[25];
cx q[1],q[14];
cx q[25],q[31];
cx q[31],q[25];
cx q[1],q[14];
cx q[12],q[14];
cx q[20],q[21];
cx q[7],q[18];
cx q[20],q[21];
cx q[7],q[18];
cx q[8],q[21];
cx q[12],q[14];
cx q[28],q[33];
swap q[3],q[1];
cx q[34],q[22];
cx q[27],q[26];
swap q[14],q[12];
cx q[34],q[22];
swap q[21],q[8];
cx q[34],q[29];
cx q[29],q[34];
cx q[34],q[29];
cx q[21],q[8];
cx q[3],q[8];
cx q[6],q[12];
cx q[20],q[22];
cx q[6],q[12];
cx q[3],q[8];
cx q[33],q[28];
cx q[7],q[12];
cx q[7],q[12];
cx q[20],q[22];
cx q[21],q[22];
cx q[21],q[22];
cx q[21],q[16];
cx q[20],q[13];
swap q[9],q[3];
cx q[9],q[22];
swap q[31],q[30];
swap q[31],q[24];
swap q[25],q[24];
cx q[9],q[22];
cx q[18],q[12];
swap q[26],q[25];
cx q[18],q[12];
cx q[14],q[8];
cx q[16],q[21];
cx q[14],q[8];
cx q[21],q[16];
cx q[6],q[8];
cx q[6],q[8];
swap q[26],q[22];
cx q[14],q[26];
swap q[18],q[6];
cx q[14],q[26];
swap q[26],q[19];
cx q[7],q[8];
cx q[7],q[8];
cx q[6],q[8];
swap q[20],q[7];
cx q[18],q[19];
cx q[14],q[15];
swap q[3],q[2];
cx q[18],q[19];
cx q[15],q[14];
cx q[20],q[19];
cx q[9],q[22];
cx q[18],q[31];
cx q[22],q[9];
cx q[13],q[7];
swap q[21],q[3];
cx q[7],q[13];
cx q[31],q[18];
cx q[18],q[31];
cx q[20],q[19];
cx q[6],q[8];
cx q[12],q[8];
cx q[12],q[8];
cx q[20],q[26];
cx q[6],q[19];
cx q[6],q[19];
cx q[12],q[19];
cx q[12],q[19];
cx q[6],q[0];
cx q[9],q[22];
cx q[14],q[15];
cx q[0],q[6];
cx q[6],q[0];
cx q[8],q[19];
cx q[8],q[19];
cx q[19],q[21];
cx q[8],q[4];
cx q[4],q[8];
cx q[21],q[19];
cx q[12],q[2];
cx q[8],q[4];
cx q[19],q[21];
cx q[26],q[20];
cx q[20],q[26];
cx q[2],q[12];
cx q[12],q[2];
