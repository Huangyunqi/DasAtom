OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
cx q[7],q[8];
swap q[41],q[25];
cx q[7],q[8];
cx q[7],q[1];
cx q[7],q[1];
swap q[25],q[16];
cx q[8],q[1];
swap q[26],q[23];
cx q[8],q[1];
swap q[46],q[45];
swap q[45],q[38];
cx q[7],q[15];
swap q[38],q[29];
cx q[7],q[15];
cx q[8],q[15];
cx q[7],q[9];
swap q[29],q[22];
cx q[7],q[9];
cx q[7],q[0];
swap q[48],q[27];
swap q[27],q[18];
cx q[8],q[15];
swap q[18],q[10];
cx q[1],q[15];
cx q[1],q[15];
cx q[8],q[9];
cx q[8],q[9];
cx q[7],q[0];
swap q[17],q[2];
cx q[7],q[14];
cx q[7],q[14];
swap q[33],q[32];
swap q[32],q[30];
swap q[30],q[28];
cx q[1],q[9];
cx q[8],q[0];
swap q[42],q[29];
cx q[1],q[9];
cx q[8],q[0];
cx q[15],q[9];
cx q[1],q[0];
cx q[8],q[14];
cx q[7],q[21];
cx q[15],q[9];
cx q[1],q[0];
cx q[8],q[14];
cx q[7],q[21];
cx q[15],q[0];
cx q[1],q[14];
cx q[8],q[21];
cx q[7],q[16];
cx q[15],q[0];
cx q[1],q[14];
cx q[8],q[21];
cx q[7],q[16];
swap q[29],q[21];
cx q[9],q[0];
cx q[8],q[16];
cx q[15],q[14];
swap q[9],q[1];
cx q[15],q[14];
swap q[29],q[24];
cx q[9],q[24];
cx q[7],q[23];
cx q[1],q[0];
swap q[24],q[17];
cx q[1],q[14];
cx q[1],q[14];
swap q[30],q[29];
cx q[9],q[17];
cx q[0],q[14];
cx q[8],q[16];
cx q[7],q[23];
cx q[15],q[17];
cx q[9],q[16];
cx q[0],q[14];
cx q[8],q[23];
cx q[7],q[22];
cx q[15],q[17];
cx q[9],q[16];
swap q[29],q[14];
swap q[17],q[3];
cx q[7],q[22];
swap q[37],q[35];
cx q[1],q[3];
cx q[15],q[16];
cx q[7],q[10];
cx q[8],q[23];
cx q[9],q[23];
cx q[8],q[22];
cx q[1],q[3];
cx q[15],q[16];
swap q[31],q[29];
cx q[7],q[10];
cx q[0],q[3];
cx q[9],q[23];
cx q[8],q[22];
swap q[31],q[17];
cx q[7],q[2];
cx q[7],q[2];
swap q[35],q[29];
cx q[1],q[16];
cx q[15],q[23];
cx q[9],q[22];
cx q[8],q[10];
cx q[0],q[3];
cx q[1],q[16];
cx q[15],q[23];
cx q[9],q[22];
cx q[8],q[10];
cx q[17],q[3];
cx q[15],q[22];
cx q[0],q[16];
cx q[9],q[10];
cx q[15],q[22];
cx q[8],q[2];
swap q[28],q[22];
cx q[17],q[3];
cx q[9],q[10];
cx q[7],q[22];
cx q[0],q[16];
swap q[29],q[28];
cx q[8],q[2];
cx q[17],q[16];
cx q[15],q[10];
cx q[7],q[22];
cx q[9],q[2];
cx q[8],q[22];
cx q[7],q[21];
cx q[17],q[16];
cx q[15],q[10];
swap q[35],q[29];
cx q[9],q[2];
cx q[3],q[16];
cx q[8],q[22];
cx q[7],q[21];
cx q[3],q[16];
cx q[15],q[2];
cx q[9],q[22];
cx q[8],q[21];
cx q[7],q[14];
swap q[23],q[2];
cx q[7],q[14];
swap q[36],q[35];
cx q[1],q[2];
cx q[15],q[23];
cx q[1],q[2];
swap q[36],q[23];
cx q[0],q[2];
cx q[0],q[2];
cx q[17],q[2];
swap q[21],q[7];
cx q[17],q[2];
cx q[3],q[2];
swap q[38],q[30];
cx q[9],q[22];
swap q[39],q[37];
cx q[8],q[7];
cx q[21],q[28];
cx q[3],q[2];
cx q[15],q[22];
swap q[25],q[24];
swap q[9],q[1];
cx q[15],q[22];
cx q[21],q[28];
swap q[36],q[30];
cx q[1],q[7];
swap q[25],q[17];
cx q[8],q[14];
cx q[1],q[7];
cx q[21],q[29];
cx q[9],q[23];
cx q[9],q[23];
cx q[9],q[10];
cx q[15],q[7];
swap q[29],q[28];
cx q[9],q[10];
cx q[15],q[7];
cx q[8],q[14];
swap q[23],q[17];
cx q[1],q[14];
cx q[1],q[14];
swap q[44],q[42];
swap q[26],q[25];
cx q[9],q[30];
cx q[9],q[30];
cx q[16],q[2];
cx q[21],q[28];
swap q[31],q[25];
cx q[9],q[22];
cx q[16],q[2];
cx q[21],q[36];
cx q[8],q[29];
cx q[9],q[22];
cx q[8],q[29];
cx q[9],q[7];
swap q[31],q[22];
cx q[9],q[7];
swap q[44],q[43];
swap q[3],q[0];
cx q[21],q[36];
swap q[29],q[21];
cx q[3],q[17];
cx q[3],q[17];
swap q[39],q[31];
swap q[7],q[1];
cx q[26],q[17];
cx q[29],q[37];
swap q[46],q[39];
cx q[3],q[10];
cx q[15],q[14];
cx q[29],q[37];
swap q[1],q[0];
cx q[26],q[17];
cx q[15],q[14];
cx q[3],q[10];
cx q[29],q[24];
swap q[39],q[38];
cx q[1],q[17];
swap q[28],q[22];
cx q[26],q[10];
cx q[1],q[17];
swap q[39],q[33];
cx q[16],q[17];
cx q[7],q[21];
cx q[8],q[22];
swap q[11],q[3];
cx q[16],q[17];
cx q[7],q[21];
swap q[39],q[37];
cx q[8],q[22];
cx q[2],q[17];
cx q[15],q[21];
cx q[26],q[10];
cx q[7],q[22];
swap q[47],q[39];
cx q[1],q[10];
cx q[15],q[21];
cx q[1],q[10];
swap q[30],q[24];
cx q[9],q[14];
swap q[46],q[33];
cx q[16],q[10];
cx q[29],q[30];
cx q[2],q[17];
swap q[36],q[28];
cx q[11],q[24];
cx q[11],q[24];
swap q[14],q[8];
cx q[26],q[24];
swap q[47],q[46];
cx q[26],q[24];
cx q[14],q[28];
swap q[46],q[44];
cx q[16],q[10];
cx q[14],q[28];
cx q[9],q[8];
cx q[29],q[23];
swap q[33],q[19];
cx q[7],q[22];
cx q[2],q[10];
swap q[44],q[42];
cx q[11],q[19];
cx q[15],q[22];
cx q[11],q[19];
cx q[15],q[22];
swap q[1],q[0];
cx q[26],q[19];
cx q[29],q[23];
swap q[11],q[3];
cx q[26],q[19];
cx q[7],q[28];
cx q[7],q[28];
swap q[24],q[23];
cx q[3],q[1];
swap q[26],q[19];
cx q[3],q[1];
cx q[29],q[44];
cx q[3],q[8];
cx q[29],q[44];
swap q[19],q[18];
cx q[9],q[21];
cx q[2],q[10];
cx q[15],q[28];
swap q[44],q[39];
cx q[3],q[8];
swap q[23],q[21];
swap q[26],q[25];
swap q[4],q[1];
cx q[29],q[36];
swap q[40],q[39];
swap q[21],q[14];
cx q[18],q[4];
cx q[9],q[23];
cx q[18],q[4];
cx q[29],q[36];
swap q[46],q[40];
cx q[0],q[14];
cx q[17],q[10];
cx q[29],q[43];
cx q[0],q[14];
cx q[17],q[10];
cx q[29],q[43];
cx q[16],q[14];
cx q[29],q[31];
swap q[11],q[3];
cx q[21],q[42];
cx q[16],q[14];
cx q[29],q[31];
swap q[1],q[0];
cx q[21],q[42];
swap q[10],q[1];
cx q[21],q[30];
cx q[21],q[30];
cx q[10],q[25];
cx q[29],q[35];
cx q[10],q[25];
cx q[29],q[35];
cx q[2],q[14];
swap q[39],q[30];
cx q[10],q[4];
cx q[2],q[14];
swap q[42],q[35];
swap q[34],q[26];
cx q[16],q[25];
swap q[46],q[43];
cx q[11],q[23];
cx q[9],q[22];
cx q[16],q[25];
swap q[43],q[36];
cx q[10],q[4];
cx q[11],q[23];
cx q[9],q[22];
swap q[39],q[26];
cx q[15],q[28];
cx q[16],q[4];
swap q[43],q[42];
swap q[37],q[29];
cx q[16],q[4];
swap q[46],q[32];
swap q[11],q[8];
cx q[37],q[46];
cx q[18],q[11];
cx q[8],q[22];
cx q[37],q[46];
cx q[18],q[11];
cx q[8],q[22];
cx q[37],q[45];
swap q[32],q[31];
cx q[10],q[11];
cx q[37],q[45];
cx q[10],q[11];
swap q[28],q[15];
cx q[37],q[38];
swap q[25],q[12];
cx q[9],q[15];
cx q[37],q[38];
cx q[9],q[15];
cx q[37],q[29];
swap q[34],q[27];
swap q[40],q[34];
swap q[48],q[47];
cx q[17],q[14];
swap q[47],q[38];
cx q[17],q[14];
cx q[1],q[14];
swap q[12],q[4];
cx q[21],q[24];
cx q[21],q[24];
swap q[48],q[46];
cx q[2],q[4];
cx q[21],q[36];
swap q[24],q[18];
cx q[2],q[4];
cx q[21],q[36];
cx q[8],q[15];
cx q[37],q[29];
swap q[33],q[32];
swap q[3],q[2];
cx q[24],q[23];
cx q[8],q[15];
cx q[37],q[44];
cx q[24],q[23];
cx q[37],q[44];
swap q[29],q[21];
cx q[17],q[4];
swap q[45],q[33];
cx q[3],q[12];
cx q[29],q[42];
cx q[16],q[11];
swap q[14],q[7];
cx q[10],q[23];
swap q[35],q[28];
cx q[1],q[7];
cx q[17],q[4];
cx q[37],q[30];
swap q[48],q[33];
cx q[14],q[28];
cx q[3],q[12];
cx q[37],q[30];
cx q[14],q[28];
cx q[10],q[23];
cx q[35],q[28];
cx q[16],q[11];
cx q[37],q[39];
swap q[2],q[1];
cx q[35],q[28];
cx q[17],q[12];
cx q[37],q[39];
cx q[24],q[22];
swap q[39],q[33];
cx q[24],q[22];
cx q[2],q[4];
cx q[3],q[11];
cx q[16],q[23];
swap q[7],q[1];
cx q[10],q[22];
cx q[17],q[12];
cx q[29],q[42];
cx q[24],q[15];
cx q[2],q[4];
swap q[39],q[37];
cx q[3],q[11];
cx q[16],q[23];
cx q[24],q[15];
swap q[48],q[41];
swap q[41],q[34];
swap q[4],q[2];
cx q[29],q[31];
cx q[29],q[31];
swap q[26],q[18];
cx q[1],q[2];
cx q[29],q[45];
swap q[14],q[7];
cx q[10],q[22];
swap q[34],q[33];
cx q[4],q[12];
cx q[16],q[22];
cx q[1],q[2];
cx q[17],q[11];
cx q[29],q[45];
swap q[7],q[2];
cx q[4],q[12];
cx q[16],q[22];
cx q[39],q[25];
cx q[17],q[11];
cx q[29],q[43];
swap q[26],q[19];
cx q[10],q[15];
cx q[10],q[15];
swap q[37],q[35];
cx q[2],q[18];
swap q[22],q[14];
cx q[2],q[18];
cx q[4],q[11];
cx q[39],q[25];
swap q[37],q[30];
cx q[16],q[15];
cx q[4],q[11];
cx q[39],q[40];
swap q[28],q[21];
cx q[16],q[15];
cx q[39],q[40];
cx q[39],q[38];
swap q[19],q[3];
cx q[29],q[43];
cx q[39],q[38];
swap q[16],q[8];
cx q[2],q[3];
cx q[29],q[35];
cx q[39],q[46];
swap q[25],q[19];
cx q[9],q[21];
cx q[39],q[46];
cx q[9],q[21];
cx q[30],q[18];
cx q[2],q[3];
swap q[21],q[15];
cx q[30],q[18];
cx q[39],q[32];
swap q[2],q[1];
cx q[16],q[15];
cx q[29],q[35];
cx q[39],q[32];
swap q[12],q[4];
cx q[16],q[15];
swap q[30],q[25];
cx q[2],q[4];
cx q[2],q[4];
swap q[7],q[0];
cx q[9],q[18];
cx q[30],q[23];
swap q[33],q[32];
cx q[9],q[18];
cx q[30],q[23];
cx q[16],q[18];
swap q[47],q[45];
swap q[36],q[28];
swap q[7],q[1];
cx q[16],q[18];
cx q[17],q[23];
swap q[4],q[3];
swap q[46],q[34];
cx q[7],q[28];
swap q[19],q[12];
cx q[7],q[28];
swap q[24],q[10];
swap q[46],q[38];
cx q[10],q[15];
swap q[42],q[28];
cx q[25],q[4];
swap q[1],q[0];
cx q[25],q[4];
cx q[7],q[28];
cx q[9],q[4];
cx q[29],q[32];
cx q[10],q[15];
swap q[45],q[42];
cx q[9],q[4];
cx q[24],q[15];
swap q[26],q[19];
swap q[11],q[4];
cx q[7],q[28];
cx q[30],q[14];
swap q[45],q[39];
cx q[10],q[18];
cx q[30],q[14];
cx q[25],q[39];
cx q[1],q[3];
cx q[30],q[21];
cx q[25],q[39];
cx q[2],q[4];
cx q[30],q[21];
cx q[16],q[11];
swap q[37],q[28];
cx q[17],q[23];
cx q[1],q[3];
cx q[10],q[18];
swap q[39],q[33];
cx q[25],q[37];
cx q[2],q[4];
swap q[21],q[7];
cx q[16],q[11];
cx q[25],q[37];
cx q[24],q[15];
cx q[10],q[11];
swap q[45],q[37];
cx q[17],q[14];
cx q[8],q[15];
cx q[24],q[18];
swap q[40],q[27];
cx q[26],q[23];
cx q[10],q[11];
cx q[26],q[23];
swap q[7],q[2];
cx q[24],q[18];
cx q[29],q[32];
cx q[17],q[14];
cx q[8],q[15];
cx q[24],q[11];
swap q[33],q[26];
cx q[1],q[4];
cx q[29],q[42];
cx q[7],q[23];
swap q[26],q[11];
cx q[30],q[15];
cx q[7],q[23];
swap q[45],q[39];
cx q[9],q[11];
cx q[24],q[26];
cx q[29],q[42];
cx q[9],q[11];
cx q[29],q[36];
swap q[33],q[25];
cx q[16],q[11];
cx q[29],q[36];
cx q[16],q[11];
cx q[17],q[2];
cx q[29],q[44];
swap q[39],q[32];
cx q[10],q[11];
cx q[30],q[15];
cx q[1],q[4];
cx q[29],q[44];
cx q[17],q[2];
cx q[29],q[28];
swap q[26],q[12];
cx q[17],q[15];
cx q[29],q[28];
cx q[17],q[15];
swap q[37],q[31];
swap q[47],q[46];
cx q[21],q[37];
cx q[10],q[11];
swap q[2],q[1];
swap q[32],q[26];
cx q[21],q[37];
cx q[3],q[4];
cx q[3],q[4];
swap q[17],q[9];
cx q[29],q[38];
swap q[15],q[14];
cx q[17],q[26];
cx q[29],q[38];
cx q[17],q[26];
swap q[46],q[37];
swap q[28],q[21];
swap q[8],q[2];
swap q[12],q[4];
swap q[26],q[17];
cx q[28],q[37];
swap q[46],q[40];
cx q[28],q[37];
cx q[2],q[18];
cx q[16],q[17];
cx q[28],q[43];
swap q[34],q[27];
cx q[2],q[18];
cx q[28],q[43];
swap q[23],q[15];
cx q[33],q[40];
cx q[2],q[4];
cx q[28],q[35];
cx q[33],q[40];
cx q[2],q[4];
cx q[28],q[35];
swap q[38],q[37];
cx q[25],q[23];
swap q[4],q[1];
cx q[26],q[40];
cx q[16],q[17];
cx q[33],q[38];
swap q[21],q[14];
swap q[43],q[36];
cx q[25],q[23];
swap q[14],q[1];
cx q[26],q[40];
cx q[10],q[17];
cx q[33],q[38];
cx q[10],q[17];
cx q[8],q[15];
cx q[25],q[4];
swap q[36],q[28];
cx q[7],q[23];
swap q[45],q[33];
cx q[8],q[15];
cx q[7],q[23];
swap q[10],q[4];
swap q[35],q[28];
cx q[30],q[18];
swap q[43],q[35];
cx q[24],q[11];
cx q[26],q[38];
cx q[3],q[15];
cx q[45],q[43];
cx q[25],q[10];
swap q[15],q[1];
cx q[24],q[11];
cx q[45],q[43];
cx q[26],q[38];
swap q[42],q[28];
cx q[8],q[23];
cx q[7],q[10];
swap q[40],q[26];
cx q[2],q[11];
cx q[31],q[22];
swap q[44],q[43];
cx q[24],q[17];
cx q[3],q[1];
cx q[8],q[23];
swap q[46],q[40];
cx q[30],q[18];
swap q[21],q[15];
cx q[46],q[44];
cx q[2],q[11];
cx q[45],q[42];
cx q[24],q[17];
swap q[34],q[33];
cx q[9],q[18];
cx q[46],q[44];
swap q[30],q[29];
cx q[29],q[14];
cx q[2],q[17];
swap q[44],q[42];
cx q[7],q[10];
cx q[9],q[18];
swap q[29],q[22];
cx q[45],q[44];
cx q[8],q[10];
swap q[25],q[24];
cx q[22],q[14];
cx q[46],q[44];
cx q[9],q[14];
cx q[46],q[44];
swap q[12],q[5];
cx q[24],q[15];
swap q[47],q[44];
cx q[24],q[15];
cx q[7],q[15];
cx q[24],q[18];
swap q[34],q[27];
swap q[5],q[4];
cx q[36],q[39];
cx q[4],q[1];
swap q[17],q[16];
cx q[36],q[39];
cx q[4],q[1];
cx q[45],q[39];
cx q[36],q[28];
swap q[19],q[5];
cx q[8],q[10];
cx q[45],q[39];
cx q[36],q[28];
cx q[7],q[15];
cx q[46],q[39];
cx q[36],q[35];
swap q[10],q[2];
cx q[8],q[15];
cx q[46],q[39];
cx q[36],q[35];
cx q[8],q[15];
cx q[36],q[43];
swap q[25],q[19];
cx q[9],q[14];
cx q[36],q[43];
swap q[39],q[34];
cx q[24],q[18];
cx q[36],q[21];
swap q[9],q[3];
cx q[36],q[21];
cx q[36],q[37];
swap q[24],q[17];
cx q[36],q[37];
swap q[27],q[19];
swap q[22],q[7];
swap q[11],q[4];
cx q[24],q[26];
swap q[45],q[42];
cx q[24],q[26];
swap q[15],q[1];
cx q[25],q[26];
cx q[42],q[28];
swap q[4],q[3];
cx q[24],q[38];
cx q[42],q[28];
swap q[8],q[7];
cx q[25],q[26];
cx q[10],q[16];
cx q[42],q[35];
cx q[24],q[38];
cx q[27],q[26];
cx q[42],q[35];
swap q[11],q[10];
cx q[25],q[38];
cx q[42],q[43];
cx q[27],q[26];
cx q[42],q[43];
cx q[31],q[29];
cx q[9],q[23];
swap q[38],q[32];
cx q[8],q[3];
cx q[9],q[23];
swap q[43],q[28];
swap q[26],q[19];
cx q[24],q[45];
cx q[8],q[3];
cx q[25],q[32];
cx q[10],q[23];
swap q[42],q[35];
cx q[24],q[45];
cx q[9],q[2];
swap q[19],q[12];
cx q[17],q[14];
cx q[46],q[43];
cx q[4],q[3];
cx q[8],q[16];
cx q[27],q[32];
cx q[35],q[21];
swap q[46],q[45];
cx q[11],q[12];
cx q[9],q[2];
cx q[27],q[32];
cx q[35],q[21];
cx q[10],q[23];
swap q[41],q[27];
cx q[17],q[14];
cx q[45],q[43];
cx q[4],q[3];
cx q[8],q[16];
cx q[45],q[42];
cx q[25],q[46];
cx q[9],q[1];
swap q[43],q[28];
cx q[11],q[12];
cx q[15],q[23];
cx q[25],q[46];
cx q[9],q[1];
cx q[10],q[2];
cx q[41],q[46];
cx q[15],q[23];
cx q[35],q[37];
swap q[32],q[26];
cx q[17],q[3];
cx q[41],q[46];
cx q[35],q[37];
cx q[4],q[16];
swap q[45],q[36];
cx q[11],q[26];
cx q[10],q[2];
swap q[26],q[24];
cx q[15],q[2];
cx q[36],q[42];
cx q[15],q[2];
cx q[36],q[43];
swap q[47],q[41];
swap q[46],q[40];
cx q[17],q[3];
cx q[36],q[43];
swap q[14],q[0];
cx q[26],q[41];
cx q[30],q[38];
cx q[26],q[41];
cx q[30],q[38];
swap q[43],q[35];
cx q[25],q[41];
cx q[45],q[38];
swap q[8],q[3];
cx q[26],q[34];
cx q[45],q[38];
cx q[25],q[41];
swap q[43],q[37];
cx q[3],q[12];
cx q[26],q[34];
cx q[4],q[16];
cx q[47],q[41];
swap q[28],q[21];
swap q[26],q[25];
cx q[3],q[12];
cx q[17],q[16];
cx q[47],q[41];
cx q[36],q[28];
cx q[10],q[1];
cx q[36],q[28];
cx q[11],q[24];
cx q[26],q[34];
cx q[36],q[43];
swap q[15],q[1];
cx q[4],q[12];
cx q[26],q[34];
cx q[37],q[38];
cx q[4],q[12];
cx q[47],q[34];
cx q[36],q[43];
cx q[3],q[24];
cx q[47],q[34];
swap q[28],q[14];
cx q[17],q[16];
cx q[37],q[38];
cx q[3],q[24];
swap q[40],q[27];
cx q[10],q[15];
swap q[25],q[24];
cx q[1],q[15];
cx q[36],q[38];
cx q[1],q[15];
cx q[36],q[38];
swap q[41],q[20];
cx q[23],q[2];
swap q[42],q[35];
cx q[11],q[27];
cx q[31],q[46];
cx q[11],q[27];
cx q[31],q[46];
swap q[22],q[16];
cx q[11],q[20];
swap q[7],q[1];
swap q[37],q[32];
cx q[11],q[20];
swap q[9],q[1];
cx q[30],q[33];
swap q[12],q[11];
cx q[24],q[21];
cx q[16],q[18];
swap q[27],q[20];
cx q[16],q[18];
swap q[36],q[31];
cx q[17],q[11];
swap q[34],q[27];
cx q[9],q[18];
cx q[16],q[0];
swap q[38],q[35];
cx q[9],q[18];
cx q[16],q[0];
cx q[12],q[27];
swap q[42],q[36];
cx q[9],q[0];
cx q[12],q[27];
cx q[16],q[8];
swap q[18],q[3];
cx q[16],q[8];
cx q[16],q[22];
cx q[1],q[3];
cx q[24],q[21];
swap q[20],q[19];
cx q[1],q[3];
cx q[24],q[38];
cx q[9],q[0];
cx q[24],q[38];
cx q[10],q[3];
cx q[1],q[0];
cx q[18],q[19];
cx q[16],q[22];
cx q[9],q[8];
cx q[4],q[25];
swap q[22],q[14];
cx q[10],q[3];
cx q[1],q[0];
cx q[18],q[19];
cx q[9],q[8];
cx q[17],q[11];
swap q[43],q[38];
cx q[1],q[8];
cx q[18],q[34];
cx q[23],q[2];
swap q[34],q[13];
cx q[23],q[15];
cx q[9],q[14];
cx q[4],q[25];
cx q[1],q[8];
cx q[16],q[11];
swap q[27],q[20];
cx q[17],q[25];
cx q[24],q[36];
swap q[7],q[1];
cx q[1],q[3];
cx q[23],q[15];
cx q[18],q[13];
cx q[4],q[19];
swap q[8],q[0];
cx q[17],q[25];
cx q[4],q[19];
swap q[35],q[21];
cx q[10],q[8];
cx q[30],q[33];
swap q[13],q[12];
cx q[1],q[3];
cx q[24],q[36];
cx q[10],q[8];
cx q[45],q[33];
swap q[36],q[28];
cx q[9],q[14];
cx q[4],q[12];
cx q[45],q[33];
cx q[16],q[11];
cx q[7],q[14];
cx q[32],q[33];
cx q[2],q[15];
cx q[18],q[20];
cx q[30],q[44];
cx q[32],q[33];
swap q[35],q[28];
cx q[1],q[8];
cx q[17],q[19];
cx q[30],q[44];
cx q[9],q[11];
cx q[7],q[14];
cx q[45],q[44];
cx q[31],q[33];
cx q[16],q[25];
cx q[4],q[12];
cx q[45],q[44];
swap q[7],q[1];
cx q[24],q[22];
cx q[16],q[25];
swap q[11],q[3];
cx q[7],q[8];
cx q[24],q[22];
cx q[23],q[11];
swap q[38],q[32];
cx q[2],q[15];
cx q[18],q[20];
cx q[23],q[11];
swap q[7],q[0];
cx q[9],q[3];
cx q[24],q[32];
cx q[17],q[19];
cx q[30],q[39];
cx q[10],q[7];
cx q[30],q[39];
cx q[10],q[7];
cx q[38],q[44];
cx q[2],q[11];
cx q[0],q[7];
cx q[24],q[32];
cx q[30],q[37];
swap q[28],q[22];
cx q[1],q[3];
cx q[45],q[39];
cx q[30],q[37];
cx q[9],q[25];
cx q[38],q[44];
cx q[23],q[8];
cx q[4],q[20];
cx q[45],q[39];
cx q[16],q[19];
cx q[0],q[7];
cx q[38],q[39];
cx q[17],q[12];
cx q[45],q[37];
swap q[32],q[26];
cx q[2],q[11];
cx q[30],q[29];
cx q[23],q[8];
cx q[45],q[37];
cx q[1],q[3];
cx q[24],q[21];
cx q[9],q[25];
cx q[30],q[29];
cx q[16],q[19];
cx q[38],q[39];
cx q[17],q[12];
cx q[2],q[8];
cx q[38],q[37];
cx q[4],q[20];
cx q[23],q[7];
cx q[17],q[20];
cx q[38],q[37];
swap q[14],q[1];
cx q[17],q[20];
cx q[31],q[33];
swap q[17],q[11];
cx q[2],q[8];
cx q[31],q[44];
cx q[31],q[44];
swap q[14],q[2];
cx q[31],q[39];
cx q[31],q[39];
swap q[18],q[12];
cx q[23],q[7];
swap q[44],q[39];
cx q[10],q[1];
cx q[10],q[1];
swap q[23],q[22];
cx q[0],q[1];
cx q[10],q[3];
cx q[31],q[37];
swap q[20],q[12];
swap q[47],q[40];
cx q[15],q[17];
cx q[0],q[1];
cx q[31],q[37];
cx q[16],q[18];
cx q[14],q[7];
swap q[43],q[37];
cx q[32],q[23];
cx q[10],q[3];
cx q[14],q[7];
cx q[32],q[23];
cx q[0],q[3];
swap q[35],q[28];
cx q[32],q[37];
cx q[15],q[17];
swap q[19],q[12];
swap q[3],q[2];
cx q[32],q[37];
cx q[15],q[8];
cx q[16],q[18];
swap q[37],q[28];
cx q[9],q[12];
cx q[9],q[12];
cx q[32],q[37];
cx q[15],q[8];
swap q[17],q[3];
cx q[32],q[37];
cx q[15],q[7];
cx q[45],q[29];
swap q[40],q[32];
cx q[22],q[1];
cx q[9],q[18];
cx q[45],q[29];
swap q[27],q[13];
cx q[32],q[23];
cx q[0],q[2];
swap q[44],q[35];
cx q[32],q[23];
cx q[17],q[25];
cx q[3],q[8];
cx q[17],q[25];
cx q[30],q[46];
cx q[3],q[8];
cx q[24],q[21];
swap q[46],q[44];
cx q[10],q[25];
cx q[15],q[7];
cx q[17],q[12];
swap q[35],q[28];
cx q[40],q[46];
cx q[16],q[19];
swap q[7],q[0];
cx q[40],q[46];
cx q[10],q[25];
cx q[30],q[44];
swap q[28],q[21];
cx q[40],q[26];
cx q[3],q[0];
cx q[45],q[44];
cx q[30],q[36];
cx q[40],q[26];
cx q[3],q[0];
cx q[45],q[44];
cx q[30],q[36];
cx q[24],q[33];
cx q[8],q[0];
cx q[45],q[36];
swap q[19],q[11];
cx q[22],q[1];
cx q[24],q[33];
cx q[14],q[1];
cx q[45],q[36];
cx q[17],q[12];
cx q[14],q[1];
swap q[40],q[37];
cx q[9],q[18];
cx q[16],q[11];
cx q[37],q[28];
cx q[10],q[12];
cx q[15],q[1];
swap q[39],q[33];
cx q[37],q[28];
cx q[17],q[18];
cx q[15],q[1];
cx q[9],q[11];
swap q[35],q[30];
cx q[10],q[12];
cx q[8],q[0];
cx q[17],q[18];
cx q[3],q[1];
cx q[32],q[30];
cx q[9],q[11];
cx q[32],q[30];
cx q[10],q[18];
swap q[15],q[7];
cx q[32],q[40];
cx q[17],q[11];
cx q[32],q[40];
cx q[10],q[18];
swap q[45],q[43];
swap q[30],q[28];
cx q[17],q[11];
swap q[15],q[2];
cx q[32],q[46];
cx q[32],q[46];
swap q[27],q[19];
cx q[22],q[15];
cx q[37],q[39];
cx q[10],q[11];
cx q[32],q[26];
cx q[22],q[15];
cx q[3],q[1];
swap q[38],q[37];
cx q[32],q[26];
cx q[14],q[15];
cx q[10],q[11];
cx q[24],q[33];
cx q[14],q[15];
swap q[10],q[2];
cx q[32],q[30];
cx q[7],q[15];
cx q[32],q[30];
cx q[7],q[15];
cx q[10],q[25];
cx q[38],q[39];
cx q[8],q[1];
swap q[44],q[36];
cx q[10],q[25];
cx q[8],q[1];
cx q[24],q[33];
cx q[0],q[1];
cx q[22],q[25];
cx q[0],q[1];
cx q[10],q[12];
cx q[32],q[39];
cx q[37],q[29];
cx q[38],q[33];
cx q[10],q[12];
cx q[24],q[21];
cx q[22],q[25];
cx q[32],q[39];
cx q[37],q[29];
cx q[10],q[18];
cx q[38],q[33];
cx q[10],q[18];
swap q[14],q[8];
cx q[32],q[33];
cx q[37],q[36];
cx q[10],q[11];
swap q[23],q[16];
cx q[19],q[16];
cx q[37],q[36];
cx q[19],q[16];
cx q[37],q[44];
cx q[24],q[21];
cx q[37],q[44];
swap q[20],q[4];
cx q[24],q[45];
cx q[31],q[29];
swap q[9],q[8];
cx q[4],q[16];
cx q[32],q[33];
swap q[37],q[28];
cx q[4],q[16];
cx q[9],q[25];
cx q[9],q[25];
swap q[20],q[12];
cx q[24],q[45];
swap q[35],q[21];
cx q[10],q[11];
swap q[8],q[7];
cx q[38],q[35];
swap q[26],q[20];
cx q[38],q[35];
cx q[3],q[15];
cx q[31],q[29];
swap q[12],q[9];
cx q[38],q[45];
cx q[31],q[36];
cx q[9],q[16];
cx q[38],q[45];
swap q[27],q[18];
cx q[9],q[16];
swap q[30],q[22];
swap q[43],q[35];
swap q[9],q[8];
cx q[24],q[29];
cx q[24],q[29];
swap q[33],q[19];
cx q[3],q[15];
swap q[44],q[38];
cx q[18],q[16];
cx q[31],q[36];
cx q[18],q[16];
cx q[44],q[29];
swap q[39],q[33];
cx q[23],q[16];
cx q[23],q[16];
swap q[27],q[20];
swap q[4],q[3];
swap q[14],q[8];
cx q[39],q[37];
swap q[25],q[18];
swap q[23],q[22];
cx q[39],q[37];
cx q[9],q[18];
cx q[39],q[40];
cx q[9],q[18];
swap q[37],q[31];
cx q[39],q[40];
cx q[7],q[16];
cx q[39],q[46];
cx q[7],q[16];
swap q[26],q[25];
cx q[39],q[46];
cx q[17],q[16];
cx q[44],q[29];
swap q[11],q[3];
cx q[39],q[27];
cx q[17],q[16];
swap q[43],q[29];
cx q[39],q[27];
cx q[2],q[16];
cx q[2],q[16];
swap q[18],q[11];
cx q[39],q[23];
swap q[7],q[0];
swap q[26],q[20];
cx q[39],q[23];
cx q[4],q[11];
cx q[18],q[31];
cx q[8],q[15];
swap q[40],q[33];
cx q[18],q[31];
cx q[8],q[15];
cx q[10],q[16];
cx q[37],q[38];
swap q[28],q[14];
cx q[18],q[33];
cx q[37],q[38];
cx q[18],q[33];
swap q[43],q[37];
cx q[39],q[40];
cx q[4],q[11];
cx q[7],q[15];
cx q[39],q[40];
cx q[10],q[16];
swap q[26],q[19];
cx q[28],q[31];
swap q[10],q[8];
cx q[28],q[31];
cx q[30],q[25];
swap q[34],q[20];
cx q[30],q[25];
cx q[10],q[11];
cx q[7],q[15];
cx q[34],q[31];
cx q[10],q[11];
cx q[1],q[15];
cx q[34],q[31];
cx q[1],q[15];
swap q[43],q[28];
cx q[12],q[25];
swap q[7],q[2];
cx q[39],q[26];
cx q[32],q[29];
cx q[22],q[31];
cx q[2],q[11];
swap q[46],q[40];
cx q[22],q[31];
cx q[2],q[11];
cx q[12],q[25];
swap q[7],q[0];
swap q[16],q[2];
cx q[39],q[26];
swap q[21],q[7];
swap q[40],q[19];
swap q[31],q[30];
swap q[15],q[0];
cx q[18],q[19];
cx q[31],q[40];
cx q[18],q[19];
cx q[31],q[40];
cx q[18],q[27];
cx q[21],q[30];
swap q[3],q[1];
cx q[18],q[27];
cx q[21],q[30];
cx q[18],q[23];
swap q[46],q[39];
cx q[18],q[23];
swap q[46],q[43];
cx q[3],q[11];
cx q[3],q[11];
swap q[1],q[0];
swap q[26],q[19];
cx q[17],q[30];
swap q[14],q[0];
cx q[46],q[33];
cx q[17],q[30];
cx q[46],q[33];
swap q[20],q[12];
cx q[34],q[33];
cx q[15],q[30];
swap q[25],q[17];
cx q[34],q[33];
cx q[15],q[30];
swap q[2],q[1];
swap q[30],q[14];
cx q[9],q[17];
cx q[9],q[17];
swap q[40],q[27];
cx q[18],q[39];
cx q[8],q[14];
swap q[23],q[22];
cx q[18],q[39];
cx q[8],q[14];
cx q[20],q[27];
cx q[2],q[11];
swap q[46],q[40];
cx q[31],q[30];
cx q[18],q[19];
cx q[24],q[36];
cx q[20],q[27];
swap q[8],q[1];
cx q[40],q[26];
cx q[31],q[30];
cx q[4],q[17];
cx q[40],q[26];
cx q[4],q[17];
swap q[31],q[30];
cx q[34],q[26];
cx q[10],q[17];
cx q[40],q[46];
cx q[18],q[19];
swap q[15],q[8];
cx q[34],q[26];
cx q[30],q[15];
cx q[10],q[17];
swap q[40],q[39];
cx q[24],q[36];
cx q[44],q[36];
cx q[16],q[17];
swap q[27],q[12];
cx q[39],q[46];
cx q[30],q[15];
cx q[34],q[46];
cx q[9],q[12];
cx q[34],q[46];
cx q[9],q[12];
cx q[32],q[29];
cx q[4],q[12];
cx q[16],q[17];
swap q[27],q[20];
cx q[43],q[29];
cx q[24],q[38];
cx q[4],q[12];
cx q[32],q[45];
cx q[3],q[17];
swap q[30],q[22];
cx q[32],q[45];
cx q[10],q[12];
cx q[10],q[12];
swap q[34],q[25];
cx q[43],q[29];
cx q[43],q[45];
cx q[22],q[14];
swap q[12],q[10];
cx q[39],q[30];
swap q[27],q[26];
cx q[39],q[30];
cx q[16],q[10];
cx q[39],q[40];
cx q[44],q[36];
cx q[22],q[14];
cx q[2],q[11];
swap q[31],q[24];
cx q[39],q[40];
cx q[16],q[10];
cx q[43],q[45];
swap q[29],q[22];
cx q[26],q[24];
swap q[12],q[4];
cx q[25],q[30];
swap q[16],q[15];
cx q[25],q[30];
cx q[25],q[40];
swap q[36],q[30];
cx q[25],q[40];
cx q[3],q[17];
swap q[39],q[27];
cx q[2],q[17];
cx q[3],q[10];
swap q[22],q[15];
swap q[24],q[18];
cx q[3],q[10];
swap q[22],q[15];
swap q[39],q[33];
cx q[24],q[22];
cx q[23],q[39];
cx q[27],q[19];
cx q[23],q[39];
cx q[26],q[18];
cx q[32],q[37];
cx q[27],q[19];
swap q[28],q[21];
cx q[32],q[37];
cx q[31],q[38];
cx q[9],q[18];
cx q[24],q[22];
cx q[43],q[37];
swap q[40],q[27];
cx q[44],q[38];
cx q[2],q[17];
cx q[43],q[37];
cx q[11],q[17];
swap q[29],q[28];
cx q[25],q[19];
cx q[44],q[38];
cx q[2],q[10];
cx q[24],q[45];
cx q[32],q[30];
cx q[11],q[17];
cx q[9],q[18];
swap q[39],q[36];
cx q[12],q[18];
cx q[12],q[18];
cx q[29],q[36];
cx q[25],q[19];
cx q[2],q[10];
swap q[39],q[34];
cx q[29],q[36];
cx q[4],q[18];
cx q[4],q[18];
swap q[14],q[8];
cx q[39],q[36];
cx q[11],q[10];
cx q[39],q[36];
cx q[11],q[10];
cx q[24],q[45];
swap q[21],q[14];
cx q[32],q[30];
cx q[17],q[10];
swap q[8],q[1];
cx q[21],q[36];
cx q[32],q[38];
cx q[17],q[10];
cx q[21],q[36];
cx q[32],q[38];
cx q[24],q[37];
swap q[26],q[19];
swap q[4],q[1];
cx q[43],q[30];
swap q[40],q[39];
cx q[19],q[16];
swap q[36],q[29];
cx q[19],q[16];
cx q[9],q[16];
cx q[19],q[4];
swap q[39],q[32];
cx q[8],q[29];
cx q[19],q[4];
cx q[8],q[29];
swap q[33],q[25];
cx q[28],q[29];
cx q[9],q[16];
cx q[28],q[29];
cx q[9],q[4];
swap q[46],q[39];
cx q[23],q[25];
swap q[12],q[10];
cx q[23],q[25];
cx q[43],q[30];
swap q[40],q[34];
cx q[23],q[39];
swap q[26],q[19];
cx q[23],q[39];
cx q[24],q[37];
swap q[40],q[39];
cx q[10],q[16];
cx q[24],q[30];
cx q[9],q[4];
swap q[27],q[19];
cx q[23],q[39];
cx q[23],q[39];
cx q[43],q[38];
cx q[10],q[16];
cx q[24],q[30];
cx q[10],q[4];
swap q[39],q[26];
cx q[1],q[16];
cx q[1],q[16];
swap q[36],q[30];
cx q[10],q[4];
swap q[45],q[40];
cx q[15],q[18];
cx q[15],q[18];
swap q[31],q[30];
cx q[3],q[18];
cx q[15],q[16];
swap q[45],q[39];
cx q[1],q[4];
cx q[3],q[18];
cx q[15],q[16];
cx q[31],q[25];
cx q[1],q[4];
swap q[36],q[21];
cx q[31],q[25];
cx q[34],q[25];
swap q[4],q[1];
cx q[31],q[39];
swap q[37],q[36];
cx q[34],q[25];
cx q[15],q[1];
cx q[31],q[39];
cx q[2],q[18];
cx q[37],q[25];
cx q[15],q[1];
cx q[34],q[39];
cx q[3],q[16];
cx q[31],q[26];
swap q[21],q[9];
cx q[34],q[39];
cx q[31],q[26];
cx q[37],q[25];
swap q[16],q[10];
cx q[37],q[39];
cx q[34],q[26];
cx q[37],q[39];
cx q[34],q[26];
cx q[45],q[29];
cx q[2],q[18];
cx q[45],q[29];
cx q[3],q[10];
swap q[32],q[24];
cx q[21],q[29];
cx q[11],q[18];
cx q[21],q[29];
cx q[2],q[10];
swap q[15],q[8];
cx q[43],q[38];
cx q[11],q[18];
cx q[24],q[22];
cx q[3],q[1];
swap q[39],q[37];
cx q[24],q[22];
cx q[2],q[10];
cx q[3],q[1];
cx q[17],q[18];
cx q[24],q[40];
cx q[2],q[1];
cx q[24],q[40];
cx q[2],q[1];
cx q[39],q[26];
cx q[16],q[29];
swap q[3],q[1];
cx q[24],q[36];
cx q[39],q[26];
cx q[16],q[29];
cx q[24],q[36];
swap q[4],q[1];
cx q[32],q[38];
cx q[32],q[38];
swap q[29],q[22];
cx q[24],q[9];
swap q[26],q[19];
cx q[23],q[26];
cx q[11],q[10];
cx q[23],q[26];
cx q[11],q[10];
cx q[31],q[26];
cx q[11],q[3];
swap q[22],q[16];
cx q[31],q[26];
cx q[11],q[3];
cx q[34],q[26];
cx q[34],q[26];
cx q[24],q[9];
swap q[37],q[29];
cx q[39],q[26];
cx q[1],q[16];
cx q[39],q[26];
cx q[1],q[16];
cx q[24],q[38];
swap q[27],q[26];
cx q[24],q[38];
cx q[8],q[16];
swap q[38],q[37];
cx q[8],q[16];
cx q[4],q[16];
swap q[40],q[32];
cx q[23],q[26];
cx q[33],q[38];
cx q[17],q[18];
cx q[23],q[26];
cx q[33],q[38];
cx q[4],q[16];
cx q[31],q[26];
cx q[17],q[10];
cx q[23],q[38];
cx q[12],q[18];
cx q[33],q[32];
cx q[2],q[16];
cx q[12],q[18];
swap q[45],q[36];
cx q[33],q[32];
cx q[17],q[10];
cx q[2],q[16];
cx q[31],q[26];
swap q[12],q[4];
swap q[17],q[9];
cx q[34],q[26];
cx q[33],q[45];
swap q[22],q[15];
swap q[36],q[28];
cx q[22],q[25];
cx q[22],q[25];
cx q[22],q[29];
cx q[34],q[26];
cx q[4],q[10];
cx q[22],q[29];
cx q[33],q[45];
cx q[9],q[3];
cx q[23],q[38];
cx q[4],q[10];
swap q[40],q[34];
cx q[31],q[38];
cx q[11],q[16];
cx q[31],q[38];
cx q[9],q[3];
swap q[36],q[29];
cx q[39],q[26];
cx q[4],q[3];
cx q[23],q[32];
cx q[18],q[10];
swap q[45],q[37];
cx q[23],q[32];
cx q[18],q[10];
cx q[31],q[32];
cx q[11],q[16];
cx q[40],q[38];
cx q[9],q[16];
cx q[4],q[3];
cx q[23],q[37];
cx q[18],q[3];
cx q[33],q[17];
swap q[30],q[29];
cx q[39],q[26];
cx q[9],q[16];
cx q[33],q[17];
cx q[30],q[25];
cx q[30],q[25];
cx q[30],q[36];
cx q[33],q[45];
swap q[28],q[21];
swap q[24],q[17];
cx q[30],q[36];
cx q[33],q[45];
cx q[40],q[38];
swap q[21],q[15];
cx q[31],q[32];
cx q[18],q[3];
swap q[45],q[36];
cx q[40],q[32];
cx q[4],q[16];
cx q[40],q[32];
cx q[4],q[16];
cx q[23],q[37];
cx q[10],q[3];
swap q[26],q[19];
swap q[40],q[32];
cx q[23],q[24];
cx q[10],q[3];
cx q[23],q[24];
cx q[39],q[38];
cx q[23],q[36];
swap q[26],q[11];
cx q[31],q[37];
cx q[39],q[38];
cx q[23],q[36];
cx q[31],q[37];
cx q[39],q[40];
cx q[18],q[16];
cx q[32],q[37];
swap q[16],q[11];
cx q[39],q[40];
cx q[32],q[37];
cx q[22],q[16];
cx q[31],q[24];
cx q[18],q[11];
swap q[27],q[26];
cx q[22],q[16];
cx q[31],q[24];
cx q[10],q[11];
cx q[30],q[16];
cx q[30],q[16];
swap q[26],q[18];
cx q[39],q[37];
cx q[31],q[36];
swap q[22],q[16];
cx q[39],q[37];
cx q[10],q[11];
swap q[25],q[24];
cx q[3],q[11];
cx q[3],q[11];
swap q[45],q[29];
cx q[32],q[25];
swap q[17],q[16];
cx q[32],q[25];
cx q[31],q[36];
swap q[40],q[26];
swap q[30],q[24];
swap q[8],q[1];
swap q[14],q[8];
swap q[8],q[1];
swap q[19],q[10];
swap q[39],q[32];
swap q[37],q[23];
cx q[32],q[25];
cx q[15],q[30];
cx q[17],q[18];
cx q[15],q[30];
cx q[17],q[18];
cx q[28],q[30];
cx q[17],q[10];
cx q[32],q[25];
cx q[15],q[29];
cx q[24],q[18];
cx q[28],q[30];
cx q[17],q[10];
cx q[15],q[29];
cx q[24],q[18];
cx q[21],q[30];
swap q[16],q[8];
cx q[28],q[29];
cx q[21],q[30];
swap q[33],q[12];
cx q[15],q[22];
cx q[24],q[10];
swap q[37],q[36];
cx q[15],q[22];
cx q[24],q[10];
cx q[15],q[18];
cx q[28],q[29];
cx q[17],q[38];
cx q[14],q[30];
swap q[9],q[2];
cx q[21],q[29];
cx q[28],q[22];
cx q[39],q[37];
swap q[34],q[27];
swap q[34],q[19];
cx q[14],q[30];
cx q[21],q[29];
swap q[40],q[19];
cx q[28],q[22];
cx q[15],q[18];
cx q[39],q[37];
cx q[17],q[38];
cx q[21],q[22];
cx q[16],q[30];
swap q[40],q[39];
cx q[14],q[29];
cx q[17],q[26];
cx q[15],q[10];
swap q[37],q[31];
cx q[21],q[22];
cx q[17],q[26];
cx q[15],q[10];
swap q[46],q[33];
cx q[16],q[30];
cx q[14],q[29];
swap q[10],q[2];
cx q[24],q[38];
cx q[14],q[22];
cx q[24],q[38];
cx q[14],q[22];
cx q[46],q[30];
swap q[18],q[9];
cx q[46],q[30];
cx q[32],q[31];
swap q[28],q[14];
cx q[24],q[26];
cx q[17],q[23];
cx q[32],q[31];
swap q[46],q[44];
swap q[2],q[1];
cx q[16],q[29];
cx q[14],q[9];
cx q[24],q[26];
swap q[44],q[36];
cx q[14],q[9];
cx q[21],q[9];
cx q[14],q[1];
cx q[17],q[23];
swap q[38],q[37];
cx q[21],q[9];
cx q[14],q[1];
cx q[24],q[23];
cx q[16],q[29];
swap q[7],q[1];
cx q[18],q[30];
cx q[36],q[29];
cx q[17],q[25];
cx q[16],q[22];
swap q[32],q[18];
cx q[21],q[7];
cx q[36],q[29];
swap q[14],q[9];
cx q[32],q[30];
cx q[24],q[23];
swap q[37],q[29];
cx q[17],q[25];
cx q[24],q[25];
cx q[15],q[29];
cx q[24],q[25];
cx q[15],q[29];
swap q[26],q[18];
cx q[28],q[14];
cx q[16],q[22];
cx q[15],q[18];
cx q[28],q[14];
cx q[32],q[37];
swap q[18],q[10];
cx q[21],q[7];
cx q[39],q[30];
cx q[36],q[22];
cx q[15],q[10];
cx q[32],q[37];
cx q[16],q[14];
swap q[37],q[32];
cx q[15],q[23];
cx q[16],q[14];
swap q[11],q[4];
cx q[28],q[7];
cx q[17],q[31];
cx q[39],q[30];
swap q[25],q[17];
cx q[36],q[22];
cx q[28],q[7];
cx q[15],q[23];
cx q[39],q[32];
swap q[36],q[28];
cx q[25],q[31];
cx q[16],q[7];
cx q[15],q[17];
cx q[39],q[32];
cx q[18],q[30];
cx q[37],q[22];
swap q[25],q[11];
cx q[28],q[14];
cx q[37],q[22];
cx q[24],q[31];
cx q[28],q[14];
swap q[16],q[9];
cx q[24],q[31];
swap q[33],q[19];
swap q[28],q[14];
swap q[31],q[30];
cx q[9],q[7];
cx q[16],q[29];
cx q[14],q[7];
swap q[39],q[38];
cx q[16],q[29];
cx q[14],q[7];
cx q[21],q[29];
cx q[16],q[10];
swap q[8],q[7];
cx q[21],q[29];
cx q[16],q[10];
cx q[36],q[29];
cx q[16],q[23];
swap q[31],q[25];
cx q[36],q[29];
cx q[16],q[23];
swap q[11],q[3];
cx q[38],q[22];
cx q[18],q[25];
cx q[37],q[28];
swap q[12],q[4];
cx q[15],q[17];
cx q[37],q[28];
cx q[31],q[25];
cx q[16],q[17];
swap q[44],q[28];
cx q[18],q[32];
cx q[31],q[25];
cx q[16],q[17];
swap q[19],q[12];
cx q[15],q[30];
cx q[33],q[25];
cx q[38],q[22];
cx q[33],q[25];
swap q[17],q[10];
cx q[15],q[30];
cx q[18],q[32];
swap q[29],q[8];
cx q[34],q[25];
cx q[34],q[25];
cx q[9],q[8];
cx q[31],q[32];
cx q[37],q[29];
cx q[9],q[8];
cx q[38],q[44];
cx q[11],q[25];
swap q[23],q[22];
cx q[14],q[8];
cx q[31],q[32];
cx q[38],q[44];
cx q[14],q[8];
cx q[33],q[32];
cx q[18],q[23];
cx q[33],q[32];
swap q[44],q[38];
cx q[18],q[23];
cx q[11],q[25];
swap q[14],q[8];
cx q[16],q[30];
cx q[19],q[25];
cx q[37],q[29];
cx q[34],q[32];
swap q[21],q[14];
cx q[16],q[30];
cx q[19],q[25];
cx q[14],q[17];
swap q[38],q[33];
cx q[14],q[17];
cx q[14],q[22];
swap q[24],q[18];
cx q[14],q[22];
cx q[44],q[29];
swap q[10],q[9];
swap q[40],q[34];
cx q[14],q[9];
cx q[31],q[23];
swap q[44],q[35];
cx q[14],q[9];
cx q[31],q[23];
cx q[24],q[33];
cx q[37],q[21];
cx q[40],q[32];
swap q[17],q[16];
cx q[35],q[29];
cx q[37],q[21];
swap q[40],q[26];
cx q[14],q[30];
swap q[17],q[11];
cx q[14],q[30];
cx q[35],q[21];
swap q[32],q[26];
cx q[35],q[21];
swap q[30],q[16];
swap q[25],q[19];
cx q[36],q[30];
cx q[36],q[30];
swap q[21],q[9];
cx q[36],q[22];
cx q[17],q[26];
cx q[36],q[22];
cx q[17],q[26];
cx q[36],q[21];
cx q[24],q[33];
cx q[36],q[21];
cx q[31],q[33];
swap q[18],q[9];
cx q[31],q[33];
swap q[36],q[22];
cx q[25],q[26];
cx q[38],q[23];
cx q[25],q[26];
swap q[10],q[9];
cx q[24],q[29];
cx q[19],q[26];
cx q[9],q[30];
cx q[19],q[26];
cx q[9],q[30];
cx q[22],q[16];
swap q[26],q[19];
cx q[38],q[23];
cx q[24],q[29];
swap q[15],q[9];
cx q[38],q[33];
cx q[24],q[18];
cx q[31],q[29];
cx q[15],q[36];
swap q[11],q[10];
cx q[15],q[36];
cx q[15],q[21];
cx q[38],q[33];
cx q[15],q[21];
cx q[24],q[18];
swap q[30],q[29];
swap q[39],q[18];
swap q[22],q[8];
swap q[19],q[18];
cx q[22],q[29];
cx q[8],q[16];
cx q[22],q[29];
cx q[8],q[14];
swap q[19],q[11];
cx q[37],q[29];
cx q[15],q[16];
cx q[22],q[36];
cx q[37],q[29];
cx q[15],q[16];
swap q[40],q[19];
cx q[22],q[36];
cx q[35],q[29];
cx q[14],q[8];
swap q[24],q[23];
cx q[37],q[36];
cx q[8],q[14];
cx q[35],q[29];
cx q[32],q[24];
cx q[22],q[21];
cx q[37],q[36];
cx q[32],q[24];
cx q[22],q[21];
cx q[35],q[36];
cx q[31],q[30];
cx q[22],q[16];
cx q[35],q[36];
cx q[17],q[24];
cx q[32],q[33];
cx q[38],q[30];
swap q[35],q[28];
cx q[31],q[39];
cx q[22],q[16];
cx q[17],q[24];
cx q[32],q[33];
cx q[38],q[30];
swap q[3],q[2];
swap q[19],q[11];
swap q[7],q[2];
cx q[23],q[29];
cx q[37],q[21];
cx q[25],q[24];
cx q[31],q[39];
cx q[22],q[9];
cx q[23],q[29];
cx q[38],q[39];
cx q[37],q[21];
cx q[17],q[33];
cx q[32],q[30];
cx q[28],q[21];
cx q[31],q[29];
cx q[23],q[36];
cx q[38],q[39];
cx q[37],q[16];
cx q[28],q[21];
cx q[25],q[24];
cx q[15],q[10];
swap q[38],q[36];
cx q[10],q[15];
cx q[15],q[10];
cx q[17],q[33];
cx q[32],q[30];
swap q[21],q[15];
cx q[26],q[24];
cx q[25],q[33];
cx q[31],q[29];
cx q[23],q[38];
cx q[25],q[33];
cx q[37],q[16];
cx q[17],q[30];
cx q[32],q[39];
cx q[36],q[29];
cx q[31],q[38];
cx q[23],q[15];
cx q[28],q[16];
cx q[32],q[39];
cx q[26],q[24];
cx q[36],q[29];
cx q[17],q[30];
cx q[26],q[33];
cx q[31],q[38];
cx q[23],q[15];
cx q[26],q[33];
cx q[28],q[16];
swap q[39],q[26];
cx q[23],q[16];
cx q[28],q[7];
cx q[23],q[16];
cx q[7],q[28];
cx q[18],q[24];
cx q[36],q[38];
swap q[22],q[15];
cx q[17],q[26];
cx q[36],q[38];
cx q[18],q[24];
cx q[25],q[30];
cx q[9],q[15];
cx q[32],q[29];
cx q[31],q[22];
cx q[25],q[30];
cx q[15],q[9];
cx q[17],q[26];
cx q[28],q[7];
cx q[32],q[29];
cx q[31],q[22];
cx q[18],q[33];
cx q[36],q[22];
cx q[39],q[30];
cx q[25],q[26];
cx q[17],q[29];
cx q[32],q[38];
cx q[31],q[16];
cx q[18],q[33];
cx q[36],q[22];
cx q[39],q[30];
cx q[25],q[26];
cx q[17],q[29];
cx q[32],q[38];
cx q[31],q[16];
cx q[24],q[33];
swap q[36],q[29];
cx q[39],q[26];
cx q[18],q[30];
cx q[17],q[38];
cx q[29],q[16];
cx q[39],q[26];
cx q[24],q[33];
swap q[45],q[36];
swap q[31],q[25];
cx q[29],q[16];
cx q[31],q[45];
swap q[22],q[16];
cx q[31],q[45];
cx q[39],q[45];
cx q[25],q[34];
cx q[32],q[16];
cx q[18],q[30];
cx q[39],q[45];
cx q[17],q[38];
swap q[29],q[22];
cx q[18],q[26];
cx q[31],q[38];
cx q[18],q[26];
cx q[31],q[38];
swap q[9],q[2];
cx q[24],q[30];
cx q[39],q[38];
swap q[43],q[35];
cx q[32],q[16];
swap q[45],q[37];
cx q[17],q[16];
cx q[17],q[16];
swap q[25],q[19];
cx q[39],q[38];
swap q[23],q[16];
cx q[45],q[40];
cx q[40],q[45];
swap q[43],q[36];
cx q[32],q[29];
swap q[16],q[9];
cx q[24],q[30];
cx q[34],q[19];
cx q[32],q[29];
cx q[33],q[30];
cx q[9],q[11];
cx q[24],q[26];
cx q[11],q[9];
swap q[39],q[37];
cx q[24],q[26];
cx q[9],q[11];
cx q[19],q[34];
cx q[18],q[39];
swap q[34],q[27];
cx q[31],q[23];
cx q[17],q[29];
swap q[38],q[32];
cx q[17],q[29];
cx q[17],q[4];
swap q[47],q[34];
cx q[31],q[23];
cx q[33],q[30];
cx q[18],q[39];
cx q[33],q[26];
cx q[37],q[23];
swap q[47],q[46];
cx q[18],q[32];
cx q[37],q[23];
cx q[31],q[29];
cx q[33],q[26];
cx q[24],q[39];
cx q[24],q[39];
cx q[18],q[32];
cx q[31],q[29];
swap q[18],q[16];
cx q[33],q[39];
cx q[37],q[29];
cx q[33],q[39];
cx q[37],q[29];
cx q[24],q[32];
cx q[37],q[46];
swap q[35],q[28];
cx q[16],q[23];
cx q[46],q[37];
cx q[24],q[32];
cx q[37],q[46];
cx q[16],q[23];
cx q[33],q[32];
cx q[24],q[23];
cx q[33],q[32];
cx q[16],q[29];
cx q[45],q[40];
cx q[24],q[23];
cx q[38],q[36];
cx q[16],q[29];
swap q[26],q[18];
cx q[30],q[18];
cx q[30],q[18];
cx q[30],q[39];
cx q[30],q[39];
cx q[18],q[39];
cx q[30],q[32];
cx q[4],q[17];
cx q[24],q[29];
cx q[18],q[39];
cx q[30],q[32];
cx q[17],q[4];
cx q[24],q[29];
cx q[18],q[32];
cx q[18],q[32];
cx q[39],q[32];
cx q[16],q[28];
cx q[39],q[32];
cx q[28],q[16];
cx q[24],q[12];
cx q[31],q[26];
cx q[22],q[25];
cx q[25],q[22];
cx q[36],q[38];
swap q[25],q[23];
cx q[38],q[36];
cx q[33],q[25];
cx q[16],q[28];
cx q[33],q[25];
cx q[22],q[23];
cx q[30],q[25];
cx q[30],q[25];
cx q[18],q[25];
swap q[38],q[29];
cx q[18],q[25];
swap q[24],q[18];
cx q[33],q[38];
cx q[33],q[38];
swap q[7],q[0];
swap q[9],q[1];
swap q[14],q[7];
swap q[15],q[14];
swap q[42],q[36];
cx q[39],q[25];
swap q[16],q[15];
cx q[30],q[38];
cx q[33],q[47];
cx q[12],q[18];
cx q[30],q[38];
cx q[47],q[33];
cx q[18],q[12];
cx q[24],q[38];
cx q[33],q[47];
cx q[39],q[25];
cx q[30],q[43];
cx q[24],q[38];
cx q[32],q[25];
cx q[43],q[30];
swap q[17],q[16];
cx q[39],q[38];
cx q[32],q[25];
cx q[30],q[43];
cx q[39],q[38];
swap q[24],q[17];
cx q[32],q[38];
cx q[32],q[38];
cx q[25],q[38];
cx q[25],q[38];
cx q[38],q[36];
cx q[17],q[3];
cx q[25],q[9];
cx q[32],q[44];
cx q[39],q[24];
cx q[26],q[31];
cx q[3],q[17];
cx q[36],q[38];
cx q[24],q[39];
cx q[44],q[32];
cx q[9],q[25];
cx q[38],q[36];
cx q[17],q[3];
cx q[25],q[9];
cx q[32],q[44];
cx q[39],q[24];
cx q[31],q[26];
