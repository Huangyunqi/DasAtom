OPENQASM 2.0;
include "qelib1.inc";
qreg q[64];
creg c[64];
cx q[32],q[49];
swap q[23],q[6];
cx q[11],q[26];
swap q[47],q[44];
swap q[32],q[16];
cx q[23],q[39];
swap q[19],q[3];
cx q[39],q[47];
swap q[62],q[61];
swap q[61],q[44];
swap q[6],q[5];
swap q[16],q[9];
cx q[11],q[5];
swap q[46],q[39];
cx q[19],q[43];
swap q[22],q[14];
swap q[43],q[19];
swap q[16],q[1];
swap q[61],q[46];
swap q[23],q[6];
cx q[43],q[44];
cx q[61],q[59];
swap q[32],q[16];
swap q[28],q[12];
swap q[45],q[38];
cx q[19],q[4];
swap q[1],q[0];
swap q[43],q[34];
swap q[60],q[52];
swap q[62],q[55];
swap q[23],q[7];
cx q[34],q[32];
cx q[19],q[12];
swap q[2],q[1];
swap q[54],q[44];
cx q[6],q[4];
swap q[19],q[9];
cx q[38],q[22];
swap q[34],q[32];
swap q[63],q[53];
cx q[19],q[13];
cx q[4],q[13];
swap q[32],q[8];
cx q[28],q[52];
swap q[55],q[39];
cx q[34],q[44];
cx q[13],q[10];
cx q[44],q[53];
swap q[22],q[12];
cx q[53],q[35];
cx q[38],q[39];
cx q[52],q[46];
swap q[49],q[40];
swap q[10],q[2];
swap q[7],q[6];
swap q[40],q[16];
swap q[46],q[22];
swap q[11],q[3];
swap q[52],q[51];
swap q[49],q[34];
cx q[12],q[10];
cx q[53],q[46];
cx q[22],q[14];
cx q[16],q[10];
cx q[38],q[52];
swap q[6],q[4];
swap q[28],q[21];
swap q[62],q[55];
swap q[50],q[44];
cx q[10],q[11];
cx q[21],q[23];
swap q[34],q[26];
swap q[17],q[16];
swap q[55],q[46];
cx q[11],q[26];
cx q[50],q[32];
swap q[30],q[21];
cx q[40],q[32];
swap q[15],q[7];
swap q[44],q[36];
swap q[10],q[4];
cx q[30],q[46];
cx q[32],q[24];
swap q[50],q[34];
cx q[19],q[21];
swap q[53],q[44];
cx q[17],q[10];
cx q[15],q[39];
cx q[34],q[17];
swap q[28],q[12];
swap q[49],q[40];
swap q[39],q[23];
swap q[51],q[45];
cx q[23],q[14];
swap q[18],q[16];
swap q[41],q[34];
cx q[28],q[46];
swap q[16],q[10];
cx q[22],q[36];
cx q[36],q[46];
swap q[50],q[41];
cx q[45],q[39];
swap q[21],q[20];
cx q[40],q[16];
swap q[35],q[34];
swap q[59],q[45];
swap q[39],q[30];
swap q[40],q[16];
cx q[35],q[20];
swap q[47],q[38];
cx q[41],q[40];
swap q[2],q[1];
swap q[3],q[2];
swap q[28],q[18];
swap q[44],q[43];
swap q[12],q[3];
cx q[35],q[45];
swap q[25],q[17];
cx q[20],q[38];
cx q[36],q[45];
swap q[30],q[12];
cx q[35],q[26];
cx q[41],q[26];
cx q[30],q[38];
cx q[12],q[28];
cx q[22],q[38];
cx q[25],q[43];
swap q[20],q[11];
cx q[25],q[41];
swap q[57],q[56];
swap q[58],q[56];
cx q[20],q[37];
swap q[5],q[4];
swap q[17],q[2];
cx q[30],q[37];
cx q[51],q[37];
swap q[27],q[20];
cx q[51],q[57];
swap q[40],q[33];
swap q[15],q[14];
swap q[14],q[13];
swap q[33],q[17];
cx q[51],q[58];
cx q[30],q[54];
swap q[4],q[3];
swap q[20],q[19];
cx q[33],q[57];
swap q[44],q[36];
swap q[47],q[30];
cx q[49],q[57];
swap q[10],q[3];
cx q[49],q[52];
swap q[28],q[22];
swap q[47],q[39];
swap q[51],q[44];
cx q[38],q[20];
swap q[17],q[10];
cx q[49],q[51];
cx q[28],q[18];
swap q[12],q[5];
cx q[44],q[27];
cx q[27],q[10];
cx q[44],q[46];
swap q[22],q[14];
swap q[58],q[41];
cx q[27],q[13];
swap q[39],q[38];
cx q[17],q[41];
cx q[20],q[26];
swap q[59],q[44];
cx q[20],q[30];
swap q[4],q[3];
swap q[41],q[25];
cx q[28],q[38];
cx q[45],q[38];
swap q[10],q[3];
cx q[43],q[46];
swap q[21],q[20];
cx q[25],q[10];
swap q[31],q[23];
swap q[54],q[37];
swap q[43],q[35];
swap q[25],q[19];
cx q[45],q[31];
cx q[17],q[20];
swap q[62],q[45];
swap q[51],q[43];
swap q[39],q[23];
swap q[27],q[10];
swap q[21],q[12];
swap q[59],q[41];
cx q[20],q[37];
swap q[23],q[6];
swap q[10],q[1];
cx q[27],q[45];
swap q[6],q[4];
cx q[27],q[34];
cx q[41],q[27];
swap q[30],q[22];
swap q[45],q[37];
cx q[4],q[19];
cx q[35],q[21];
swap q[44],q[26];
swap q[23],q[21];
cx q[18],q[26];
cx q[38],q[44];
swap q[20],q[14];
cx q[18],q[17];
swap q[46],q[43];
swap q[18],q[10];
cx q[46],q[30];
swap q[13],q[7];
swap q[25],q[24];
swap q[43],q[36];
swap q[46],q[30];
swap q[25],q[10];
swap q[14],q[5];
cx q[34],q[20];
cx q[62],q[46];
swap q[52],q[44];
cx q[20],q[13];
swap q[9],q[2];
cx q[5],q[13];
swap q[62],q[38];
swap q[27],q[26];
swap q[5],q[2];
cx q[38],q[35];
swap q[44],q[28];
swap q[26],q[18];
cx q[30],q[14];
cx q[36],q[20];
cx q[27],q[21];
cx q[13],q[10];
swap q[37],q[35];
cx q[7],q[13];
cx q[13],q[5];
swap q[55],q[54];
swap q[54],q[53];
swap q[24],q[16];
swap q[40],q[24];
swap q[58],q[40];
swap q[58],q[51];
swap q[32],q[8];
swap q[54],q[47];
swap q[19],q[18];
swap q[3],q[2];
swap q[22],q[15];
cx q[26],q[35];
swap q[16],q[2];
cx q[53],q[35];
cx q[50],q[35];
swap q[12],q[5];
swap q[30],q[15];
cx q[27],q[34];
swap q[12],q[10];
cx q[35],q[45];
cx q[51],q[45];
swap q[57],q[41];
cx q[28],q[12];
cx q[51],q[41];
cx q[13],q[19];
swap q[46],q[38];
cx q[49],q[41];
swap q[27],q[3];
cx q[49],q[32];
cx q[16],q[32];
swap q[22],q[21];
cx q[51],q[54];
cx q[16],q[10];
swap q[30],q[28];
cx q[41],q[17];
swap q[14],q[13];
swap q[17],q[11];
cx q[35],q[27];
cx q[37],q[51];
swap q[33],q[32];
cx q[11],q[13];
swap q[28],q[26];
swap q[58],q[43];
swap q[46],q[31];
swap q[41],q[32];
swap q[13],q[11];
cx q[33],q[26];
swap q[50],q[36];
cx q[31],q[13];
swap q[61],q[53];
swap q[15],q[13];
cx q[26],q[41];
cx q[45],q[27];
swap q[31],q[30];
swap q[42],q[41];
cx q[28],q[21];
cx q[21],q[18];
swap q[46],q[44];
cx q[20],q[11];
swap q[25],q[8];
cx q[36],q[28];
cx q[42],q[44];
swap q[13],q[12];
cx q[42],q[25];
swap q[38],q[30];
swap q[53],q[44];
cx q[26],q[11];
cx q[26],q[12];
swap q[38],q[37];
cx q[10],q[25];
swap q[61],q[52];
cx q[12],q[10];
cx q[3],q[10];
swap q[31],q[22];
swap q[43],q[35];
swap q[7],q[6];
swap q[46],q[31];
swap q[33],q[26];
swap q[14],q[6];
swap q[21],q[20];
swap q[52],q[36];
swap q[46],q[38];
cx q[20],q[35];
cx q[28],q[44];
swap q[23],q[13];
cx q[28],q[36];
cx q[38],q[36];
swap q[26],q[17];
swap q[54],q[53];
cx q[20],q[26];
cx q[38],q[23];
cx q[18],q[26];
cx q[21],q[36];
swap q[53],q[51];
cx q[20],q[18];
swap q[30],q[14];
swap q[36],q[35];
swap q[49],q[41];
swap q[41],q[33];
swap q[55],q[47];
swap q[19],q[12];
cx q[30],q[36];
swap q[51],q[42];
cx q[20],q[30];
swap q[47],q[46];
swap q[33],q[25];
swap q[44],q[36];
cx q[26],q[42];
cx q[14],q[12];
cx q[18],q[26];
swap q[47],q[31];
cx q[18],q[28];
swap q[50],q[40];
cx q[36],q[22];
swap q[26],q[17];
cx q[44],q[46];
cx q[36],q[50];
swap q[22],q[12];
swap q[50],q[36];
cx q[31],q[22];
swap q[25],q[18];
swap q[30],q[23];
cx q[20],q[18];
swap q[40],q[32];
swap q[43],q[33];
swap q[58],q[57];
swap q[20],q[13];
swap q[31],q[15];
cx q[12],q[36];
swap q[60],q[58];
cx q[27],q[30];
swap q[16],q[8];
swap q[17],q[16];
swap q[33],q[32];
cx q[27],q[33];
cx q[30],q[46];
swap q[52],q[50];
cx q[31],q[30];
swap q[12],q[10];
swap q[35],q[26];
swap q[38],q[31];
cx q[10],q[34];
swap q[61],q[59];
cx q[37],q[35];
swap q[10],q[4];
cx q[46],q[43];
cx q[60],q[46];
swap q[33],q[25];
swap q[28],q[20];
cx q[38],q[52];
swap q[42],q[35];
swap q[32],q[17];
cx q[35],q[28];
swap q[47],q[45];
cx q[27],q[10];
swap q[14],q[6];
swap q[23],q[14];
swap q[60],q[58];
swap q[30],q[23];
swap q[21],q[13];
swap q[41],q[32];
cx q[27],q[44];
cx q[45],q[35];
swap q[16],q[8];
swap q[61],q[60];
cx q[58],q[41];
cx q[45],q[30];
swap q[12],q[11];
swap q[35],q[26];
cx q[58],q[60];
cx q[45],q[31];
swap q[42],q[41];
cx q[12],q[36];
swap q[59],q[51];
cx q[26],q[20];
swap q[47],q[31];
cx q[51],q[34];
swap q[20],q[4];
cx q[42],q[60];
cx q[26],q[16];
swap q[31],q[14];
cx q[19],q[43];
swap q[54],q[46];
swap q[33],q[32];
cx q[46],q[28];
cx q[12],q[14];
cx q[60],q[43];
swap q[46],q[37];
cx q[25],q[10];
cx q[27],q[21];
swap q[50],q[41];
cx q[21],q[45];
swap q[10],q[3];
cx q[25],q[41];
swap q[28],q[13];
swap q[30],q[23];
swap q[51],q[44];
swap q[8],q[2];
cx q[13],q[5];
cx q[42],q[28];
swap q[47],q[23];
cx q[10],q[34];
cx q[28],q[18];
swap q[13],q[5];
swap q[42],q[33];
cx q[5],q[2];
swap q[62],q[38];
swap q[27],q[13];
swap q[40],q[33];
swap q[44],q[37];
swap q[10],q[1];
swap q[33],q[19];
swap q[54],q[46];
swap q[14],q[5];
swap q[37],q[30];
swap q[44],q[43];
swap q[26],q[10];
cx q[42],q[43];
cx q[14],q[38];
cx q[20],q[30];
swap q[32],q[16];
swap q[43],q[42];
cx q[27],q[36];
cx q[30],q[47];
cx q[28],q[22];
swap q[5],q[4];
swap q[32],q[25];
cx q[20],q[36];
swap q[30],q[22];
cx q[42],q[26];
cx q[27],q[51];
swap q[4],q[3];
swap q[46],q[45];
cx q[20],q[19];
cx q[42],q[25];
cx q[22],q[5];
swap q[10],q[3];
cx q[37],q[34];
cx q[34],q[52];
swap q[20],q[19];
cx q[37],q[23];
swap q[50],q[34];
swap q[16],q[1];
swap q[38],q[20];
swap q[53],q[52];
swap q[14],q[6];
swap q[34],q[17];
cx q[45],q[38];
swap q[28],q[21];
cx q[26],q[10];
swap q[45],q[31];
cx q[20],q[34];
cx q[10],q[17];
cx q[34],q[51];
cx q[16],q[17];
cx q[31],q[14];
swap q[53],q[46];
cx q[26],q[41];
cx q[11],q[25];
swap q[31],q[14];
cx q[28],q[43];
swap q[17],q[3];
cx q[46],q[30];
cx q[47],q[31];
swap q[51],q[41];
cx q[13],q[3];
swap q[45],q[36];
cx q[52],q[51];
swap q[19],q[18];
cx q[51],q[50];
cx q[34],q[36];
swap q[24],q[16];
swap q[31],q[30];
swap q[12],q[4];
cx q[52],q[59];
cx q[25],q[34];
swap q[38],q[28];
swap q[18],q[10];
swap q[50],q[49];
swap q[33],q[24];
cx q[28],q[35];
swap q[59],q[52];
swap q[26],q[18];
cx q[35],q[44];
cx q[35],q[36];
swap q[22],q[12];
swap q[41],q[24];
cx q[3],q[27];
swap q[58],q[56];
swap q[52],q[46];
cx q[24],q[26];
swap q[21],q[3];
cx q[49],q[33];
swap q[60],q[43];
cx q[24],q[10];
cx q[22],q[46];
swap q[56],q[48];
cx q[22],q[12];
cx q[27],q[43];
swap q[17],q[10];
cx q[34],q[28];
cx q[34],q[36];
swap q[62],q[60];
swap q[46],q[38];
swap q[12],q[4];
cx q[41],q[17];
swap q[38],q[22];
cx q[41],q[42];
cx q[27],q[18];
cx q[17],q[32];
swap q[60],q[59];
swap q[53],q[44];
swap q[4],q[2];
cx q[41],q[59];
swap q[37],q[27];
cx q[24],q[48];
swap q[47],q[31];
swap q[15],q[14];
swap q[61],q[59];
cx q[26],q[27];
cx q[4],q[22];
swap q[10],q[1];
swap q[49],q[42];
cx q[61],q[47];
swap q[33],q[32];
swap q[44],q[37];
swap q[18],q[4];
cx q[38],q[52];
swap q[42],q[33];
cx q[62],q[52];
cx q[38],q[31];
swap q[17],q[11];
cx q[21],q[37];
swap q[35],q[27];
swap q[61],q[52];
cx q[33],q[17];
cx q[20],q[23];
swap q[47],q[46];
cx q[52],q[35];
cx q[13],q[20];
swap q[17],q[3];
cx q[28],q[20];
cx q[28],q[27];
swap q[30],q[22];
swap q[45],q[42];
swap q[15],q[6];
swap q[12],q[10];
cx q[45],q[30];
swap q[43],q[34];
swap q[61],q[60];
swap q[46],q[29];
cx q[34],q[10];
swap q[13],q[6];
cx q[35],q[17];
swap q[60],q[52];
cx q[17],q[3];
cx q[20],q[27];
swap q[45],q[38];
swap q[42],q[34];
cx q[29],q[12];
swap q[17],q[10];
swap q[48],q[40];
swap q[40],q[32];
swap q[54],q[47];
cx q[10],q[13];
swap q[52],q[44];
swap q[32],q[24];
swap q[38],q[30];
cx q[15],q[13];
swap q[42],q[35];
swap q[24],q[10];
cx q[35],q[53];
cx q[62],q[53];
swap q[49],q[48];
swap q[30],q[15];
swap q[49],q[41];
swap q[51],q[35];
cx q[30],q[47];
swap q[25],q[18];
cx q[15],q[5];
cx q[30],q[31];
cx q[60],q[51];
swap q[35],q[27];
cx q[13],q[10];
swap q[47],q[45];
swap q[41],q[24];
cx q[19],q[22];
swap q[45],q[37];
cx q[10],q[24];
cx q[27],q[25];
swap q[22],q[5];
swap q[35],q[17];
swap q[53],q[46];
swap q[23],q[21];
cx q[44],q[35];
swap q[5],q[4];
swap q[24],q[10];
cx q[37],q[22];
cx q[35],q[34];
cx q[22],q[46];
swap q[35],q[34];
cx q[10],q[12];
cx q[29],q[19];
swap q[61],q[47];
cx q[18],q[4];
swap q[35],q[28];
cx q[10],q[4];
swap q[22],q[14];
swap q[47],q[38];
cx q[27],q[21];
swap q[45],q[44];
cx q[31],q[28];
swap q[42],q[25];
swap q[22],q[12];
swap q[10],q[1];
swap q[36],q[27];
swap q[9],q[2];
swap q[32],q[25];
cx q[22],q[38];
swap q[43],q[42];
swap q[11],q[5];
cx q[37],q[43];
swap q[41],q[32];
cx q[26],q[12];
cx q[28],q[43];
swap q[38],q[30];
cx q[10],q[12];
swap q[41],q[35];
cx q[12],q[19];
swap q[25],q[24];
swap q[30],q[14];
cx q[10],q[27];
swap q[44],q[42];
cx q[10],q[11];
cx q[3],q[19];
swap q[24],q[16];
swap q[29],q[22];
cx q[17],q[11];
swap q[43],q[40];
cx q[28],q[29];
cx q[14],q[4];
swap q[16],q[1];
cx q[28],q[35];
swap q[31],q[13];
cx q[37],q[44];
cx q[19],q[26];
swap q[2],q[1];
cx q[28],q[25];
swap q[22],q[14];
cx q[2],q[3];
swap q[36],q[33];
cx q[3],q[6];
cx q[9],q[27];
swap q[45],q[30];
cx q[18],q[42];
swap q[27],q[10];
swap q[15],q[14];
swap q[38],q[22];
cx q[35],q[27];
cx q[2],q[10];
swap q[20],q[12];
cx q[16],q[10];
cx q[29],q[36];
cx q[42],q[45];
cx q[2],q[17];
swap q[22],q[14];
cx q[35],q[20];
swap q[18],q[9];
cx q[3],q[13];
swap q[48],q[41];
swap q[38],q[37];
swap q[28],q[27];
swap q[31],q[23];
swap q[12],q[5];
cx q[28],q[22];
swap q[45],q[43];
cx q[12],q[22];
swap q[33],q[17];
cx q[37],q[45];
cx q[43],q[34];
swap q[14],q[11];
swap q[40],q[24];
swap q[60],q[52];
swap q[39],q[31];
swap q[42],q[35];
cx q[22],q[14];
cx q[10],q[11];
swap q[37],q[29];
swap q[32],q[16];
cx q[14],q[6];
cx q[28],q[46];
cx q[14],q[29];
swap q[52],q[43];
cx q[18],q[21];
swap q[46],q[39];
swap q[41],q[33];
cx q[5],q[29];
swap q[16],q[10];
swap q[43],q[35];
swap q[6],q[4];
cx q[43],q[46];
swap q[21],q[20];
swap q[24],q[18];
swap q[43],q[37];
cx q[4],q[18];
cx q[35],q[11];
cx q[35],q[18];
swap q[5],q[3];
cx q[43],q[25];
swap q[61],q[60];
swap q[49],q[40];
swap q[60],q[53];
swap q[53],q[46];
swap q[46],q[28];
cx q[3],q[10];
swap q[40],q[25];
swap q[11],q[3];
swap q[51],q[42];
swap q[33],q[27];
swap q[22],q[15];
swap q[47],q[45];
swap q[14],q[13];
swap q[17],q[9];
swap q[30],q[14];
cx q[27],q[20];
swap q[4],q[3];
cx q[27],q[45];
swap q[50],q[40];
cx q[45],q[47];
cx q[19],q[10];
swap q[24],q[16];
swap q[21],q[14];
cx q[11],q[25];
cx q[38],q[45];
swap q[61],q[51];
cx q[38],q[36];
cx q[12],q[9];
cx q[10],q[28];
swap q[33],q[32];
cx q[44],q[20];
swap q[47],q[31];
cx q[33],q[27];
swap q[6],q[5];
swap q[10],q[9];
cx q[44],q[62];
cx q[20],q[22];
swap q[5],q[4];
cx q[42],q[25];
cx q[22],q[21];
cx q[31],q[30];
cx q[19],q[16];
swap q[62],q[46];
cx q[20],q[23];
swap q[51],q[44];
cx q[31],q[39];
swap q[34],q[33];
swap q[6],q[5];
cx q[36],q[50];
swap q[19],q[10];
swap q[54],q[53];
swap q[31],q[22];
cx q[37],q[19];
swap q[13],q[5];
cx q[31],q[45];
swap q[34],q[25];
swap q[28],q[19];
cx q[22],q[6];
swap q[61],q[45];
swap q[25],q[9];
swap q[4],q[3];
swap q[21],q[13];
cx q[9],q[3];
swap q[31],q[15];
swap q[35],q[27];
cx q[37],q[22];
swap q[61],q[51];
cx q[19],q[3];
cx q[37],q[45];
swap q[23],q[6];
cx q[19],q[34];
swap q[53],q[37];
swap q[17],q[16];
swap q[27],q[17];
swap q[11],q[3];
swap q[51],q[43];
swap q[30],q[21];
swap q[46],q[45];
swap q[26],q[11];
cx q[28],q[37];
swap q[6],q[5];
swap q[50],q[42];
swap q[32],q[16];
swap q[54],q[52];
cx q[37],q[27];
cx q[11],q[13];
swap q[41],q[32];
cx q[45],q[35];
cx q[11],q[5];
cx q[26],q[43];
cx q[46],q[30];
swap q[20],q[12];
cx q[37],q[52];
cx q[46],q[31];
swap q[14],q[5];
swap q[35],q[27];
cx q[23],q[31];
swap q[50],q[41];
swap q[52],q[45];
cx q[21],q[18];
swap q[55],q[39];
cx q[52],q[50];
swap q[27],q[21];
cx q[41],q[50];
swap q[54],q[38];
cx q[21],q[5];
swap q[51],q[41];
cx q[38],q[14];
swap q[62],q[61];
cx q[20],q[44];
cx q[36],q[18];
swap q[31],q[15];
cx q[54],q[51];
swap q[12],q[5];
cx q[51],q[61];
cx q[22],q[38];
swap q[25],q[9];
cx q[61],q[52];
swap q[35],q[28];
swap q[38],q[31];
cx q[61],q[43];
swap q[14],q[12];
cx q[25],q[43];
swap q[29],q[14];
cx q[43],q[52];
cx q[18],q[42];
swap q[62],q[55];
cx q[44],q[29];
cx q[18],q[9];
swap q[58],q[48];
cx q[45],q[28];
cx q[19],q[37];
swap q[62],q[61];
swap q[14],q[13];
swap q[33],q[25];
swap q[16],q[9];
cx q[44],q[58];
cx q[19],q[12];
swap q[34],q[33];
cx q[36],q[29];
cx q[19],q[11];
cx q[22],q[28];
swap q[54],q[47];
swap q[60],q[50];
swap q[11],q[9];
cx q[34],q[44];
cx q[61],q[60];
swap q[6],q[5];
swap q[47],q[30];
swap q[58],q[50];
cx q[27],q[45];
swap q[5],q[2];
cx q[45],q[42];
swap q[31],q[23];
cx q[42],q[26];
swap q[28],q[20];
cx q[43],q[46];
swap q[61],q[60];
cx q[11],q[29];
swap q[23],q[14];
cx q[11],q[2];
cx q[27],q[35];
swap q[25],q[24];
swap q[58],q[41];
cx q[11],q[3];
cx q[27],q[17];
swap q[53],q[38];
cx q[2],q[9];
swap q[14],q[5];
cx q[58],q[60];
cx q[30],q[28];
swap q[34],q[25];
cx q[51],q[53];
cx q[2],q[5];
swap q[36],q[30];
swap q[18],q[4];
cx q[60],q[44];
swap q[50],q[42];
swap q[30],q[29];
cx q[25],q[18];
swap q[52],q[35];
swap q[5],q[3];
cx q[35],q[18];
swap q[20],q[13];
cx q[18],q[10];
cx q[36],q[34];
cx q[9],q[3];
swap q[42],q[36];
swap q[55],q[47];
swap q[47],q[46];
swap q[46],q[30];
swap q[59],q[58];
swap q[32],q[25];
swap q[30],q[22];
swap q[12],q[10];
swap q[55],q[46];
swap q[59],q[52];
swap q[35],q[25];
swap q[37],q[29];
swap q[55],q[54];
cx q[20],q[36];
swap q[26],q[24];
cx q[12],q[20];
cx q[12],q[22];
swap q[61],q[47];
cx q[36],q[35];
cx q[46],q[22];
swap q[18],q[17];
cx q[22],q[6];
cx q[45],q[28];
cx q[28],q[44];
swap q[15],q[13];
cx q[35],q[26];
cx q[37],q[34];
cx q[29],q[35];
swap q[14],q[12];
cx q[35],q[18];
swap q[54],q[52];
swap q[35],q[28];
swap q[50],q[42];
swap q[54],q[38];
cx q[28],q[12];
swap q[18],q[10];
swap q[23],q[22];
swap q[52],q[44];
cx q[38],q[20];
swap q[43],q[33];
cx q[29],q[47];
cx q[46],q[47];
cx q[13],q[10];
cx q[42],q[28];
cx q[45],q[46];
cx q[45],q[37];
swap q[22],q[13];
cx q[44],q[34];
swap q[18],q[11];
swap q[47],q[39];
swap q[26],q[12];
swap q[54],q[45];
swap q[31],q[30];
cx q[13],q[12];
swap q[44],q[26];
cx q[30],q[13];
cx q[13],q[15];
cx q[39],q[21];
cx q[18],q[26];
swap q[53],q[43];
swap q[32],q[16];
swap q[29],q[12];
swap q[27],q[26];
swap q[23],q[14];
swap q[50],q[32];
cx q[45],q[29];
cx q[11],q[29];
swap q[61],q[54];
cx q[15],q[29];
cx q[30],q[44];
swap q[24],q[10];
cx q[38],q[53];
swap q[20],q[13];
cx q[11],q[10];
cx q[59],q[53];
cx q[39],q[38];
swap q[42],q[35];
swap q[46],q[29];
cx q[59],q[43];
swap q[6],q[3];
swap q[31],q[23];
swap q[35],q[26];
cx q[3],q[10];
cx q[54],q[46];
cx q[3],q[5];
cx q[39],q[46];
swap q[59],q[52];
swap q[32],q[16];
swap q[49],q[32];
swap q[23],q[22];
cx q[53],q[50];
cx q[26],q[11];
cx q[20],q[36];
cx q[26],q[10];
swap q[47],q[23];
cx q[50],q[43];
cx q[28],q[11];
cx q[33],q[50];
cx q[28],q[10];
cx q[5],q[21];
cx q[54],q[47];
cx q[33],q[51];
swap q[38],q[37];
cx q[5],q[4];
swap q[52],q[35];
swap q[22],q[20];
swap q[5],q[4];
cx q[60],q[36];
swap q[47],q[39];
cx q[37],q[35];
cx q[11],q[20];
swap q[22],q[5];
cx q[35],q[49];
cx q[27],q[45];
swap q[16],q[9];
swap q[60],q[59];
swap q[10],q[9];
cx q[22],q[39];
swap q[44],q[37];
swap q[27],q[20];
swap q[61],q[47];
cx q[29],q[19];
cx q[21],q[37];
cx q[61],q[52];
swap q[18],q[3];
cx q[44],q[60];
cx q[29],q[30];
cx q[53],q[60];
cx q[47],q[37];
swap q[20],q[13];
swap q[24],q[16];
swap q[33],q[24];
swap q[16],q[8];
swap q[8],q[1];
swap q[45],q[31];
cx q[13],q[14];
swap q[50],q[34];
swap q[11],q[10];
swap q[13],q[6];
cx q[53],q[45];
cx q[35],q[19];
cx q[53],q[62];
swap q[2],q[1];
cx q[20],q[36];
swap q[60],q[59];
swap q[25],q[18];
cx q[4],q[20];
swap q[46],q[38];
swap q[59],q[43];
cx q[20],q[18];
cx q[4],q[2];
swap q[22],q[14];
swap q[46],q[29];
cx q[43],q[25];
swap q[3],q[1];
cx q[34],q[36];
swap q[20],q[13];
cx q[34],q[17];
cx q[36],q[42];
swap q[60],q[52];
swap q[54],q[39];
swap q[17],q[2];
cx q[45],q[21];
swap q[39],q[31];
cx q[14],q[11];
swap q[45],q[35];
cx q[4],q[14];
swap q[10],q[2];
cx q[36],q[33];
swap q[31],q[14];
cx q[35],q[17];
swap q[52],q[45];
cx q[20],q[10];
swap q[26],q[25];
cx q[11],q[29];
swap q[46],q[39];
cx q[20],q[27];
swap q[14],q[4];
cx q[27],q[9];
swap q[45],q[38];
cx q[20],q[12];
cx q[13],q[12];
swap q[35],q[17];
cx q[5],q[12];
swap q[38],q[23];
cx q[12],q[4];
cx q[4],q[2];
swap q[49],q[41];
swap q[28],q[26];
cx q[5],q[23];
swap q[59],q[50];
swap q[50],q[43];
cx q[21],q[28];
swap q[11],q[5];
swap q[46],q[44];
swap q[41],q[25];
swap q[31],q[23];
swap q[29],q[21];
cx q[44],q[35];
cx q[11],q[25];
swap q[46],q[31];
cx q[12],q[6];
swap q[27],q[18];
cx q[21],q[3];
cx q[44],q[46];
swap q[15],q[6];
cx q[37],q[27];
swap q[12],q[10];
swap q[54],q[47];
swap q[37],q[27];
swap q[23],q[6];
swap q[61],q[60];
swap q[60],q[52];
swap q[61],q[55];
cx q[37],q[23];
cx q[4],q[18];
cx q[27],q[3];
swap q[30],q[15];
swap q[45],q[44];
swap q[4],q[1];
cx q[29],q[43];
swap q[47],q[39];
cx q[5],q[15];
cx q[12],q[22];
swap q[44],q[26];
cx q[39],q[22];
cx q[26],q[19];
cx q[14],q[5];
swap q[47],q[44];
cx q[14],q[20];
cx q[6],q[12];
cx q[26],q[44];
cx q[20],q[29];
swap q[14],q[4];
swap q[55],q[39];
swap q[59],q[50];
swap q[35],q[33];
cx q[14],q[15];
swap q[53],q[52];
cx q[39],q[23];
swap q[50],q[36];
swap q[12],q[6];
swap q[54],q[39];
swap q[23],q[22];
swap q[36],q[12];
swap q[62],q[47];
cx q[14],q[12];
swap q[35],q[29];
swap q[25],q[18];
swap q[47],q[39];
cx q[22],q[29];
swap q[13],q[12];
swap q[54],q[45];
swap q[19],q[18];
cx q[45],q[51];
cx q[14],q[31];
cx q[51],q[43];
cx q[14],q[23];
swap q[54],q[38];
cx q[42],q[43];
swap q[20],q[10];
cx q[23],q[38];
swap q[33],q[24];
swap q[33],q[26];
swap q[43],q[36];
swap q[31],q[13];
cx q[43],q[53];
swap q[44],q[26];
cx q[31],q[39];
swap q[21],q[3];
cx q[39],q[46];
swap q[35],q[33];
cx q[31],q[21];
swap q[18],q[11];
cx q[42],q[28];
swap q[46],q[45];
swap q[22],q[13];
swap q[37],q[35];
swap q[14],q[11];
swap q[46],q[36];
swap q[23],q[14];
cx q[29],q[19];
swap q[52],q[45];
cx q[22],q[39];
cx q[22],q[13];
cx q[39],q[55];
swap q[35],q[34];
swap q[25],q[24];
swap q[17],q[9];
swap q[19],q[9];
cx q[37],q[31];
swap q[62],q[59];
cx q[30],q[27];
cx q[26],q[12];
cx q[39],q[23];
swap q[45],q[37];
cx q[30],q[21];
cx q[26],q[25];
cx q[33],q[25];
swap q[59],q[49];
cx q[37],q[13];
swap q[25],q[10];
cx q[45],q[51];
cx q[51],q[35];
swap q[31],q[13];
cx q[33],q[18];
swap q[49],q[48];
swap q[53],q[45];
cx q[35],q[29];
swap q[23],q[14];
cx q[20],q[38];
cx q[51],q[41];
cx q[38],q[44];
swap q[10],q[3];
swap q[26],q[25];
cx q[13],q[29];
cx q[44],q[42];
swap q[61],q[60];
cx q[13],q[19];
swap q[39],q[23];
cx q[35],q[17];
swap q[13],q[3];
cx q[37],q[52];
cx q[24],q[17];
cx q[24],q[48];
cx q[39],q[46];
swap q[27],q[19];
cx q[29],q[14];
swap q[52],q[43];
cx q[13],q[14];
swap q[62],q[61];
swap q[39],q[31];
cx q[45],q[27];
swap q[62],q[55];
cx q[26],q[43];
cx q[14],q[20];
cx q[44],q[41];
cx q[39],q[55];
swap q[36],q[12];
swap q[26],q[18];
swap q[61],q[59];
swap q[23],q[15];
swap q[55],q[39];
cx q[36],q[43];
cx q[11],q[12];
swap q[25],q[9];
cx q[39],q[21];
cx q[45],q[62];
cx q[39],q[47];
swap q[27],q[12];
cx q[62],q[47];
swap q[30],q[23];
swap q[33],q[25];
swap q[47],q[45];
cx q[12],q[5];
swap q[33],q[27];
swap q[59],q[57];
swap q[57],q[51];
swap q[30],q[29];
swap q[3],q[2];
swap q[14],q[5];
cx q[36],q[34];
swap q[46],q[39];
cx q[34],q[26];
cx q[21],q[3];
cx q[45],q[27];
swap q[31],q[14];
cx q[26],q[50];
cx q[50],q[51];
cx q[17],q[3];
swap q[37],q[29];
cx q[31],q[39];
cx q[50],q[33];
swap q[53],q[46];
swap q[19],q[5];
cx q[45],q[43];
swap q[53],q[43];
cx q[19],q[25];
swap q[30],q[6];
cx q[51],q[33];
swap q[19],q[10];
swap q[44],q[30];
swap q[6],q[4];
swap q[17],q[10];
cx q[51],q[44];
swap q[55],q[47];
swap q[27],q[19];
swap q[48],q[33];
swap q[47],q[30];
cx q[4],q[10];
swap q[35],q[33];
swap q[14],q[6];
swap q[52],q[51];
swap q[12],q[10];
cx q[29],q[27];
swap q[23],q[14];
cx q[42],q[27];
swap q[52],q[45];
cx q[42],q[28];
swap q[17],q[1];
cx q[35],q[28];
swap q[11],q[4];
swap q[31],q[14];
cx q[28],q[19];
swap q[48],q[32];
cx q[12],q[30];
swap q[44],q[42];
cx q[30],q[38];
cx q[12],q[2];
swap q[18],q[17];
cx q[29],q[45];
swap q[32],q[24];
cx q[53],q[38];
cx q[42],q[27];
swap q[15],q[14];
swap q[12],q[5];
cx q[34],q[37];
swap q[24],q[16];
cx q[19],q[37];
cx q[34],q[43];
cx q[37],q[39];
cx q[34],q[42];
swap q[13],q[10];
cx q[46],q[39];
cx q[46],q[36];
swap q[42],q[33];
cx q[29],q[13];
swap q[46],q[39];
cx q[34],q[10];
cx q[30],q[22];
cx q[33],q[25];
cx q[11],q[27];
swap q[53],q[52];
swap q[22],q[14];
swap q[16],q[9];
swap q[37],q[29];
swap q[43],q[42];
swap q[27],q[12];
cx q[39],q[23];
swap q[52],q[43];
cx q[11],q[9];
swap q[45],q[37];
swap q[47],q[31];
cx q[45],q[52];
cx q[35],q[27];
swap q[4],q[1];
cx q[37],q[47];
swap q[43],q[27];
cx q[4],q[13];
cx q[37],q[22];
swap q[54],q[47];
cx q[27],q[18];
cx q[43],q[41];
cx q[13],q[29];
cx q[53],q[35];
swap q[15],q[12];
cx q[53],q[54];
swap q[36],q[26];
swap q[24],q[16];
swap q[32],q[24];
swap q[53],q[46];
cx q[12],q[18];
swap q[41],q[32];
cx q[53],q[36];
swap q[31],q[22];
swap q[19],q[12];
swap q[10],q[2];
swap q[53],q[43];
cx q[12],q[22];
cx q[43],q[41];
cx q[54],q[53];
swap q[26],q[20];
swap q[23],q[6];
swap q[52],q[43];
cx q[26],q[41];
cx q[12],q[6];
swap q[62],q[46];
swap q[30],q[23];
cx q[26],q[28];
cx q[62],q[52];
cx q[19],q[43];
swap q[6],q[5];
cx q[20],q[28];
swap q[33],q[25];
swap q[42],q[35];
swap q[53],q[39];
swap q[2],q[1];
swap q[11],q[2];
swap q[5],q[4];
cx q[18],q[36];
swap q[33],q[32];
swap q[39],q[23];
cx q[43],q[33];
swap q[62],q[55];
swap q[29],q[19];
swap q[4],q[2];
swap q[43],q[41];
cx q[29],q[46];
swap q[25],q[17];
swap q[11],q[5];
swap q[46],q[29];
cx q[17],q[2];
swap q[52],q[43];
cx q[19],q[29];
swap q[34],q[33];
swap q[15],q[6];
cx q[35],q[21];
swap q[24],q[17];
cx q[20],q[21];
cx q[35],q[42];
swap q[55],q[39];
swap q[52],q[45];
cx q[21],q[31];
swap q[2],q[1];
swap q[17],q[3];
cx q[45],q[30];
cx q[31],q[23];
swap q[42],q[34];
swap q[44],q[28];
cx q[34],q[17];
cx q[23],q[5];
cx q[39],q[31];
swap q[58],q[57];
swap q[59],q[58];
swap q[60],q[59];
swap q[11],q[2];
cx q[29],q[14];
swap q[34],q[17];
cx q[12],q[28];
cx q[6],q[14];
swap q[61],q[60];
swap q[51],q[44];
cx q[22],q[28];
swap q[61],q[46];
cx q[34],q[50];
cx q[6],q[3];
swap q[40],q[24];
swap q[30],q[29];
cx q[34],q[44];
swap q[12],q[10];
cx q[43],q[40];
swap q[62],q[47];
cx q[29],q[11];
swap q[26],q[25];
cx q[12],q[36];
swap q[50],q[42];
cx q[22],q[46];
swap q[25],q[11];
swap q[50],q[36];
cx q[25],q[32];
swap q[47],q[23];
swap q[11],q[5];
cx q[38],q[36];
swap q[53],q[51];
cx q[29],q[26];
swap q[23],q[15];
cx q[28],q[26];
cx q[53],q[46];
swap q[23],q[13];
cx q[42],q[26];
swap q[32],q[16];
swap q[11],q[10];
swap q[46],q[36];
cx q[12],q[15];
swap q[42],q[26];
cx q[39],q[23];
swap q[16],q[1];
swap q[13],q[11];
swap q[36],q[29];
swap q[42],q[41];
cx q[26],q[10];
swap q[55],q[54];
swap q[29],q[5];
swap q[34],q[18];
swap q[5],q[3];
cx q[43],q[34];
cx q[42],q[36];
swap q[46],q[30];
cx q[1],q[11];
cx q[42],q[51];
cx q[36],q[54];
swap q[30],q[21];
cx q[3],q[9];
cx q[51],q[53];
swap q[33],q[25];
swap q[27],q[19];
cx q[51],q[33];
swap q[9],q[2];
cx q[27],q[44];
swap q[21],q[5];
cx q[33],q[9];
swap q[44],q[37];
swap q[61],q[54];
swap q[54],q[39];
swap q[5],q[2];
cx q[33],q[35];
swap q[55],q[45];
cx q[10],q[2];
cx q[10],q[34];
swap q[31],q[23];
cx q[21],q[37];
cx q[1],q[2];
swap q[50],q[34];
cx q[21],q[15];
cx q[39],q[21];
swap q[61],q[60];
swap q[60],q[52];
swap q[17],q[2];
cx q[21],q[38];
swap q[52],q[50];
cx q[11],q[28];
cx q[17],q[27];
cx q[15],q[5];
swap q[39],q[38];
cx q[11],q[18];
swap q[28],q[22];
cx q[34],q[18];
cx q[12],q[5];
swap q[54],q[52];
swap q[18],q[17];
cx q[38],q[44];
cx q[38],q[55];
swap q[11],q[2];
swap q[40],q[32];
swap q[50],q[49];
swap q[33],q[32];
swap q[22],q[13];
cx q[18],q[28];
cx q[54],q[47];
swap q[43],q[42];
cx q[21],q[37];
cx q[27],q[11];
swap q[49],q[40];
cx q[53],q[29];
swap q[47],q[39];
cx q[36],q[20];
swap q[24],q[17];
swap q[42],q[41];
cx q[5],q[23];
cx q[5],q[14];
swap q[46],q[36];
cx q[24],q[40];
swap q[60],q[51];
swap q[18],q[10];
swap q[22],q[12];
cx q[34],q[36];
swap q[53],q[46];
cx q[28],q[42];
swap q[10],q[4];
cx q[22],q[39];
swap q[45],q[43];
cx q[22],q[20];
swap q[25],q[10];
cx q[45],q[30];
swap q[15],q[13];
cx q[42],q[25];
swap q[37],q[28];
swap q[31],q[30];
swap q[43],q[25];
swap q[47],q[46];
swap q[11],q[3];
swap q[21],q[13];
swap q[46],q[37];
cx q[18],q[25];
swap q[31],q[15];
swap q[21],q[12];
cx q[1],q[25];
cx q[46],q[39];
cx q[27],q[33];
swap q[44],q[43];
swap q[47],q[38];
cx q[18],q[11];
cx q[10],q[28];
swap q[42],q[32];
cx q[38],q[21];
swap q[27],q[11];
cx q[32],q[16];
swap q[23],q[15];
swap q[15],q[6];
swap q[55],q[53];
swap q[51],q[42];
cx q[37],q[27];
swap q[17],q[9];
cx q[37],q[53];
swap q[22],q[14];
swap q[12],q[11];
cx q[26],q[29];
swap q[53],q[51];
cx q[36],q[19];
cx q[36],q[35];
swap q[25],q[17];
cx q[35],q[42];
swap q[39],q[22];
swap q[19],q[10];
cx q[44],q[30];
swap q[40],q[32];
swap q[34],q[25];
swap q[29],q[15];
cx q[34],q[51];
swap q[18],q[10];
swap q[46],q[39];
swap q[32],q[24];
cx q[19],q[43];
cx q[26],q[20];
cx q[44],q[46];
cx q[25],q[11];
swap q[36],q[29];
cx q[26],q[42];
swap q[32],q[16];
cx q[27],q[3];
cx q[34],q[36];
cx q[3],q[6];
swap q[21],q[20];
swap q[35],q[32];
swap q[38],q[23];
cx q[35],q[36];
swap q[26],q[18];
swap q[42],q[33];
swap q[24],q[16];
cx q[43],q[26];
cx q[35],q[38];
swap q[11],q[9];
cx q[36],q[42];
cx q[46],q[38];
cx q[28],q[31];
cx q[17],q[20];
cx q[15],q[31];
cx q[26],q[9];
cx q[9],q[33];
cx q[30],q[20];
cx q[19],q[16];
