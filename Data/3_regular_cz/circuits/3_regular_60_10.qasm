OPENQASM 2.0;
include "qelib1.inc";
qreg q[60];
cz q[25],q[32];
cz q[25],q[9];
cz q[25],q[44];
cz q[32],q[3];
cz q[32],q[58];
cz q[18],q[26];
cz q[18],q[23];
cz q[18],q[29];
cz q[26],q[7];
cz q[26],q[36];
cz q[24],q[42];
cz q[24],q[36];
cz q[24],q[40];
cz q[42],q[39];
cz q[42],q[16];
cz q[7],q[14];
cz q[7],q[23];
cz q[39],q[55];
cz q[39],q[29];
cz q[38],q[44];
cz q[38],q[50];
cz q[38],q[1];
cz q[44],q[4];
cz q[0],q[5];
cz q[0],q[45];
cz q[0],q[8];
cz q[5],q[6];
cz q[5],q[27];
cz q[14],q[31];
cz q[14],q[28];
cz q[31],q[47];
cz q[31],q[54];
cz q[47],q[10];
cz q[47],q[58];
cz q[17],q[30];
cz q[17],q[54];
cz q[17],q[37];
cz q[30],q[50];
cz q[30],q[6];
cz q[12],q[43];
cz q[12],q[1];
cz q[12],q[2];
cz q[43],q[4];
cz q[43],q[19];
cz q[40],q[59];
cz q[40],q[22];
cz q[59],q[46];
cz q[59],q[36];
cz q[33],q[56];
cz q[33],q[2];
cz q[33],q[57];
cz q[56],q[53];
cz q[56],q[35];
cz q[46],q[22];
cz q[46],q[51];
cz q[22],q[9];
cz q[15],q[16];
cz q[15],q[48];
cz q[15],q[35];
cz q[16],q[52];
cz q[8],q[41];
cz q[8],q[54];
cz q[41],q[34];
cz q[41],q[10];
cz q[11],q[37];
cz q[11],q[57];
cz q[11],q[52];
cz q[37],q[20];
cz q[50],q[3];
cz q[10],q[3];
cz q[9],q[49];
cz q[49],q[45];
cz q[49],q[13];
cz q[23],q[29];
cz q[4],q[28];
cz q[19],q[52];
cz q[19],q[35];
cz q[57],q[13];
cz q[21],q[34];
cz q[21],q[28];
cz q[21],q[53];
cz q[34],q[45];
cz q[48],q[20];
cz q[48],q[27];
cz q[13],q[51];
cz q[51],q[27];
cz q[1],q[55];
cz q[20],q[55];
cz q[2],q[58];
cz q[53],q[6];
cz q[16],q[29];
cz q[16],q[47];
cz q[16],q[52];
cz q[29],q[35];
cz q[29],q[37];
cz q[24],q[42];
cz q[24],q[58];
cz q[24],q[19];
cz q[42],q[52];
cz q[42],q[46];
cz q[30],q[55];
cz q[30],q[28];
cz q[30],q[6];
cz q[55],q[39];
cz q[55],q[6];
cz q[0],q[51];
cz q[0],q[2];
cz q[0],q[36];
cz q[51],q[19];
cz q[51],q[26];
cz q[47],q[49];
cz q[47],q[17];
cz q[14],q[22];
cz q[14],q[21];
cz q[14],q[50];
cz q[22],q[20];
cz q[22],q[15];
cz q[12],q[34];
cz q[12],q[9];
cz q[12],q[49];
cz q[34],q[21];
cz q[34],q[5];
cz q[28],q[18];
cz q[28],q[44];
cz q[19],q[48];
cz q[48],q[38];
cz q[48],q[33];
cz q[18],q[20];
cz q[18],q[49];
cz q[20],q[57];
cz q[9],q[56];
cz q[9],q[5];
cz q[56],q[10];
cz q[56],q[43];
cz q[2],q[4];
cz q[2],q[36];
cz q[4],q[13];
cz q[4],q[8];
cz q[23],q[54];
cz q[23],q[5];
cz q[23],q[31];
cz q[54],q[8];
cz q[54],q[38];
cz q[1],q[17];
cz q[1],q[35];
cz q[1],q[3];
cz q[17],q[57];
cz q[41],q[44];
cz q[41],q[13];
cz q[41],q[45];
cz q[44],q[40];
cz q[35],q[27];
cz q[52],q[11];
cz q[25],q[45];
cz q[25],q[38];
cz q[25],q[46];
cz q[45],q[50];
cz q[10],q[33];
cz q[10],q[59];
cz q[32],q[53];
cz q[32],q[43];
cz q[32],q[40];
cz q[53],q[7];
cz q[53],q[31];
cz q[58],q[37];
cz q[58],q[33];
cz q[37],q[59];
cz q[36],q[39];
cz q[21],q[11];
cz q[31],q[6];
cz q[3],q[7];
cz q[3],q[11];
cz q[7],q[8];
cz q[57],q[39];
cz q[13],q[27];
cz q[15],q[43];
cz q[15],q[40];
cz q[27],q[26];
cz q[50],q[59];
cz q[26],q[46];
cz q[7],q[17];
cz q[7],q[0];
cz q[7],q[57];
cz q[17],q[43];
cz q[17],q[33];
cz q[25],q[32];
cz q[25],q[40];
cz q[25],q[26];
cz q[32],q[40];
cz q[32],q[6];
cz q[30],q[46];
cz q[30],q[48];
cz q[30],q[0];
cz q[46],q[51];
cz q[46],q[3];
cz q[5],q[19];
cz q[5],q[12];
cz q[5],q[8];
cz q[19],q[45];
cz q[19],q[26];
cz q[38],q[53];
cz q[38],q[55];
cz q[38],q[35];
cz q[53],q[20];
cz q[53],q[29];
cz q[0],q[14];
cz q[14],q[44];
cz q[14],q[54];
cz q[15],q[23];
cz q[15],q[45];
cz q[15],q[49];
cz q[23],q[21];
cz q[23],q[59];
cz q[55],q[59];
cz q[55],q[39];
cz q[59],q[36];
cz q[48],q[31];
cz q[48],q[52];
cz q[42],q[50];
cz q[42],q[9];
cz q[42],q[44];
cz q[50],q[47];
cz q[50],q[24];
cz q[10],q[45];
cz q[10],q[31];
cz q[10],q[28];
cz q[12],q[18];
cz q[12],q[8];
cz q[18],q[51];
cz q[18],q[20];
cz q[3],q[24];
cz q[3],q[36];
cz q[24],q[58];
cz q[35],q[37];
cz q[35],q[29];
cz q[37],q[44];
cz q[37],q[27];
cz q[58],q[33];
cz q[58],q[43];
cz q[51],q[9];
cz q[31],q[34];
cz q[47],q[56];
cz q[47],q[28];
cz q[36],q[28];
cz q[9],q[22];
cz q[40],q[56];
cz q[56],q[34];
cz q[4],q[11];
cz q[4],q[8];
cz q[4],q[49];
cz q[11],q[1];
cz q[11],q[21];
cz q[49],q[52];
cz q[52],q[41];
cz q[2],q[13];
cz q[2],q[6];
cz q[2],q[33];
cz q[13],q[57];
cz q[13],q[34];
cz q[21],q[29];
cz q[6],q[16];
cz q[57],q[1];
cz q[39],q[27];
cz q[39],q[16];
cz q[26],q[54];
cz q[43],q[22];
cz q[54],q[20];
cz q[22],q[41];
cz q[41],q[16];
cz q[27],q[1];
cz q[32],q[37];
cz q[32],q[1];
cz q[32],q[59];
cz q[37],q[51];
cz q[37],q[44];
cz q[8],q[46];
cz q[8],q[35];
cz q[8],q[53];
cz q[46],q[21];
cz q[46],q[16];
cz q[39],q[42];
cz q[39],q[10];
cz q[39],q[29];
cz q[42],q[31];
cz q[42],q[9];
cz q[2],q[11];
cz q[2],q[52];
cz q[2],q[21];
cz q[11],q[59];
cz q[11],q[1];
cz q[17],q[30];
cz q[17],q[3];
cz q[17],q[54];
cz q[30],q[41];
cz q[30],q[31];
cz q[21],q[13];
cz q[23],q[52];
cz q[23],q[29];
cz q[23],q[18];
cz q[52],q[28];
cz q[4],q[48];
cz q[4],q[45];
cz q[4],q[54];
cz q[48],q[7];
cz q[48],q[3];
cz q[24],q[26];
cz q[24],q[58];
cz q[24],q[13];
cz q[26],q[40];
cz q[26],q[51];
cz q[27],q[43];
cz q[27],q[56];
cz q[27],q[20];
cz q[43],q[40];
cz q[43],q[20];
cz q[36],q[55];
cz q[36],q[0];
cz q[36],q[18];
cz q[55],q[28];
cz q[55],q[5];
cz q[25],q[34];
cz q[25],q[14];
cz q[25],q[51];
cz q[34],q[0];
cz q[34],q[56];
cz q[16],q[31];
cz q[16],q[58];
cz q[33],q[47];
cz q[33],q[10];
cz q[33],q[12];
cz q[47],q[7];
cz q[47],q[14];
cz q[0],q[44];
cz q[44],q[41];
cz q[40],q[28];
cz q[6],q[50];
cz q[6],q[38];
cz q[6],q[12];
cz q[50],q[19];
cz q[50],q[49];
cz q[13],q[19];
cz q[19],q[12];
cz q[41],q[22];
cz q[3],q[35];
cz q[29],q[57];
cz q[58],q[49];
cz q[35],q[15];
cz q[53],q[57];
cz q[53],q[45];
cz q[57],q[15];
cz q[45],q[18];
cz q[56],q[49];
cz q[9],q[38];
cz q[9],q[22];
cz q[15],q[54];
cz q[38],q[59];
cz q[10],q[14];
cz q[5],q[1];
cz q[5],q[22];
cz q[7],q[20];
cz q[41],q[49];
cz q[41],q[7];
cz q[41],q[52];
cz q[49],q[3];
cz q[49],q[12];
cz q[13],q[42];
cz q[13],q[26];
cz q[13],q[11];
cz q[42],q[15];
cz q[42],q[4];
cz q[28],q[58];
cz q[28],q[6];
cz q[28],q[8];
cz q[58],q[8];
cz q[58],q[35];
cz q[30],q[55];
cz q[30],q[17];
cz q[30],q[32];
cz q[55],q[26];
cz q[55],q[24];
cz q[35],q[51];
cz q[35],q[40];
cz q[51],q[14];
cz q[51],q[59];
cz q[20],q[29];
cz q[20],q[21];
cz q[20],q[59];
cz q[29],q[15];
cz q[29],q[48];
cz q[18],q[44];
cz q[18],q[50];
cz q[18],q[10];
cz q[44],q[1];
cz q[44],q[53];
cz q[2],q[11];
cz q[2],q[26];
cz q[2],q[5];
cz q[11],q[21];
cz q[17],q[59];
cz q[17],q[54];
cz q[3],q[10];
cz q[3],q[47];
cz q[32],q[39];
cz q[32],q[16];
cz q[39],q[34];
cz q[39],q[45];
cz q[16],q[31];
cz q[16],q[6];
cz q[31],q[23];
cz q[31],q[57];
cz q[7],q[37];
cz q[7],q[27];
cz q[37],q[0];
cz q[37],q[23];
cz q[22],q[46];
cz q[22],q[12];
cz q[22],q[27];
cz q[46],q[45];
cz q[46],q[25];
cz q[14],q[52];
cz q[14],q[47];
cz q[0],q[54];
cz q[0],q[40];
cz q[1],q[23];
cz q[1],q[38];
cz q[40],q[54];
cz q[12],q[47];
cz q[19],q[34];
cz q[19],q[21];
cz q[19],q[53];
cz q[34],q[45];
cz q[10],q[8];
cz q[15],q[48];
cz q[48],q[24];
cz q[4],q[27];
cz q[4],q[38];
cz q[36],q[43];
cz q[36],q[56];
cz q[36],q[5];
cz q[43],q[9];
cz q[43],q[33];
cz q[38],q[56];
cz q[24],q[57];
cz q[56],q[25];
cz q[57],q[50];
cz q[25],q[6];
cz q[50],q[9];
cz q[9],q[52];
cz q[33],q[53];
cz q[33],q[5];
cz q[4],q[55];
cz q[4],q[15];
cz q[4],q[33];
cz q[55],q[17];
cz q[55],q[22];
cz q[33],q[45];
cz q[33],q[17];
cz q[45],q[22];
cz q[45],q[41];
cz q[1],q[58];
cz q[1],q[34];
cz q[1],q[54];
cz q[58],q[27];
cz q[58],q[2];
cz q[5],q[28];
cz q[5],q[3];
cz q[5],q[31];
cz q[28],q[59];
cz q[28],q[54];
cz q[13],q[26];
cz q[13],q[50];
cz q[13],q[25];
cz q[26],q[56];
cz q[26],q[42];
cz q[11],q[44];
cz q[11],q[29];
cz q[11],q[12];
cz q[44],q[16];
cz q[44],q[50];
cz q[14],q[24];
cz q[14],q[16];
cz q[14],q[38];
cz q[24],q[6];
cz q[24],q[27];
cz q[21],q[57];
cz q[21],q[20];
cz q[21],q[0];
cz q[57],q[36];
cz q[57],q[49];
cz q[30],q[41];
cz q[30],q[59];
cz q[30],q[32];
cz q[41],q[10];
cz q[36],q[47];
cz q[36],q[34];
cz q[0],q[37];
cz q[0],q[38];
cz q[37],q[9];
cz q[37],q[52];
cz q[8],q[59];
cz q[8],q[52];
cz q[8],q[7];
cz q[39],q[46];
cz q[39],q[31];
cz q[39],q[7];
cz q[46],q[3];
cz q[46],q[20];
cz q[49],q[12];
cz q[49],q[17];
cz q[12],q[47];
cz q[47],q[56];
cz q[19],q[43];
cz q[19],q[56];
cz q[19],q[10];
cz q[43],q[35];
cz q[43],q[16];
cz q[52],q[54];
cz q[22],q[23];
cz q[23],q[42];
cz q[23],q[51];
cz q[48],q[51];
cz q[48],q[15];
cz q[48],q[20];
cz q[51],q[2];
cz q[15],q[6];
cz q[18],q[53];
cz q[18],q[25];
cz q[18],q[9];
cz q[53],q[29];
cz q[53],q[40];
cz q[27],q[31];
cz q[25],q[40];
cz q[40],q[35];
cz q[42],q[6];
cz q[9],q[34];
cz q[3],q[29];
cz q[7],q[50];
cz q[10],q[32];
cz q[38],q[32];
cz q[2],q[35];
cz q[41],q[58];
cz q[41],q[6];
cz q[41],q[50];
cz q[58],q[34];
cz q[58],q[51];
cz q[2],q[39];
cz q[2],q[36];
cz q[2],q[23];
cz q[39],q[52];
cz q[39],q[36];
cz q[33],q[54];
cz q[33],q[48];
cz q[33],q[5];
cz q[54],q[59];
cz q[54],q[16];
cz q[3],q[13];
cz q[3],q[42];
cz q[3],q[28];
cz q[13],q[1];
cz q[13],q[49];
cz q[27],q[34];
cz q[27],q[29];
cz q[27],q[7];
cz q[34],q[24];
cz q[24],q[35];
cz q[24],q[10];
cz q[35],q[25];
cz q[35],q[11];
cz q[8],q[48];
cz q[8],q[52];
cz q[8],q[4];
cz q[48],q[32];
cz q[0],q[53];
cz q[0],q[57];
cz q[0],q[56];
cz q[53],q[20];
cz q[53],q[25];
cz q[6],q[28];
cz q[6],q[12];
cz q[29],q[52];
cz q[29],q[15];
cz q[42],q[9];
cz q[42],q[19];
cz q[10],q[47];
cz q[10],q[12];
cz q[47],q[23];
cz q[47],q[56];
cz q[51],q[57];
cz q[51],q[21];
cz q[57],q[37];
cz q[11],q[55];
cz q[11],q[36];
cz q[55],q[44];
cz q[55],q[23];
cz q[22],q[30];
cz q[22],q[56];
cz q[22],q[31];
cz q[30],q[45];
cz q[30],q[14];
cz q[15],q[46];
cz q[15],q[18];
cz q[46],q[18];
cz q[46],q[40];
cz q[4],q[43];
cz q[4],q[21];
cz q[43],q[19];
cz q[43],q[16];
cz q[18],q[5];
cz q[16],q[17];
cz q[17],q[40];
cz q[17],q[37];
cz q[19],q[44];
cz q[9],q[32];
cz q[9],q[31];
cz q[38],q[50];
cz q[38],q[49];
cz q[38],q[14];
cz q[50],q[20];
cz q[28],q[31];
cz q[49],q[59];
cz q[59],q[45];
cz q[45],q[7];
cz q[32],q[7];
cz q[5],q[26];
cz q[20],q[37];
cz q[14],q[1];
cz q[1],q[25];
cz q[12],q[26];
cz q[26],q[40];
cz q[44],q[21];
cz q[2],q[39];
cz q[2],q[11];
cz q[2],q[15];
cz q[39],q[19];
cz q[39],q[38];
cz q[8],q[55];
cz q[8],q[27];
cz q[8],q[35];
cz q[55],q[47];
cz q[55],q[17];
cz q[3],q[13];
cz q[3],q[42];
cz q[3],q[23];
cz q[13],q[12];
cz q[13],q[49];
cz q[11],q[35];
cz q[11],q[28];
cz q[4],q[30];
cz q[4],q[59];
cz q[4],q[33];
cz q[30],q[5];
cz q[30],q[42];
cz q[0],q[23];
cz q[0],q[45];
cz q[0],q[31];
cz q[23],q[37];
cz q[22],q[53];
cz q[22],q[16];
cz q[22],q[31];
cz q[53],q[44];
cz q[53],q[19];
cz q[1],q[24];
cz q[1],q[21];
cz q[1],q[54];
cz q[24],q[50];
cz q[24],q[17];
cz q[16],q[42];
cz q[16],q[58];
cz q[47],q[10];
cz q[47],q[52];
cz q[19],q[37];
cz q[5],q[46];
cz q[5],q[29];
cz q[59],q[45];
cz q[59],q[37];
cz q[10],q[31];
cz q[10],q[33];
cz q[33],q[49];
cz q[49],q[52];
cz q[38],q[36];
cz q[38],q[26];
cz q[12],q[20];
cz q[12],q[26];
cz q[20],q[43];
cz q[20],q[45];
cz q[21],q[41];
cz q[21],q[29];
cz q[41],q[18];
cz q[41],q[27];
cz q[18],q[51];
cz q[18],q[15];
cz q[51],q[56];
cz q[51],q[6];
cz q[15],q[36];
cz q[28],q[46];
cz q[28],q[34];
cz q[46],q[44];
cz q[17],q[27];
cz q[9],q[14];
cz q[9],q[57];
cz q[9],q[34];
cz q[14],q[25];
cz q[14],q[43];
cz q[52],q[56];
cz q[43],q[48];
cz q[48],q[7];
cz q[48],q[25];
cz q[6],q[29];
cz q[6],q[35];
cz q[58],q[26];
cz q[58],q[32];
cz q[7],q[50];
cz q[7],q[40];
cz q[36],q[40];
cz q[34],q[40];
cz q[57],q[50];
cz q[57],q[25];
cz q[56],q[54];
cz q[32],q[54];
cz q[32],q[44];
cz q[10],q[43];
cz q[10],q[27];
cz q[10],q[46];
cz q[43],q[28];
cz q[43],q[41];
cz q[25],q[41];
cz q[25],q[28];
cz q[25],q[46];
cz q[41],q[26];
cz q[18],q[44];
cz q[18],q[8];
cz q[18],q[36];
cz q[44],q[38];
cz q[44],q[30];
cz q[38],q[15];
cz q[38],q[58];
cz q[8],q[24];
cz q[8],q[35];
cz q[0],q[23];
cz q[0],q[52];
cz q[0],q[4];
cz q[23],q[49];
cz q[23],q[42];
cz q[37],q[42];
cz q[37],q[39];
cz q[37],q[6];
cz q[42],q[2];
cz q[27],q[40];
cz q[27],q[51];
cz q[11],q[16];
cz q[11],q[22];
cz q[11],q[26];
cz q[16],q[53];
cz q[16],q[6];
cz q[31],q[49];
cz q[31],q[2];
cz q[31],q[7];
cz q[49],q[9];
cz q[9],q[14];
cz q[9],q[55];
cz q[22],q[30];
cz q[22],q[21];
cz q[30],q[34];
cz q[19],q[34];
cz q[19],q[56];
cz q[19],q[15];
cz q[34],q[13];
cz q[36],q[50];
cz q[36],q[32];
cz q[50],q[45];
cz q[50],q[1];
cz q[47],q[59];
cz q[47],q[21];
cz q[47],q[17];
cz q[59],q[5];
cz q[59],q[20];
cz q[17],q[55];
cz q[17],q[12];
cz q[55],q[46];
cz q[14],q[3];
cz q[14],q[54];
cz q[40],q[6];
cz q[40],q[7];
cz q[33],q[35];
cz q[33],q[39];
cz q[33],q[26];
cz q[35],q[4];
cz q[20],q[28];
cz q[20],q[3];
cz q[2],q[13];
cz q[13],q[1];
cz q[45],q[48];
cz q[45],q[57];
cz q[48],q[5];
cz q[48],q[12];
cz q[57],q[3];
cz q[57],q[29];
cz q[51],q[29];
cz q[51],q[53];
cz q[5],q[58];
cz q[56],q[15];
cz q[56],q[1];
cz q[52],q[32];
cz q[52],q[54];
cz q[12],q[4];
cz q[7],q[39];
cz q[29],q[53];
cz q[21],q[58];
cz q[32],q[24];
cz q[24],q[54];
cz q[55],q[57];
cz q[55],q[7];
cz q[55],q[9];
cz q[57],q[12];
cz q[57],q[25];
cz q[6],q[18];
cz q[6],q[38];
cz q[6],q[2];
cz q[18],q[50];
cz q[18],q[10];
cz q[16],q[20];
cz q[16],q[58];
cz q[16],q[10];
cz q[20],q[23];
cz q[20],q[1];
cz q[7],q[17];
cz q[7],q[52];
cz q[17],q[9];
cz q[17],q[43];
cz q[43],q[46];
cz q[43],q[49];
cz q[46],q[19];
cz q[46],q[59];
cz q[19],q[3];
cz q[19],q[24];
cz q[0],q[51];
cz q[0],q[27];
cz q[0],q[31];
cz q[51],q[10];
cz q[51],q[52];
cz q[9],q[53];
cz q[29],q[59];
cz q[29],q[49];
cz q[29],q[48];
cz q[59],q[15];
cz q[37],q[42];
cz q[37],q[54];
cz q[37],q[4];
cz q[42],q[14];
cz q[42],q[8];
cz q[31],q[56];
cz q[31],q[58];
cz q[56],q[21];
cz q[56],q[22];
cz q[30],q[48];
cz q[30],q[36];
cz q[30],q[40];
cz q[48],q[5];
cz q[44],q[47];
cz q[44],q[27];
cz q[44],q[28];
cz q[47],q[26];
cz q[47],q[22];
cz q[2],q[41];
cz q[2],q[21];
cz q[41],q[11];
cz q[41],q[33];
cz q[14],q[24];
cz q[14],q[50];
cz q[24],q[45];
cz q[1],q[8];
cz q[1],q[13];
cz q[8],q[28];
cz q[58],q[53];
cz q[25],q[36];
cz q[25],q[38];
cz q[36],q[54];
cz q[40],q[45];
cz q[40],q[38];
cz q[45],q[28];
cz q[22],q[39];
cz q[39],q[34];
cz q[39],q[23];
cz q[5],q[32];
cz q[5],q[54];
cz q[32],q[34];
cz q[32],q[13];
cz q[3],q[12];
cz q[3],q[52];
cz q[33],q[35];
cz q[33],q[21];
cz q[35],q[50];
cz q[35],q[11];
cz q[11],q[13];
cz q[53],q[23];
cz q[12],q[4];
cz q[49],q[34];
cz q[15],q[27];
cz q[15],q[26];
cz q[26],q[4];
