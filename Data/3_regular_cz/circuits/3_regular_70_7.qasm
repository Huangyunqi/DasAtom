OPENQASM 2.0;
include "qelib1.inc";
qreg q[70];
cz q[47],q[62];
cz q[47],q[33];
cz q[47],q[58];
cz q[62],q[29];
cz q[62],q[4];
cz q[7],q[35];
cz q[7],q[21];
cz q[7],q[13];
cz q[35],q[14];
cz q[35],q[21];
cz q[18],q[44];
cz q[18],q[69];
cz q[18],q[25];
cz q[44],q[39];
cz q[44],q[68];
cz q[25],q[59];
cz q[25],q[54];
cz q[59],q[57];
cz q[59],q[63];
cz q[0],q[69];
cz q[0],q[61];
cz q[0],q[28];
cz q[69],q[66];
cz q[9],q[17];
cz q[9],q[38];
cz q[9],q[22];
cz q[17],q[3];
cz q[17],q[42];
cz q[66],q[11];
cz q[66],q[16];
cz q[23],q[43];
cz q[23],q[36];
cz q[23],q[29];
cz q[43],q[60];
cz q[43],q[26];
cz q[24],q[26];
cz q[24],q[12];
cz q[24],q[56];
cz q[26],q[38];
cz q[28],q[51];
cz q[28],q[20];
cz q[51],q[53];
cz q[51],q[55];
cz q[33],q[40];
cz q[33],q[5];
cz q[10],q[63];
cz q[10],q[29];
cz q[10],q[3];
cz q[63],q[27];
cz q[39],q[64];
cz q[39],q[67];
cz q[60],q[56];
cz q[60],q[61];
cz q[57],q[1];
cz q[57],q[48];
cz q[2],q[4];
cz q[2],q[65];
cz q[2],q[58];
cz q[4],q[21];
cz q[36],q[48];
cz q[36],q[49];
cz q[31],q[49];
cz q[31],q[32];
cz q[31],q[50];
cz q[49],q[42];
cz q[15],q[16];
cz q[15],q[68];
cz q[15],q[65];
cz q[16],q[14];
cz q[48],q[37];
cz q[27],q[45];
cz q[27],q[46];
cz q[45],q[64];
cz q[45],q[22];
cz q[56],q[22];
cz q[40],q[8];
cz q[40],q[5];
cz q[54],q[34];
cz q[54],q[41];
cz q[3],q[52];
cz q[6],q[52];
cz q[6],q[12];
cz q[6],q[32];
cz q[52],q[1];
cz q[14],q[53];
cz q[13],q[67];
cz q[13],q[64];
cz q[67],q[20];
cz q[19],q[34];
cz q[19],q[61];
cz q[19],q[53];
cz q[34],q[30];
cz q[30],q[42];
cz q[30],q[11];
cz q[11],q[65];
cz q[38],q[50];
cz q[50],q[41];
cz q[20],q[8];
cz q[12],q[46];
cz q[8],q[37];
cz q[68],q[37];
cz q[55],q[58];
cz q[55],q[46];
cz q[1],q[32];
cz q[5],q[41];
cz q[27],q[50];
cz q[27],q[44];
cz q[27],q[18];
cz q[50],q[30];
cz q[50],q[21];
cz q[55],q[66];
cz q[55],q[36];
cz q[55],q[65];
cz q[66],q[42];
cz q[66],q[56];
cz q[25],q[32];
cz q[25],q[28];
cz q[25],q[6];
cz q[32],q[46];
cz q[32],q[62];
cz q[30],q[46];
cz q[30],q[29];
cz q[46],q[22];
cz q[24],q[51];
cz q[24],q[37];
cz q[24],q[62];
cz q[51],q[68];
cz q[51],q[52];
cz q[42],q[2];
cz q[42],q[38];
cz q[23],q[34];
cz q[23],q[7];
cz q[23],q[64];
cz q[34],q[9];
cz q[34],q[13];
cz q[9],q[35];
cz q[9],q[12];
cz q[35],q[44];
cz q[35],q[57];
cz q[36],q[54];
cz q[36],q[3];
cz q[44],q[3];
cz q[28],q[60];
cz q[28],q[4];
cz q[60],q[69];
cz q[60],q[1];
cz q[2],q[41];
cz q[2],q[38];
cz q[41],q[67];
cz q[41],q[64];
cz q[10],q[54];
cz q[10],q[49];
cz q[10],q[21];
cz q[54],q[26];
cz q[26],q[53];
cz q[26],q[31];
cz q[53],q[17];
cz q[53],q[68];
cz q[40],q[52];
cz q[40],q[67];
cz q[40],q[17];
cz q[52],q[3];
cz q[22],q[8];
cz q[22],q[0];
cz q[18],q[58];
cz q[18],q[38];
cz q[58],q[14];
cz q[58],q[19];
cz q[15],q[16];
cz q[15],q[6];
cz q[15],q[29];
cz q[16],q[19];
cz q[16],q[45];
cz q[47],q[48];
cz q[47],q[12];
cz q[47],q[4];
cz q[48],q[20];
cz q[48],q[61];
cz q[37],q[1];
cz q[37],q[31];
cz q[17],q[39];
cz q[56],q[69];
cz q[56],q[39];
cz q[69],q[63];
cz q[8],q[59];
cz q[8],q[0];
cz q[59],q[43];
cz q[59],q[1];
cz q[12],q[67];
cz q[62],q[6];
cz q[21],q[63];
cz q[7],q[11];
cz q[7],q[45];
cz q[49],q[11];
cz q[49],q[5];
cz q[68],q[14];
cz q[39],q[13];
cz q[19],q[33];
cz q[65],q[61];
cz q[65],q[43];
cz q[14],q[29];
cz q[43],q[57];
cz q[57],q[33];
cz q[31],q[45];
cz q[4],q[63];
cz q[61],q[5];
cz q[13],q[64];
cz q[11],q[20];
cz q[0],q[20];
cz q[33],q[5];
cz q[17],q[21];
cz q[17],q[41];
cz q[17],q[12];
cz q[21],q[64];
cz q[21],q[47];
cz q[64],q[48];
cz q[64],q[10];
cz q[13],q[35];
cz q[13],q[62];
cz q[13],q[34];
cz q[35],q[6];
cz q[35],q[56];
cz q[15],q[32];
cz q[15],q[2];
cz q[15],q[7];
cz q[32],q[60];
cz q[32],q[42];
cz q[27],q[52];
cz q[27],q[19];
cz q[27],q[55];
cz q[52],q[46];
cz q[52],q[11];
cz q[16],q[31];
cz q[16],q[69];
cz q[16],q[44];
cz q[31],q[0];
cz q[31],q[11];
cz q[44],q[47];
cz q[44],q[14];
cz q[47],q[67];
cz q[20],q[22];
cz q[20],q[4];
cz q[20],q[61];
cz q[22],q[62];
cz q[22],q[29];
cz q[8],q[11];
cz q[8],q[43];
cz q[8],q[28];
cz q[9],q[65];
cz q[9],q[18];
cz q[9],q[45];
cz q[65],q[33];
cz q[65],q[53];
cz q[60],q[46];
cz q[60],q[36];
cz q[26],q[53];
cz q[26],q[40];
cz q[26],q[37];
cz q[53],q[46];
cz q[14],q[51];
cz q[14],q[63];
cz q[51],q[5];
cz q[51],q[69];
cz q[24],q[28];
cz q[24],q[7];
cz q[24],q[40];
cz q[28],q[39];
cz q[36],q[57];
cz q[36],q[49];
cz q[57],q[54];
cz q[57],q[40];
cz q[23],q[29];
cz q[23],q[37];
cz q[23],q[2];
cz q[29],q[56];
cz q[4],q[34];
cz q[4],q[42];
cz q[34],q[48];
cz q[38],q[66];
cz q[38],q[30];
cz q[38],q[67];
cz q[66],q[12];
cz q[66],q[25];
cz q[43],q[19];
cz q[43],q[55];
cz q[19],q[0];
cz q[55],q[25];
cz q[49],q[59];
cz q[49],q[58];
cz q[59],q[39];
cz q[59],q[54];
cz q[56],q[54];
cz q[39],q[33];
cz q[62],q[68];
cz q[30],q[45];
cz q[30],q[6];
cz q[45],q[3];
cz q[3],q[67];
cz q[3],q[18];
cz q[33],q[50];
cz q[12],q[42];
cz q[48],q[1];
cz q[41],q[5];
cz q[41],q[37];
cz q[69],q[10];
cz q[2],q[10];
cz q[5],q[63];
cz q[7],q[50];
cz q[50],q[18];
cz q[68],q[1];
cz q[68],q[61];
cz q[6],q[25];
cz q[1],q[61];
cz q[0],q[58];
cz q[63],q[58];
cz q[13],q[33];
cz q[13],q[51];
cz q[13],q[69];
cz q[33],q[60];
cz q[33],q[53];
cz q[8],q[46];
cz q[8],q[48];
cz q[8],q[24];
cz q[46],q[34];
cz q[46],q[47];
cz q[34],q[65];
cz q[34],q[2];
cz q[65],q[57];
cz q[65],q[4];
cz q[15],q[39];
cz q[15],q[3];
cz q[15],q[31];
cz q[39],q[21];
cz q[39],q[6];
cz q[12],q[25];
cz q[12],q[58];
cz q[12],q[4];
cz q[25],q[24];
cz q[25],q[0];
cz q[14],q[22];
cz q[14],q[7];
cz q[14],q[50];
cz q[22],q[44];
cz q[22],q[53];
cz q[44],q[11];
cz q[44],q[53];
cz q[55],q[68];
cz q[55],q[0];
cz q[55],q[37];
cz q[68],q[66];
cz q[68],q[63];
cz q[48],q[38];
cz q[48],q[50];
cz q[11],q[54];
cz q[11],q[56];
cz q[3],q[67];
cz q[3],q[64];
cz q[23],q[27];
cz q[23],q[63];
cz q[23],q[1];
cz q[27],q[58];
cz q[27],q[62];
cz q[21],q[19];
cz q[21],q[69];
cz q[9],q[28];
cz q[9],q[50];
cz q[9],q[29];
cz q[28],q[57];
cz q[28],q[42];
cz q[49],q[64];
cz q[49],q[52];
cz q[49],q[60];
cz q[64],q[43];
cz q[63],q[16];
cz q[2],q[40];
cz q[2],q[60];
cz q[0],q[17];
cz q[38],q[4];
cz q[38],q[45];
cz q[16],q[26];
cz q[16],q[52];
cz q[7],q[1];
cz q[7],q[59];
cz q[26],q[36];
cz q[26],q[35];
cz q[42],q[54];
cz q[42],q[40];
cz q[54],q[41];
cz q[51],q[37];
cz q[51],q[67];
cz q[43],q[59];
cz q[43],q[24];
cz q[29],q[56];
cz q[29],q[18];
cz q[56],q[30];
cz q[69],q[31];
cz q[37],q[61];
cz q[47],q[61];
cz q[47],q[6];
cz q[61],q[36];
cz q[58],q[32];
cz q[36],q[32];
cz q[57],q[10];
cz q[67],q[10];
cz q[52],q[59];
cz q[66],q[5];
cz q[66],q[20];
cz q[6],q[17];
cz q[1],q[5];
cz q[5],q[40];
cz q[30],q[20];
cz q[30],q[31];
cz q[20],q[17];
cz q[41],q[62];
cz q[41],q[45];
cz q[62],q[19];
cz q[45],q[18];
cz q[10],q[18];
cz q[32],q[35];
cz q[19],q[35];
cz q[27],q[50];
cz q[27],q[56];
cz q[27],q[40];
cz q[50],q[5];
cz q[50],q[24];
cz q[32],q[46];
cz q[32],q[41];
cz q[32],q[40];
cz q[46],q[19];
cz q[46],q[45];
cz q[19],q[54];
cz q[19],q[58];
cz q[16],q[47];
cz q[16],q[15];
cz q[16],q[24];
cz q[47],q[12];
cz q[47],q[0];
cz q[38],q[44];
cz q[38],q[23];
cz q[38],q[60];
cz q[44],q[1];
cz q[44],q[8];
cz q[11],q[69];
cz q[11],q[23];
cz q[11],q[8];
cz q[69],q[66];
cz q[69],q[49];
cz q[66],q[57];
cz q[66],q[34];
cz q[57],q[34];
cz q[57],q[29];
cz q[29],q[59];
cz q[29],q[5];
cz q[59],q[21];
cz q[59],q[49];
cz q[23],q[18];
cz q[36],q[64];
cz q[36],q[15];
cz q[36],q[35];
cz q[64],q[3];
cz q[64],q[42];
cz q[37],q[63];
cz q[37],q[55];
cz q[37],q[12];
cz q[63],q[4];
cz q[63],q[68];
cz q[8],q[53];
cz q[25],q[61];
cz q[25],q[33];
cz q[25],q[6];
cz q[61],q[5];
cz q[61],q[58];
cz q[14],q[33];
cz q[14],q[42];
cz q[14],q[52];
cz q[33],q[15];
cz q[42],q[53];
cz q[1],q[17];
cz q[1],q[6];
cz q[17],q[20];
cz q[17],q[49];
cz q[24],q[13];
cz q[9],q[12];
cz q[9],q[60];
cz q[9],q[48];
cz q[18],q[51];
cz q[18],q[31];
cz q[51],q[30];
cz q[51],q[67];
cz q[21],q[26];
cz q[21],q[53];
cz q[56],q[58];
cz q[56],q[39];
cz q[3],q[65];
cz q[3],q[60];
cz q[65],q[26];
cz q[65],q[22];
cz q[31],q[35];
cz q[31],q[0];
cz q[35],q[2];
cz q[41],q[67];
cz q[41],q[7];
cz q[67],q[10];
cz q[45],q[68];
cz q[45],q[52];
cz q[40],q[48];
cz q[22],q[62];
cz q[22],q[34];
cz q[62],q[48];
cz q[62],q[2];
cz q[54],q[55];
cz q[54],q[43];
cz q[20],q[28];
cz q[20],q[0];
cz q[28],q[43];
cz q[28],q[52];
cz q[7],q[55];
cz q[7],q[13];
cz q[4],q[13];
cz q[4],q[10];
cz q[26],q[43];
cz q[68],q[39];
cz q[10],q[30];
cz q[30],q[2];
cz q[39],q[6];
cz q[12],q[68];
cz q[12],q[34];
cz q[12],q[2];
cz q[68],q[56];
cz q[68],q[62];
cz q[41],q[49];
cz q[41],q[37];
cz q[41],q[45];
cz q[49],q[55];
cz q[49],q[44];
cz q[25],q[50];
cz q[25],q[54];
cz q[25],q[33];
cz q[50],q[0];
cz q[50],q[35];
cz q[33],q[63];
cz q[33],q[59];
cz q[63],q[36];
cz q[63],q[14];
cz q[11],q[60];
cz q[11],q[46];
cz q[11],q[52];
cz q[60],q[34];
cz q[60],q[64];
cz q[20],q[38];
cz q[20],q[15];
cz q[20],q[7];
cz q[38],q[10];
cz q[38],q[65];
cz q[34],q[69];
cz q[4],q[30];
cz q[4],q[66];
cz q[4],q[26];
cz q[30],q[1];
cz q[30],q[54];
cz q[9],q[35];
cz q[9],q[39];
cz q[9],q[66];
cz q[35],q[64];
cz q[1],q[15];
cz q[1],q[42];
cz q[15],q[45];
cz q[18],q[65];
cz q[18],q[28];
cz q[18],q[54];
cz q[65],q[69];
cz q[66],q[32];
cz q[28],q[51];
cz q[28],q[45];
cz q[51],q[40];
cz q[51],q[32];
cz q[42],q[0];
cz q[42],q[46];
cz q[55],q[39];
cz q[55],q[14];
cz q[29],q[52];
cz q[29],q[61];
cz q[29],q[31];
cz q[52],q[22];
cz q[61],q[8];
cz q[61],q[67];
cz q[47],q[57];
cz q[47],q[32];
cz q[47],q[5];
cz q[57],q[22];
cz q[57],q[23];
cz q[10],q[21];
cz q[10],q[48];
cz q[24],q[37];
cz q[24],q[48];
cz q[24],q[8];
cz q[37],q[43];
cz q[17],q[53];
cz q[17],q[27];
cz q[17],q[0];
cz q[53],q[48];
cz q[53],q[40];
cz q[69],q[14];
cz q[44],q[58];
cz q[44],q[6];
cz q[46],q[5];
cz q[43],q[62];
cz q[43],q[58];
cz q[62],q[58];
cz q[16],q[26];
cz q[16],q[59];
cz q[16],q[6];
cz q[26],q[67];
cz q[8],q[13];
cz q[27],q[40];
cz q[27],q[7];
cz q[13],q[23];
cz q[13],q[2];
cz q[23],q[3];
cz q[5],q[36];
cz q[31],q[39];
cz q[31],q[21];
cz q[64],q[7];
cz q[3],q[21];
cz q[3],q[2];
cz q[36],q[56];
cz q[19],q[22];
cz q[19],q[6];
cz q[19],q[67];
cz q[56],q[59];
cz q[34],q[65];
cz q[34],q[39];
cz q[34],q[5];
cz q[65],q[9];
cz q[65],q[6];
cz q[10],q[43];
cz q[10],q[69];
cz q[10],q[13];
cz q[43],q[27];
cz q[43],q[6];
cz q[30],q[55];
cz q[30],q[63];
cz q[30],q[11];
cz q[55],q[0];
cz q[55],q[42];
cz q[1],q[58];
cz q[1],q[28];
cz q[1],q[11];
cz q[58],q[69];
cz q[58],q[35];
cz q[3],q[31];
cz q[3],q[64];
cz q[3],q[39];
cz q[31],q[33];
cz q[31],q[63];
cz q[23],q[52];
cz q[23],q[46];
cz q[23],q[21];
cz q[52],q[60];
cz q[52],q[44];
cz q[27],q[8];
cz q[27],q[57];
cz q[32],q[39];
cz q[32],q[54];
cz q[32],q[40];
cz q[44],q[56];
cz q[44],q[40];
cz q[56],q[4];
cz q[56],q[0];
cz q[12],q[18];
cz q[12],q[9];
cz q[12],q[59];
cz q[18],q[14];
cz q[18],q[16];
cz q[9],q[62];
cz q[22],q[28];
cz q[22],q[60];
cz q[22],q[50];
cz q[28],q[66];
cz q[2],q[68];
cz q[2],q[63];
cz q[2],q[62];
cz q[68],q[15];
cz q[68],q[20];
cz q[24],q[37];
cz q[24],q[6];
cz q[24],q[50];
cz q[37],q[36];
cz q[37],q[38];
cz q[0],q[48];
cz q[33],q[57];
cz q[33],q[41];
cz q[51],q[66];
cz q[51],q[29];
cz q[51],q[42];
cz q[66],q[15];
cz q[14],q[26];
cz q[14],q[19];
cz q[26],q[54];
cz q[26],q[47];
cz q[21],q[41];
cz q[21],q[4];
cz q[41],q[17];
cz q[60],q[8];
cz q[48],q[15];
cz q[48],q[64];
cz q[17],q[64];
cz q[17],q[5];
cz q[19],q[54];
cz q[19],q[67];
cz q[40],q[47];
cz q[47],q[67];
cz q[38],q[50];
cz q[38],q[49];
cz q[8],q[29];
cz q[62],q[61];
cz q[29],q[53];
cz q[53],q[61];
cz q[53],q[25];
cz q[61],q[13];
cz q[11],q[25];
cz q[25],q[35];
cz q[69],q[46];
cz q[35],q[57];
cz q[5],q[45];
cz q[36],q[49];
cz q[36],q[7];
cz q[49],q[45];
cz q[16],q[46];
cz q[16],q[4];
cz q[13],q[59];
cz q[59],q[20];
cz q[45],q[7];
cz q[7],q[20];
cz q[42],q[67];
