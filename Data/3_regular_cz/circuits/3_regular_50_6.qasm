OPENQASM 2.0;
include "qelib1.inc";
qreg q[50];
cz q[32],q[37];
cz q[32],q[28];
cz q[32],q[12];
cz q[37],q[34];
cz q[37],q[29];
cz q[24],q[27];
cz q[24],q[10];
cz q[24],q[47];
cz q[27],q[6];
cz q[27],q[10];
cz q[3],q[4];
cz q[3],q[19];
cz q[3],q[38];
cz q[4],q[20];
cz q[4],q[47];
cz q[26],q[33];
cz q[26],q[46];
cz q[26],q[43];
cz q[33],q[20];
cz q[33],q[0];
cz q[20],q[23];
cz q[23],q[41];
cz q[23],q[44];
cz q[21],q[22];
cz q[21],q[34];
cz q[21],q[19];
cz q[22],q[18];
cz q[22],q[2];
cz q[1],q[43];
cz q[1],q[2];
cz q[1],q[41];
cz q[43],q[18];
cz q[12],q[13];
cz q[12],q[0];
cz q[13],q[48];
cz q[13],q[49];
cz q[6],q[35];
cz q[6],q[16];
cz q[48],q[30];
cz q[48],q[29];
cz q[19],q[35];
cz q[16],q[47];
cz q[16],q[2];
cz q[34],q[46];
cz q[8],q[18];
cz q[8],q[44];
cz q[8],q[28];
cz q[7],q[44];
cz q[7],q[15];
cz q[7],q[36];
cz q[9],q[11];
cz q[9],q[17];
cz q[9],q[15];
cz q[11],q[0];
cz q[11],q[41];
cz q[17],q[36];
cz q[17],q[25];
cz q[36],q[40];
cz q[40],q[45];
cz q[40],q[30];
cz q[10],q[42];
cz q[46],q[14];
cz q[14],q[45];
cz q[14],q[35];
cz q[30],q[42];
cz q[42],q[31];
cz q[25],q[28];
cz q[25],q[31];
cz q[5],q[49];
cz q[5],q[15];
cz q[5],q[38];
cz q[49],q[39];
cz q[31],q[39];
cz q[29],q[45];
cz q[38],q[39];
cz q[6],q[18];
cz q[6],q[4];
cz q[6],q[8];
cz q[18],q[26];
cz q[18],q[36];
cz q[7],q[23];
cz q[7],q[47];
cz q[7],q[40];
cz q[23],q[0];
cz q[23],q[21];
cz q[13],q[33];
cz q[13],q[45];
cz q[13],q[14];
cz q[33],q[48];
cz q[33],q[11];
cz q[26],q[41];
cz q[26],q[3];
cz q[35],q[36];
cz q[35],q[1];
cz q[35],q[43];
cz q[36],q[43];
cz q[4],q[27];
cz q[4],q[17];
cz q[48],q[47];
cz q[48],q[20];
cz q[45],q[28];
cz q[45],q[25];
cz q[21],q[31];
cz q[21],q[1];
cz q[31],q[37];
cz q[31],q[40];
cz q[12],q[28];
cz q[12],q[15];
cz q[12],q[24];
cz q[28],q[25];
cz q[5],q[16];
cz q[5],q[15];
cz q[5],q[32];
cz q[16],q[46];
cz q[16],q[10];
cz q[47],q[42];
cz q[14],q[22];
cz q[14],q[29];
cz q[22],q[38];
cz q[22],q[11];
cz q[27],q[34];
cz q[27],q[30];
cz q[38],q[11];
cz q[38],q[30];
cz q[2],q[8];
cz q[2],q[19];
cz q[2],q[46];
cz q[8],q[41];
cz q[0],q[32];
cz q[0],q[37];
cz q[34],q[29];
cz q[34],q[39];
cz q[1],q[44];
cz q[43],q[24];
cz q[25],q[39];
cz q[15],q[17];
cz q[17],q[10];
cz q[32],q[3];
cz q[42],q[10];
cz q[42],q[29];
cz q[3],q[9];
cz q[9],q[19];
cz q[9],q[49];
cz q[24],q[20];
cz q[37],q[49];
cz q[46],q[40];
cz q[39],q[44];
cz q[44],q[49];
cz q[41],q[30];
cz q[19],q[20];
cz q[35],q[36];
cz q[35],q[21];
cz q[35],q[5];
cz q[36],q[43];
cz q[36],q[14];
cz q[6],q[24];
cz q[6],q[31];
cz q[6],q[34];
cz q[24],q[14];
cz q[24],q[0];
cz q[25],q[47];
cz q[25],q[18];
cz q[25],q[1];
cz q[47],q[18];
cz q[47],q[13];
cz q[3],q[10];
cz q[3],q[16];
cz q[3],q[19];
cz q[10],q[11];
cz q[10],q[28];
cz q[20],q[26];
cz q[20],q[43];
cz q[20],q[8];
cz q[26],q[11];
cz q[26],q[1];
cz q[16],q[37];
cz q[16],q[49];
cz q[2],q[48];
cz q[2],q[17];
cz q[2],q[32];
cz q[48],q[12];
cz q[48],q[40];
cz q[38],q[41];
cz q[38],q[11];
cz q[38],q[15];
cz q[41],q[31];
cz q[41],q[14];
cz q[19],q[23];
cz q[19],q[0];
cz q[23],q[28];
cz q[23],q[13];
cz q[28],q[32];
cz q[12],q[13];
cz q[12],q[9];
cz q[18],q[9];
cz q[7],q[44];
cz q[7],q[4];
cz q[7],q[33];
cz q[44],q[15];
cz q[44],q[46];
cz q[4],q[33];
cz q[4],q[42];
cz q[33],q[49];
cz q[31],q[46];
cz q[27],q[37];
cz q[27],q[30];
cz q[27],q[32];
cz q[37],q[0];
cz q[43],q[9];
cz q[42],q[5];
cz q[42],q[40];
cz q[17],q[39];
cz q[17],q[22];
cz q[39],q[30];
cz q[39],q[34];
cz q[30],q[22];
cz q[8],q[45];
cz q[8],q[34];
cz q[45],q[1];
cz q[45],q[21];
cz q[5],q[46];
cz q[15],q[29];
cz q[29],q[49];
cz q[29],q[21];
cz q[22],q[40];
cz q[1],q[28];
cz q[1],q[38];
cz q[1],q[7];
cz q[28],q[39];
cz q[28],q[48];
cz q[13],q[30];
cz q[13],q[43];
cz q[13],q[34];
cz q[30],q[27];
cz q[30],q[37];
cz q[25],q[35];
cz q[25],q[32];
cz q[25],q[8];
cz q[35],q[6];
cz q[35],q[3];
cz q[32],q[26];
cz q[32],q[6];
cz q[0],q[45];
cz q[0],q[46];
cz q[0],q[6];
cz q[45],q[15];
cz q[45],q[21];
cz q[26],q[33];
cz q[26],q[23];
cz q[33],q[10];
cz q[33],q[49];
cz q[16],q[38];
cz q[16],q[19];
cz q[16],q[17];
cz q[38],q[47];
cz q[17],q[40];
cz q[17],q[39];
cz q[40],q[36];
cz q[40],q[5];
cz q[24],q[48];
cz q[24],q[21];
cz q[24],q[23];
cz q[48],q[5];
cz q[47],q[9];
cz q[47],q[14];
cz q[15],q[8];
cz q[15],q[29];
cz q[39],q[42];
cz q[42],q[37];
cz q[42],q[44];
cz q[18],q[41];
cz q[18],q[22];
cz q[18],q[31];
cz q[41],q[21];
cz q[41],q[34];
cz q[3],q[22];
cz q[3],q[49];
cz q[22],q[10];
cz q[20],q[44];
cz q[20],q[2];
cz q[20],q[31];
cz q[44],q[7];
cz q[7],q[4];
cz q[8],q[29];
cz q[4],q[36];
cz q[4],q[46];
cz q[36],q[12];
cz q[2],q[14];
cz q[2],q[43];
cz q[14],q[49];
cz q[37],q[5];
cz q[12],q[46];
cz q[12],q[27];
cz q[19],q[34];
cz q[19],q[9];
cz q[10],q[11];
cz q[9],q[27];
cz q[29],q[23];
cz q[31],q[11];
cz q[11],q[43];
cz q[11],q[36];
cz q[11],q[39];
cz q[11],q[7];
cz q[36],q[30];
cz q[36],q[7];
cz q[1],q[37];
cz q[1],q[21];
cz q[1],q[29];
cz q[37],q[25];
cz q[37],q[31];
cz q[39],q[24];
cz q[39],q[48];
cz q[24],q[47];
cz q[24],q[49];
cz q[2],q[45];
cz q[2],q[9];
cz q[2],q[30];
cz q[45],q[34];
cz q[45],q[3];
cz q[4],q[21];
cz q[4],q[19];
cz q[4],q[46];
cz q[21],q[20];
cz q[38],q[47];
cz q[38],q[13];
cz q[38],q[26];
cz q[47],q[23];
cz q[3],q[19];
cz q[3],q[0];
cz q[19],q[48];
cz q[23],q[25];
cz q[23],q[12];
cz q[25],q[43];
cz q[26],q[48];
cz q[26],q[15];
cz q[8],q[12];
cz q[8],q[42];
cz q[8],q[16];
cz q[12],q[15];
cz q[34],q[40];
cz q[34],q[44];
cz q[40],q[22];
cz q[40],q[9];
cz q[5],q[28];
cz q[5],q[42];
cz q[5],q[41];
cz q[28],q[22];
cz q[28],q[44];
cz q[17],q[33];
cz q[17],q[20];
cz q[17],q[43];
cz q[33],q[16];
cz q[33],q[0];
cz q[27],q[31];
cz q[27],q[49];
cz q[27],q[20];
cz q[31],q[43];
cz q[30],q[7];
cz q[42],q[18];
cz q[49],q[0];
cz q[6],q[14];
cz q[6],q[32];
cz q[6],q[13];
cz q[14],q[10];
cz q[14],q[44];
cz q[13],q[35];
cz q[35],q[41];
cz q[35],q[32];
cz q[15],q[29];
cz q[29],q[10];
cz q[32],q[41];
cz q[22],q[10];
cz q[18],q[46];
cz q[18],q[16];
cz q[46],q[9];
cz q[2],q[33];
cz q[2],q[38];
cz q[2],q[21];
cz q[33],q[24];
cz q[33],q[40];
cz q[30],q[43];
cz q[30],q[48];
cz q[30],q[21];
cz q[43],q[7];
cz q[43],q[42];
cz q[24],q[3];
cz q[24],q[9];
cz q[4],q[9];
cz q[4],q[26];
cz q[4],q[25];
cz q[9],q[3];
cz q[11],q[48];
cz q[11],q[23];
cz q[11],q[8];
cz q[48],q[21];
cz q[6],q[27];
cz q[6],q[39];
cz q[6],q[10];
cz q[27],q[14];
cz q[27],q[41];
cz q[18],q[35];
cz q[18],q[44];
cz q[18],q[28];
cz q[35],q[29];
cz q[35],q[0];
cz q[31],q[32];
cz q[31],q[13];
cz q[31],q[17];
cz q[32],q[7];
cz q[32],q[38];
cz q[7],q[46];
cz q[29],q[17];
cz q[29],q[1];
cz q[23],q[28];
cz q[23],q[14];
cz q[28],q[14];
cz q[3],q[25];
cz q[25],q[5];
cz q[44],q[10];
cz q[44],q[46];
cz q[39],q[37];
cz q[39],q[47];
cz q[5],q[12];
cz q[5],q[45];
cz q[20],q[47];
cz q[20],q[22];
cz q[20],q[1];
cz q[47],q[17];
cz q[22],q[41];
cz q[22],q[15];
cz q[41],q[10];
cz q[37],q[45];
cz q[37],q[0];
cz q[45],q[34];
cz q[34],q[46];
cz q[34],q[8];
cz q[36],q[49];
cz q[36],q[12];
cz q[36],q[16];
cz q[49],q[26];
cz q[49],q[15];
cz q[0],q[19];
cz q[38],q[26];
cz q[12],q[15];
cz q[8],q[19];
cz q[19],q[1];
cz q[13],q[16];
cz q[13],q[40];
cz q[16],q[42];
cz q[42],q[40];
