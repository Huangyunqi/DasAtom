OPENQASM 2.0;
include "qelib1.inc";
qreg q[40];
cz q[9],q[39];
cz q[9],q[31];
cz q[9],q[0];
cz q[39],q[33];
cz q[39],q[28];
cz q[15],q[24];
cz q[15],q[7];
cz q[15],q[22];
cz q[24],q[35];
cz q[24],q[25];
cz q[26],q[30];
cz q[26],q[2];
cz q[26],q[3];
cz q[30],q[16];
cz q[30],q[11];
cz q[16],q[29];
cz q[16],q[27];
cz q[29],q[6];
cz q[29],q[3];
cz q[1],q[34];
cz q[1],q[20];
cz q[1],q[23];
cz q[34],q[25];
cz q[34],q[13];
cz q[33],q[23];
cz q[33],q[20];
cz q[2],q[36];
cz q[2],q[17];
cz q[36],q[23];
cz q[36],q[18];
cz q[12],q[13];
cz q[12],q[19];
cz q[12],q[27];
cz q[13],q[3];
cz q[19],q[32];
cz q[19],q[22];
cz q[7],q[35];
cz q[7],q[32];
cz q[35],q[0];
cz q[20],q[32];
cz q[4],q[21];
cz q[4],q[10];
cz q[4],q[28];
cz q[21],q[38];
cz q[21],q[11];
cz q[18],q[38];
cz q[18],q[5];
cz q[38],q[10];
cz q[0],q[8];
cz q[8],q[10];
cz q[8],q[37];
cz q[14],q[31];
cz q[14],q[6];
cz q[14],q[5];
cz q[31],q[17];
cz q[27],q[37];
cz q[37],q[6];
cz q[17],q[5];
cz q[28],q[11];
cz q[25],q[22];
cz q[10],q[34];
cz q[10],q[30];
cz q[10],q[0];
cz q[34],q[36];
cz q[34],q[0];
cz q[16],q[29];
cz q[16],q[7];
cz q[16],q[27];
cz q[29],q[35];
cz q[29],q[33];
cz q[18],q[26];
cz q[18],q[38];
cz q[18],q[31];
cz q[26],q[2];
cz q[26],q[31];
cz q[6],q[24];
cz q[6],q[17];
cz q[6],q[25];
cz q[24],q[35];
cz q[24],q[9];
cz q[12],q[19];
cz q[12],q[31];
cz q[12],q[38];
cz q[19],q[13];
cz q[19],q[4];
cz q[35],q[9];
cz q[5],q[7];
cz q[5],q[4];
cz q[5],q[32];
cz q[7],q[11];
cz q[38],q[37];
cz q[3],q[13];
cz q[3],q[30];
cz q[3],q[36];
cz q[13],q[28];
cz q[23],q[25];
cz q[23],q[2];
cz q[23],q[8];
cz q[25],q[28];
cz q[17],q[21];
cz q[17],q[2];
cz q[21],q[33];
cz q[21],q[11];
cz q[0],q[11];
cz q[9],q[37];
cz q[30],q[14];
cz q[1],q[15];
cz q[1],q[27];
cz q[1],q[14];
cz q[15],q[32];
cz q[15],q[22];
cz q[28],q[8];
cz q[32],q[33];
cz q[8],q[39];
cz q[39],q[14];
cz q[39],q[22];
cz q[27],q[22];
cz q[4],q[20];
cz q[20],q[37];
cz q[20],q[36];
cz q[15],q[21];
cz q[15],q[16];
cz q[15],q[0];
cz q[21],q[29];
cz q[21],q[26];
cz q[9],q[39];
cz q[9],q[26];
cz q[9],q[32];
cz q[39],q[4];
cz q[39],q[14];
cz q[32],q[34];
cz q[32],q[29];
cz q[34],q[14];
cz q[34],q[25];
cz q[29],q[13];
cz q[12],q[28];
cz q[12],q[24];
cz q[12],q[30];
cz q[28],q[35];
cz q[28],q[17];
cz q[0],q[5];
cz q[0],q[35];
cz q[5],q[19];
cz q[5],q[8];
cz q[19],q[37];
cz q[19],q[16];
cz q[26],q[10];
cz q[14],q[10];
cz q[4],q[23];
cz q[4],q[1];
cz q[37],q[7];
cz q[37],q[8];
cz q[10],q[24];
cz q[24],q[2];
cz q[13],q[23];
cz q[13],q[7];
cz q[23],q[30];
cz q[6],q[11];
cz q[6],q[31];
cz q[6],q[2];
cz q[11],q[33];
cz q[11],q[3];
cz q[35],q[38];
cz q[16],q[27];
cz q[25],q[31];
cz q[25],q[1];
cz q[31],q[7];
cz q[1],q[33];
cz q[33],q[18];
cz q[2],q[38];
cz q[38],q[36];
cz q[20],q[22];
cz q[20],q[17];
cz q[20],q[36];
cz q[22],q[27];
cz q[22],q[30];
cz q[17],q[3];
cz q[27],q[18];
cz q[3],q[8];
cz q[18],q[36];
cz q[25],q[35];
cz q[25],q[34];
cz q[25],q[17];
cz q[35],q[11];
cz q[35],q[30];
cz q[7],q[17];
cz q[7],q[32];
cz q[7],q[11];
cz q[17],q[18];
cz q[4],q[9];
cz q[4],q[24];
cz q[4],q[13];
cz q[9],q[31];
cz q[9],q[36];
cz q[13],q[39];
cz q[13],q[1];
cz q[39],q[19];
cz q[39],q[31];
cz q[6],q[27];
cz q[6],q[3];
cz q[6],q[28];
cz q[27],q[33];
cz q[27],q[22];
cz q[26],q[36];
cz q[26],q[31];
cz q[26],q[3];
cz q[36],q[20];
cz q[12],q[19];
cz q[12],q[37];
cz q[12],q[2];
cz q[19],q[16];
cz q[32],q[11];
cz q[32],q[24];
cz q[18],q[38];
cz q[18],q[0];
cz q[38],q[29];
cz q[38],q[8];
cz q[29],q[28];
cz q[29],q[10];
cz q[24],q[37];
cz q[14],q[22];
cz q[14],q[21];
cz q[14],q[30];
cz q[22],q[34];
cz q[37],q[30];
cz q[23],q[34];
cz q[23],q[33];
cz q[23],q[5];
cz q[2],q[5];
cz q[2],q[15];
cz q[5],q[15];
cz q[10],q[21];
cz q[10],q[8];
cz q[21],q[3];
cz q[1],q[15];
cz q[1],q[0];
cz q[16],q[0];
cz q[16],q[8];
cz q[20],q[28];
cz q[20],q[33];
cz q[1],q[28];
cz q[1],q[29];
cz q[1],q[25];
cz q[28],q[18];
cz q[28],q[11];
cz q[11],q[36];
cz q[11],q[10];
cz q[36],q[17];
cz q[36],q[39];
cz q[16],q[26];
cz q[16],q[7];
cz q[16],q[24];
cz q[26],q[32];
cz q[26],q[31];
cz q[13],q[27];
cz q[13],q[3];
cz q[13],q[18];
cz q[27],q[17];
cz q[27],q[5];
cz q[7],q[20];
cz q[7],q[39];
cz q[20],q[38];
cz q[20],q[33];
cz q[12],q[19];
cz q[12],q[22];
cz q[12],q[15];
cz q[19],q[0];
cz q[19],q[35];
cz q[22],q[29];
cz q[22],q[2];
cz q[3],q[33];
cz q[3],q[8];
cz q[29],q[39];
cz q[38],q[33];
cz q[38],q[23];
cz q[17],q[21];
cz q[21],q[32];
cz q[21],q[0];
cz q[2],q[5];
cz q[2],q[37];
cz q[5],q[4];
cz q[8],q[30];
cz q[8],q[23];
cz q[30],q[10];
cz q[30],q[34];
cz q[10],q[9];
cz q[6],q[23];
cz q[6],q[25];
cz q[6],q[37];
cz q[32],q[14];
cz q[4],q[14];
cz q[4],q[35];
cz q[18],q[25];
cz q[15],q[35];
cz q[15],q[9];
cz q[14],q[24];
cz q[31],q[37];
cz q[31],q[34];
cz q[9],q[24];
cz q[34],q[0];
cz q[32],q[37];
cz q[32],q[33];
cz q[32],q[11];
cz q[37],q[27];
cz q[37],q[35];
cz q[18],q[20];
cz q[18],q[17];
cz q[18],q[3];
cz q[20],q[9];
cz q[20],q[27];
cz q[6],q[24];
cz q[6],q[4];
cz q[6],q[27];
cz q[24],q[2];
cz q[24],q[28];
cz q[26],q[33];
cz q[26],q[9];
cz q[26],q[1];
cz q[33],q[29];
cz q[4],q[39];
cz q[4],q[0];
cz q[13],q[39];
cz q[13],q[17];
cz q[13],q[21];
cz q[39],q[8];
cz q[12],q[19];
cz q[12],q[22];
cz q[12],q[38];
cz q[19],q[14];
cz q[19],q[7];
cz q[22],q[29];
cz q[22],q[5];
cz q[14],q[31];
cz q[14],q[7];
cz q[29],q[5];
cz q[23],q[34];
cz q[23],q[30];
cz q[23],q[10];
cz q[34],q[31];
cz q[34],q[36];
cz q[31],q[3];
cz q[17],q[16];
cz q[3],q[11];
cz q[9],q[0];
cz q[2],q[38];
cz q[2],q[25];
cz q[10],q[21];
cz q[10],q[28];
cz q[21],q[35];
cz q[5],q[11];
cz q[1],q[15];
cz q[1],q[36];
cz q[15],q[38];
cz q[15],q[0];
cz q[30],q[36];
cz q[30],q[28];
cz q[8],q[35];
cz q[8],q[16];
cz q[7],q[25];
cz q[16],q[25];
cz q[15],q[30];
cz q[15],q[3];
cz q[15],q[37];
cz q[30],q[6];
cz q[30],q[20];
cz q[6],q[4];
cz q[6],q[10];
cz q[4],q[7];
cz q[4],q[34];
cz q[12],q[19];
cz q[12],q[28];
cz q[12],q[26];
cz q[19],q[18];
cz q[19],q[29];
cz q[29],q[38];
cz q[29],q[27];
cz q[38],q[22];
cz q[38],q[32];
cz q[3],q[25];
cz q[3],q[28];
cz q[25],q[16];
cz q[25],q[24];
cz q[28],q[13];
cz q[21],q[37];
cz q[21],q[39];
cz q[21],q[35];
cz q[37],q[31];
cz q[11],q[14];
cz q[11],q[10];
cz q[11],q[16];
cz q[14],q[27];
cz q[14],q[5];
cz q[22],q[7];
cz q[22],q[17];
cz q[5],q[31];
cz q[5],q[36];
cz q[31],q[26];
cz q[17],q[33];
cz q[17],q[8];
cz q[33],q[8];
cz q[33],q[9];
cz q[9],q[23];
cz q[9],q[36];
cz q[23],q[27];
cz q[23],q[32];
cz q[0],q[26];
cz q[0],q[32];
cz q[0],q[7];
cz q[8],q[20];
cz q[10],q[39];
cz q[39],q[18];
cz q[18],q[2];
cz q[16],q[13];
cz q[1],q[36];
cz q[1],q[2];
cz q[1],q[35];
cz q[2],q[34];
cz q[20],q[24];
cz q[13],q[24];
cz q[35],q[34];
cz q[24],q[30];
cz q[24],q[26];
cz q[24],q[21];
cz q[30],q[13];
cz q[30],q[6];
cz q[13],q[16];
cz q[13],q[10];
cz q[7],q[17];
cz q[7],q[5];
cz q[7],q[15];
cz q[17],q[8];
cz q[17],q[3];
cz q[18],q[23];
cz q[18],q[14];
cz q[18],q[16];
cz q[23],q[15];
cz q[23],q[27];
cz q[1],q[34];
cz q[1],q[26];
cz q[1],q[25];
cz q[34],q[20];
cz q[34],q[9];
cz q[6],q[8];
cz q[6],q[25];
cz q[16],q[38];
cz q[38],q[11];
cz q[38],q[32];
cz q[21],q[22];
cz q[21],q[31];
cz q[22],q[31];
cz q[22],q[15];
cz q[5],q[0];
cz q[5],q[2];
cz q[31],q[29];
cz q[0],q[35];
cz q[0],q[28];
cz q[9],q[14];
cz q[9],q[32];
cz q[14],q[8];
cz q[3],q[28];
cz q[3],q[29];
cz q[28],q[36];
cz q[2],q[10];
cz q[2],q[19];
cz q[4],q[39];
cz q[4],q[11];
cz q[4],q[37];
cz q[39],q[32];
cz q[39],q[20];
cz q[36],q[19];
cz q[36],q[12];
cz q[19],q[10];
cz q[26],q[25];
cz q[35],q[33];
cz q[35],q[37];
cz q[11],q[12];
cz q[33],q[27];
cz q[33],q[20];
cz q[29],q[37];
cz q[12],q[27];
cz q[11],q[36];
cz q[11],q[21];
cz q[11],q[30];
cz q[36],q[10];
cz q[36],q[7];
cz q[3],q[4];
cz q[3],q[12];
cz q[3],q[2];
cz q[4],q[0];
cz q[4],q[16];
cz q[18],q[26];
cz q[18],q[34];
cz q[18],q[39];
cz q[26],q[28];
cz q[26],q[31];
cz q[35],q[39];
cz q[35],q[7];
cz q[35],q[6];
cz q[39],q[22];
cz q[7],q[16];
cz q[5],q[13];
cz q[5],q[9];
cz q[5],q[32];
cz q[13],q[23];
cz q[13],q[0];
cz q[23],q[25];
cz q[23],q[15];
cz q[25],q[16];
cz q[25],q[1];
cz q[10],q[21];
cz q[10],q[2];
cz q[21],q[32];
cz q[19],q[30];
cz q[19],q[9];
cz q[19],q[17];
cz q[30],q[31];
cz q[9],q[29];
cz q[29],q[8];
cz q[29],q[1];
cz q[8],q[27];
cz q[8],q[37];
cz q[27],q[14];
cz q[27],q[20];
cz q[17],q[34];
cz q[17],q[12];
cz q[34],q[28];
cz q[2],q[20];
cz q[20],q[22];
cz q[15],q[38];
cz q[15],q[37];
cz q[24],q[38];
cz q[24],q[14];
cz q[24],q[22];
cz q[38],q[6];
cz q[6],q[32];
cz q[12],q[33];
cz q[0],q[1];
cz q[33],q[14];
cz q[33],q[37];
cz q[31],q[28];
