OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
cz q[2],q[27];
cz q[2],q[11];
cz q[2],q[23];
cz q[27],q[8];
cz q[27],q[3];
cz q[7],q[17];
cz q[7],q[10];
cz q[7],q[8];
cz q[17],q[15];
cz q[17],q[3];
cz q[20],q[26];
cz q[20],q[9];
cz q[20],q[14];
cz q[26],q[13];
cz q[26],q[25];
cz q[12],q[22];
cz q[12],q[21];
cz q[12],q[5];
cz q[22],q[14];
cz q[22],q[1];
cz q[3],q[19];
cz q[19],q[13];
cz q[19],q[25];
cz q[4],q[18];
cz q[4],q[24];
cz q[4],q[28];
cz q[18],q[9];
cz q[18],q[15];
cz q[24],q[29];
cz q[24],q[23];
cz q[14],q[29];
cz q[0],q[5];
cz q[0],q[28];
cz q[0],q[9];
cz q[5],q[21];
cz q[1],q[6];
cz q[1],q[23];
cz q[6],q[10];
cz q[6],q[25];
cz q[11],q[10];
cz q[11],q[21];
cz q[8],q[16];
cz q[13],q[16];
cz q[15],q[16];
cz q[29],q[28];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[19];
cz q[12],q[25];
cz q[12],q[5];
cz q[13],q[27];
cz q[13],q[1];
cz q[27],q[10];
cz q[27],q[22];
cz q[15],q[24];
cz q[15],q[3];
cz q[15],q[7];
cz q[24],q[1];
cz q[24],q[23];
cz q[16],q[29];
cz q[16],q[28];
cz q[16],q[17];
cz q[29],q[11];
cz q[29],q[1];
cz q[7],q[26];
cz q[7],q[11];
cz q[26],q[18];
cz q[26],q[28];
cz q[18],q[5];
cz q[18],q[2];
cz q[20],q[23];
cz q[20],q[10];
cz q[20],q[14];
cz q[23],q[8];
cz q[4],q[21];
cz q[4],q[2];
cz q[4],q[0];
cz q[21],q[19];
cz q[21],q[9];
cz q[25],q[2];
cz q[25],q[10];
cz q[14],q[22];
cz q[14],q[3];
cz q[22],q[0];
cz q[0],q[8];
cz q[8],q[5];
cz q[19],q[9];
cz q[11],q[17];
cz q[17],q[3];
cz q[28],q[9];
cz q[1],q[28];
cz q[1],q[3];
cz q[1],q[23];
cz q[28],q[14];
cz q[28],q[4];
cz q[16],q[26];
cz q[16],q[15];
cz q[16],q[10];
cz q[26],q[4];
cz q[26],q[12];
cz q[21],q[22];
cz q[21],q[12];
cz q[21],q[2];
cz q[22],q[29];
cz q[22],q[17];
cz q[18],q[29];
cz q[18],q[10];
cz q[18],q[19];
cz q[29],q[27];
cz q[8],q[25];
cz q[8],q[11];
cz q[8],q[5];
cz q[25],q[24];
cz q[25],q[6];
cz q[5],q[19];
cz q[5],q[12];
cz q[19],q[20];
cz q[14],q[15];
cz q[14],q[7];
cz q[3],q[27];
cz q[3],q[11];
cz q[10],q[23];
cz q[0],q[17];
cz q[0],q[20];
cz q[0],q[24];
cz q[17],q[13];
cz q[9],q[23];
cz q[9],q[13];
cz q[9],q[6];
cz q[20],q[24];
cz q[13],q[2];
cz q[15],q[7];
cz q[11],q[27];
cz q[4],q[2];
cz q[6],q[7];
cz q[6],q[12];
cz q[6],q[21];
cz q[6],q[4];
cz q[12],q[16];
cz q[12],q[1];
cz q[21],q[8];
cz q[21],q[16];
cz q[18],q[20];
cz q[18],q[19];
cz q[18],q[25];
cz q[20],q[9];
cz q[20],q[28];
cz q[15],q[27];
cz q[15],q[17];
cz q[15],q[3];
cz q[27],q[5];
cz q[27],q[7];
cz q[4],q[9];
cz q[4],q[10];
cz q[9],q[23];
cz q[16],q[17];
cz q[3],q[28];
cz q[3],q[14];
cz q[28],q[29];
cz q[0],q[8];
cz q[0],q[17];
cz q[0],q[22];
cz q[8],q[26];
cz q[11],q[14];
cz q[11],q[5];
cz q[11],q[24];
cz q[14],q[5];
cz q[1],q[23];
cz q[1],q[22];
cz q[23],q[26];
cz q[10],q[24];
cz q[10],q[25];
cz q[24],q[26];
cz q[7],q[13];
cz q[7],q[2];
cz q[13],q[29];
cz q[13],q[25];
cz q[19],q[2];
cz q[19],q[29];
cz q[2],q[22];
cz q[16],q[20];
cz q[16],q[22];
cz q[16],q[0];
cz q[20],q[10];
cz q[20],q[12];
cz q[2],q[27];
cz q[2],q[5];
cz q[2],q[4];
cz q[27],q[5];
cz q[27],q[7];
cz q[6],q[21];
cz q[6],q[1];
cz q[6],q[23];
cz q[21],q[17];
cz q[21],q[18];
cz q[12],q[13];
cz q[12],q[8];
cz q[13],q[25];
cz q[13],q[4];
cz q[3],q[10];
cz q[3],q[15];
cz q[3],q[24];
cz q[10],q[11];
cz q[22],q[26];
cz q[22],q[28];
cz q[26],q[9];
cz q[26],q[4];
cz q[8],q[17];
cz q[8],q[28];
cz q[17],q[18];
cz q[1],q[9];
cz q[1],q[25];
cz q[18],q[0];
cz q[5],q[14];
cz q[9],q[7];
cz q[11],q[23];
cz q[11],q[29];
cz q[23],q[19];
cz q[19],q[24];
cz q[19],q[25];
cz q[24],q[0];
cz q[29],q[15];
cz q[29],q[14];
cz q[15],q[28];
cz q[7],q[14];
cz q[2],q[27];
cz q[2],q[29];
cz q[2],q[24];
cz q[27],q[17];
cz q[27],q[1];
cz q[6],q[15];
cz q[6],q[16];
cz q[6],q[0];
cz q[15],q[26];
cz q[15],q[28];
cz q[16],q[29];
cz q[16],q[25];
cz q[29],q[3];
cz q[3],q[7];
cz q[3],q[26];
cz q[7],q[28];
cz q[7],q[18];
cz q[12],q[22];
cz q[12],q[25];
cz q[12],q[24];
cz q[22],q[20];
cz q[22],q[8];
cz q[4],q[21];
cz q[4],q[11];
cz q[4],q[17];
cz q[21],q[8];
cz q[21],q[26];
cz q[25],q[1];
cz q[5],q[28];
cz q[5],q[9];
cz q[5],q[14];
cz q[8],q[17];
cz q[13],q[14];
cz q[13],q[20];
cz q[13],q[24];
cz q[14],q[23];
cz q[0],q[20];
cz q[0],q[10];
cz q[1],q[11];
cz q[9],q[19];
cz q[9],q[18];
cz q[11],q[18];
cz q[19],q[23];
cz q[19],q[10];
cz q[23],q[10];
cz q[6],q[12];
cz q[6],q[14];
cz q[6],q[28];
cz q[12],q[20];
cz q[12],q[0];
cz q[18],q[23];
cz q[18],q[0];
cz q[18],q[15];
cz q[23],q[14];
cz q[23],q[3];
cz q[7],q[29];
cz q[7],q[10];
cz q[7],q[25];
cz q[29],q[9];
cz q[29],q[2];
cz q[8],q[9];
cz q[8],q[17];
cz q[8],q[16];
cz q[9],q[21];
cz q[2],q[11];
cz q[2],q[22];
cz q[11],q[3];
cz q[11],q[15];
cz q[5],q[28];
cz q[5],q[15];
cz q[5],q[3];
cz q[28],q[20];
cz q[0],q[17];
cz q[17],q[26];
cz q[19],q[24];
cz q[19],q[16];
cz q[19],q[1];
cz q[24],q[26];
cz q[24],q[25];
cz q[10],q[21];
cz q[10],q[13];
cz q[21],q[20];
cz q[16],q[13];
cz q[14],q[26];
cz q[1],q[27];
cz q[1],q[4];
cz q[27],q[25];
cz q[27],q[22];
cz q[13],q[4];
cz q[22],q[4];
cz q[6],q[12];
cz q[6],q[15];
cz q[6],q[26];
cz q[12],q[22];
cz q[12],q[24];
cz q[16],q[20];
cz q[16],q[3];
cz q[16],q[4];
cz q[20],q[28];
cz q[20],q[14];
cz q[15],q[2];
cz q[15],q[18];
cz q[3],q[23];
cz q[3],q[29];
cz q[22],q[9];
cz q[22],q[2];
cz q[14],q[19];
cz q[14],q[25];
cz q[19],q[23];
cz q[19],q[4];
cz q[25],q[7];
cz q[25],q[17];
cz q[5],q[10];
cz q[5],q[0];
cz q[5],q[21];
cz q[10],q[27];
cz q[10],q[1];
cz q[0],q[2];
cz q[0],q[8];
cz q[4],q[27];
cz q[27],q[7];
cz q[17],q[18];
cz q[17],q[13];
cz q[18],q[11];
cz q[8],q[24];
cz q[8],q[1];
cz q[24],q[9];
cz q[13],q[29];
cz q[13],q[28];
cz q[1],q[21];
cz q[21],q[11];
cz q[29],q[26];
cz q[26],q[23];
cz q[7],q[9];
cz q[28],q[11];
cz q[2],q[27];
cz q[2],q[23];
cz q[2],q[6];
cz q[27],q[15];
cz q[27],q[25];
cz q[15],q[4];
cz q[15],q[12];
cz q[20],q[23];
cz q[20],q[28];
cz q[20],q[10];
cz q[23],q[13];
cz q[4],q[18];
cz q[4],q[13];
cz q[12],q[22];
cz q[12],q[7];
cz q[22],q[5];
cz q[22],q[7];
cz q[14],q[19];
cz q[14],q[8];
cz q[14],q[3];
cz q[19],q[24];
cz q[19],q[1];
cz q[18],q[1];
cz q[18],q[9];
cz q[17],q[21];
cz q[17],q[29];
cz q[17],q[28];
cz q[21],q[3];
cz q[21],q[26];
cz q[0],q[8];
cz q[0],q[11];
cz q[0],q[24];
cz q[8],q[5];
cz q[5],q[26];
cz q[1],q[3];
cz q[11],q[26];
cz q[11],q[16];
cz q[24],q[25];
cz q[13],q[29];
cz q[7],q[10];
cz q[10],q[9];
cz q[29],q[6];
cz q[16],q[28];
cz q[16],q[6];
cz q[9],q[25];
