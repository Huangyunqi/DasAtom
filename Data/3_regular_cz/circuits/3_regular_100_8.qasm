OPENQASM 2.0;
include "qelib1.inc";
qreg q[100];
cz q[36],q[53];
cz q[36],q[0];
cz q[36],q[18];
cz q[53],q[22];
cz q[53],q[41];
cz q[67],q[68];
cz q[67],q[57];
cz q[67],q[60];
cz q[68],q[40];
cz q[68],q[66];
cz q[10],q[98];
cz q[10],q[31];
cz q[10],q[42];
cz q[98],q[91];
cz q[98],q[29];
cz q[16],q[93];
cz q[16],q[99];
cz q[16],q[56];
cz q[93],q[33];
cz q[93],q[45];
cz q[1],q[49];
cz q[1],q[18];
cz q[1],q[50];
cz q[49],q[5];
cz q[49],q[94];
cz q[87],q[89];
cz q[87],q[48];
cz q[87],q[59];
cz q[89],q[45];
cz q[89],q[79];
cz q[17],q[21];
cz q[17],q[85];
cz q[17],q[86];
cz q[21],q[55];
cz q[21],q[83];
cz q[85],q[19];
cz q[85],q[8];
cz q[91],q[4];
cz q[91],q[82];
cz q[14],q[40];
cz q[14],q[62];
cz q[14],q[39];
cz q[40],q[57];
cz q[22],q[73];
cz q[22],q[61];
cz q[55],q[7];
cz q[55],q[73];
cz q[24],q[81];
cz q[24],q[66];
cz q[24],q[84];
cz q[81],q[63];
cz q[81],q[51];
cz q[27],q[43];
cz q[27],q[25];
cz q[27],q[96];
cz q[43],q[35];
cz q[43],q[95];
cz q[18],q[74];
cz q[74],q[31];
cz q[74],q[80];
cz q[33],q[31];
cz q[33],q[82];
cz q[2],q[96];
cz q[2],q[90];
cz q[2],q[62];
cz q[96],q[56];
cz q[37],q[63];
cz q[37],q[50];
cz q[37],q[54];
cz q[63],q[54];
cz q[42],q[59];
cz q[42],q[80];
cz q[59],q[3];
cz q[39],q[44];
cz q[39],q[65];
cz q[44],q[83];
cz q[44],q[71];
cz q[29],q[71];
cz q[29],q[13];
cz q[51],q[82];
cz q[51],q[13];
cz q[9],q[28];
cz q[9],q[60];
cz q[9],q[38];
cz q[28],q[23];
cz q[28],q[72];
cz q[15],q[25];
cz q[15],q[97];
cz q[15],q[26];
cz q[25],q[62];
cz q[72],q[77];
cz q[72],q[12];
cz q[77],q[64];
cz q[77],q[41];
cz q[11],q[46];
cz q[11],q[3];
cz q[11],q[58];
cz q[46],q[79];
cz q[46],q[12];
cz q[20],q[88];
cz q[20],q[26];
cz q[20],q[32];
cz q[88],q[57];
cz q[88],q[19];
cz q[54],q[99];
cz q[99],q[6];
cz q[56],q[8];
cz q[5],q[78];
cz q[5],q[47];
cz q[78],q[92];
cz q[78],q[94];
cz q[58],q[90];
cz q[58],q[75];
cz q[90],q[86];
cz q[7],q[69];
cz q[7],q[47];
cz q[69],q[41];
cz q[69],q[4];
cz q[64],q[84];
cz q[64],q[65];
cz q[84],q[79];
cz q[26],q[66];
cz q[60],q[30];
cz q[45],q[47];
cz q[86],q[75];
cz q[13],q[61];
cz q[4],q[0];
cz q[71],q[92];
cz q[92],q[48];
cz q[38],q[52];
cz q[38],q[30];
cz q[52],q[70];
cz q[52],q[95];
cz q[23],q[97];
cz q[23],q[94];
cz q[97],q[32];
cz q[83],q[76];
cz q[30],q[95];
cz q[48],q[61];
cz q[73],q[75];
cz q[12],q[76];
cz q[8],q[80];
cz q[34],q[50];
cz q[34],q[70];
cz q[34],q[3];
cz q[76],q[70];
cz q[0],q[65];
cz q[32],q[6];
cz q[19],q[35];
cz q[6],q[35];
cz q[68],q[91];
cz q[68],q[81];
cz q[68],q[99];
cz q[91],q[49];
cz q[91],q[24];
cz q[59],q[64];
cz q[59],q[67];
cz q[59],q[20];
cz q[64],q[83];
cz q[64],q[57];
cz q[16],q[38];
cz q[16],q[92];
cz q[16],q[10];
cz q[38],q[0];
cz q[38],q[58];
cz q[8],q[18];
cz q[8],q[39];
cz q[8],q[7];
cz q[18],q[76];
cz q[18],q[21];
cz q[17],q[30];
cz q[17],q[96];
cz q[17],q[52];
cz q[30],q[41];
cz q[30],q[32];
cz q[9],q[81];
cz q[9],q[80];
cz q[9],q[46];
cz q[81],q[90];
cz q[46],q[75];
cz q[46],q[86];
cz q[75],q[15];
cz q[75],q[35];
cz q[66],q[87];
cz q[66],q[49];
cz q[66],q[5];
cz q[87],q[7];
cz q[87],q[70];
cz q[23],q[61];
cz q[23],q[24];
cz q[23],q[71];
cz q[61],q[6];
cz q[61],q[90];
cz q[39],q[35];
cz q[39],q[50];
cz q[26],q[32];
cz q[26],q[52];
cz q[26],q[40];
cz q[32],q[43];
cz q[2],q[96];
cz q[2],q[78];
cz q[2],q[55];
cz q[96],q[27];
cz q[1],q[42];
cz q[1],q[21];
cz q[1],q[53];
cz q[42],q[63];
cz q[42],q[5];
cz q[3],q[6];
cz q[3],q[58];
cz q[3],q[41];
cz q[6],q[10];
cz q[45],q[85];
cz q[45],q[89];
cz q[45],q[72];
cz q[85],q[29];
cz q[85],q[55];
cz q[31],q[40];
cz q[31],q[37];
cz q[31],q[89];
cz q[40],q[93];
cz q[60],q[76];
cz q[60],q[20];
cz q[60],q[53];
cz q[76],q[97];
cz q[15],q[62];
cz q[15],q[56];
cz q[62],q[71];
cz q[62],q[22];
cz q[25],q[27];
cz q[25],q[79];
cz q[25],q[94];
cz q[27],q[44];
cz q[21],q[67];
cz q[41],q[86];
cz q[0],q[37];
cz q[0],q[47];
cz q[37],q[56];
cz q[7],q[94];
cz q[94],q[19];
cz q[48],q[49];
cz q[48],q[92];
cz q[48],q[77];
cz q[20],q[97];
cz q[97],q[10];
cz q[12],q[29];
cz q[12],q[35];
cz q[12],q[83];
cz q[29],q[55];
cz q[11],q[73];
cz q[11],q[34];
cz q[11],q[13];
cz q[73],q[82];
cz q[73],q[69];
cz q[14],q[99];
cz q[14],q[44];
cz q[14],q[52];
cz q[99],q[80];
cz q[44],q[70];
cz q[80],q[78];
cz q[63],q[77];
cz q[63],q[69];
cz q[77],q[89];
cz q[54],q[74];
cz q[54],q[4];
cz q[54],q[69];
cz q[74],q[67];
cz q[74],q[19];
cz q[93],q[95];
cz q[93],q[65];
cz q[13],q[51];
cz q[13],q[24];
cz q[51],q[95];
cz q[51],q[4];
cz q[28],q[82];
cz q[28],q[36];
cz q[28],q[47];
cz q[82],q[34];
cz q[95],q[4];
cz q[71],q[33];
cz q[22],q[57];
cz q[22],q[36];
cz q[92],q[19];
cz q[58],q[98];
cz q[36],q[70];
cz q[34],q[5];
cz q[84],q[86];
cz q[84],q[88];
cz q[84],q[98];
cz q[50],q[72];
cz q[50],q[33];
cz q[72],q[88];
cz q[90],q[83];
cz q[79],q[53];
cz q[79],q[98];
cz q[56],q[57];
cz q[88],q[47];
cz q[43],q[65];
cz q[43],q[33];
cz q[65],q[78];
cz q[68],q[91];
cz q[68],q[2];
cz q[68],q[13];
cz q[91],q[10];
cz q[91],q[9];
cz q[49],q[87];
cz q[49],q[38];
cz q[49],q[74];
cz q[87],q[44];
cz q[87],q[76];
cz q[24],q[33];
cz q[24],q[73];
cz q[24],q[5];
cz q[33],q[69];
cz q[33],q[82];
cz q[2],q[39];
cz q[2],q[1];
cz q[39],q[14];
cz q[39],q[23];
cz q[16],q[47];
cz q[16],q[77];
cz q[16],q[10];
cz q[47],q[93];
cz q[47],q[5];
cz q[1],q[58];
cz q[1],q[27];
cz q[58],q[20];
cz q[58],q[25];
cz q[56],q[92];
cz q[56],q[0];
cz q[56],q[81];
cz q[92],q[80];
cz q[92],q[51];
cz q[83],q[85];
cz q[83],q[13];
cz q[83],q[64];
cz q[85],q[97];
cz q[85],q[55];
cz q[67],q[98];
cz q[67],q[76];
cz q[67],q[40];
cz q[98],q[34];
cz q[98],q[62];
cz q[4],q[94];
cz q[4],q[77];
cz q[4],q[72];
cz q[94],q[61];
cz q[94],q[59];
cz q[6],q[57];
cz q[6],q[54];
cz q[6],q[37];
cz q[57],q[65];
cz q[57],q[54];
cz q[21],q[64];
cz q[21],q[41];
cz q[21],q[76];
cz q[64],q[7];
cz q[36],q[46];
cz q[36],q[50];
cz q[36],q[52];
cz q[46],q[62];
cz q[46],q[82];
cz q[77],q[84];
cz q[10],q[29];
cz q[88],q[93];
cz q[88],q[95];
cz q[88],q[81];
cz q[93],q[12];
cz q[18],q[28];
cz q[18],q[89];
cz q[18],q[63];
cz q[28],q[69];
cz q[28],q[11];
cz q[35],q[44];
cz q[35],q[23];
cz q[35],q[54];
cz q[44],q[95];
cz q[26],q[96];
cz q[26],q[38];
cz q[26],q[99];
cz q[96],q[9];
cz q[96],q[65];
cz q[69],q[79];
cz q[19],q[66];
cz q[19],q[34];
cz q[19],q[72];
cz q[66],q[51];
cz q[66],q[99];
cz q[17],q[78];
cz q[17],q[31];
cz q[17],q[90];
cz q[78],q[14];
cz q[78],q[70];
cz q[30],q[75];
cz q[30],q[74];
cz q[30],q[37];
cz q[75],q[45];
cz q[75],q[20];
cz q[3],q[97];
cz q[3],q[48];
cz q[3],q[11];
cz q[97],q[55];
cz q[20],q[41];
cz q[13],q[0];
cz q[29],q[86];
cz q[29],q[42];
cz q[25],q[82];
cz q[25],q[53];
cz q[95],q[63];
cz q[34],q[60];
cz q[60],q[50];
cz q[60],q[81];
cz q[8],q[59];
cz q[8],q[52];
cz q[8],q[22];
cz q[59],q[52];
cz q[51],q[32];
cz q[40],q[45];
cz q[40],q[37];
cz q[45],q[42];
cz q[12],q[22];
cz q[12],q[32];
cz q[41],q[86];
cz q[15],q[73];
cz q[15],q[38];
cz q[15],q[11];
cz q[73],q[31];
cz q[50],q[74];
cz q[42],q[80];
cz q[14],q[65];
cz q[22],q[23];
cz q[63],q[43];
cz q[86],q[99];
cz q[80],q[84];
cz q[89],q[71];
cz q[89],q[27];
cz q[70],q[84];
cz q[70],q[62];
cz q[7],q[43];
cz q[7],q[79];
cz q[53],q[61];
cz q[53],q[79];
cz q[61],q[71];
cz q[55],q[48];
cz q[5],q[43];
cz q[9],q[27];
cz q[71],q[48];
cz q[72],q[90];
cz q[90],q[32];
cz q[0],q[31];
cz q[44],q[91];
cz q[44],q[97];
cz q[44],q[20];
cz q[91],q[12];
cz q[91],q[11];
cz q[13],q[88];
cz q[13],q[55];
cz q[13],q[64];
cz q[88],q[10];
cz q[88],q[4];
cz q[53],q[69];
cz q[53],q[27];
cz q[53],q[99];
cz q[69],q[92];
cz q[69],q[48];
cz q[18],q[90];
cz q[18],q[98];
cz q[18],q[49];
cz q[90],q[39];
cz q[90],q[25];
cz q[56],q[74];
cz q[56],q[87];
cz q[56],q[57];
cz q[74],q[86];
cz q[74],q[42];
cz q[42],q[66];
cz q[42],q[33];
cz q[66],q[89];
cz q[66],q[41];
cz q[3],q[77];
cz q[3],q[11];
cz q[3],q[32];
cz q[77],q[19];
cz q[77],q[17];
cz q[14],q[22];
cz q[14],q[8];
cz q[14],q[27];
cz q[22],q[2];
cz q[22],q[55];
cz q[34],q[37];
cz q[34],q[4];
cz q[34],q[72];
cz q[37],q[86];
cz q[37],q[95];
cz q[60],q[83];
cz q[60],q[54];
cz q[60],q[29];
cz q[83],q[96];
cz q[83],q[20];
cz q[28],q[30];
cz q[28],q[12];
cz q[28],q[84];
cz q[30],q[78];
cz q[30],q[11];
cz q[4],q[48];
cz q[48],q[0];
cz q[1],q[33];
cz q[1],q[19];
cz q[1],q[0];
cz q[33],q[52];
cz q[24],q[99];
cz q[24],q[6];
cz q[24],q[75];
cz q[99],q[67];
cz q[16],q[31];
cz q[16],q[58];
cz q[16],q[64];
cz q[31],q[59];
cz q[31],q[70];
cz q[39],q[61];
cz q[39],q[67];
cz q[12],q[50];
cz q[65],q[72];
cz q[65],q[97];
cz q[65],q[57];
cz q[72],q[2];
cz q[29],q[52];
cz q[29],q[15];
cz q[52],q[43];
cz q[50],q[84];
cz q[50],q[47];
cz q[84],q[10];
cz q[96],q[61];
cz q[96],q[6];
cz q[89],q[46];
cz q[89],q[85];
cz q[23],q[63];
cz q[23],q[40];
cz q[23],q[73];
cz q[63],q[93];
cz q[63],q[25];
cz q[62],q[76];
cz q[62],q[54];
cz q[62],q[25];
cz q[76],q[75];
cz q[76],q[92];
cz q[57],q[98];
cz q[98],q[19];
cz q[7],q[94];
cz q[7],q[32];
cz q[7],q[86];
cz q[94],q[35];
cz q[94],q[0];
cz q[8],q[59];
cz q[8],q[71];
cz q[59],q[80];
cz q[93],q[26];
cz q[93],q[5];
cz q[87],q[41];
cz q[87],q[68];
cz q[75],q[9];
cz q[9],q[85];
cz q[9],q[79];
cz q[85],q[51];
cz q[47],q[95];
cz q[47],q[97];
cz q[81],q[82];
cz q[81],q[21];
cz q[81],q[43];
cz q[82],q[32];
cz q[82],q[67];
cz q[35],q[95];
cz q[35],q[79];
cz q[2],q[27];
cz q[20],q[15];
cz q[40],q[6];
cz q[40],q[36];
cz q[92],q[80];
cz q[43],q[73];
cz q[73],q[61];
cz q[80],q[79];
cz q[45],q[46];
cz q[45],q[21];
cz q[45],q[26];
cz q[46],q[49];
cz q[10],q[64];
cz q[51],q[70];
cz q[51],q[38];
cz q[70],q[38];
cz q[21],q[78];
cz q[58],q[36];
cz q[58],q[26];
cz q[41],q[68];
cz q[54],q[15];
cz q[55],q[71];
cz q[17],q[68];
cz q[17],q[5];
cz q[49],q[36];
cz q[71],q[78];
cz q[38],q[5];
cz q[10],q[34];
cz q[10],q[67];
cz q[10],q[50];
cz q[34],q[22];
cz q[34],q[93];
cz q[7],q[81];
cz q[7],q[39];
cz q[7],q[95];
cz q[81],q[87];
cz q[81],q[8];
cz q[6],q[91];
cz q[6],q[8];
cz q[6],q[30];
cz q[91],q[45];
cz q[91],q[5];
cz q[11],q[51];
cz q[11],q[75];
cz q[11],q[33];
cz q[51],q[96];
cz q[51],q[61];
cz q[63],q[89];
cz q[63],q[88];
cz q[63],q[31];
cz q[89],q[99];
cz q[89],q[46];
cz q[70],q[82];
cz q[70],q[5];
cz q[70],q[9];
cz q[82],q[17];
cz q[82],q[35];
cz q[25],q[59];
cz q[25],q[40];
cz q[25],q[66];
cz q[59],q[61];
cz q[59],q[31];
cz q[17],q[30];
cz q[17],q[4];
cz q[30],q[40];
cz q[21],q[55];
cz q[21],q[36];
cz q[21],q[33];
cz q[55],q[53];
cz q[55],q[71];
cz q[87],q[24];
cz q[87],q[69];
cz q[77],q[83];
cz q[77],q[2];
cz q[77],q[76];
cz q[83],q[31];
cz q[83],q[90];
cz q[0],q[62];
cz q[0],q[26];
cz q[0],q[92];
cz q[62],q[71];
cz q[62],q[16];
cz q[54],q[97];
cz q[54],q[60];
cz q[54],q[45];
cz q[97],q[84];
cz q[97],q[9];
cz q[57],q[68];
cz q[57],q[93];
cz q[57],q[12];
cz q[68],q[38];
cz q[68],q[43];
cz q[53],q[29];
cz q[53],q[67];
cz q[19],q[32];
cz q[19],q[23];
cz q[19],q[80];
cz q[32],q[73];
cz q[32],q[65];
cz q[42],q[43];
cz q[42],q[15];
cz q[42],q[28];
cz q[43],q[84];
cz q[8],q[50];
cz q[50],q[95];
cz q[96],q[13];
cz q[96],q[72];
cz q[78],q[93];
cz q[78],q[76];
cz q[78],q[2];
cz q[14],q[35];
cz q[14],q[1];
cz q[14],q[98];
cz q[35],q[15];
cz q[94],q[98];
cz q[94],q[79];
cz q[94],q[29];
cz q[98],q[71];
cz q[95],q[27];
cz q[61],q[58];
cz q[79],q[56];
cz q[79],q[48];
cz q[3],q[65];
cz q[3],q[64];
cz q[3],q[47];
cz q[65],q[44];
cz q[12],q[22];
cz q[12],q[41];
cz q[22],q[66];
cz q[20],q[99];
cz q[20],q[85];
cz q[20],q[15];
cz q[99],q[85];
cz q[75],q[26];
cz q[75],q[80];
cz q[73],q[45];
cz q[73],q[52];
cz q[1],q[67];
cz q[1],q[80];
cz q[38],q[47];
cz q[38],q[69];
cz q[52],q[85];
cz q[52],q[24];
cz q[27],q[18];
cz q[27],q[60];
cz q[26],q[88];
cz q[16],q[74];
cz q[16],q[60];
cz q[74],q[48];
cz q[74],q[64];
cz q[24],q[49];
cz q[2],q[29];
cz q[40],q[48];
cz q[88],q[92];
cz q[84],q[86];
cz q[86],q[9];
cz q[86],q[13];
cz q[36],q[72];
cz q[36],q[44];
cz q[44],q[92];
cz q[28],q[41];
cz q[28],q[72];
cz q[41],q[46];
cz q[76],q[90];
cz q[90],q[5];
cz q[58],q[69];
cz q[58],q[23];
cz q[46],q[49];
cz q[49],q[47];
cz q[39],q[37];
cz q[39],q[56];
cz q[13],q[18];
cz q[18],q[56];
cz q[66],q[23];
cz q[37],q[64];
cz q[37],q[4];
cz q[4],q[33];
cz q[36],q[53];
cz q[36],q[18];
cz q[36],q[81];
cz q[53],q[87];
cz q[53],q[21];
cz q[23],q[68];
cz q[23],q[94];
cz q[23],q[78];
cz q[68],q[22];
cz q[68],q[73];
cz q[67],q[77];
cz q[67],q[2];
cz q[67],q[64];
cz q[77],q[75];
cz q[77],q[10];
cz q[87],q[66];
cz q[87],q[37];
cz q[11],q[51];
cz q[11],q[84];
cz q[11],q[95];
cz q[51],q[89];
cz q[51],q[81];
cz q[5],q[65];
cz q[5],q[54];
cz q[5],q[47];
cz q[65],q[70];
cz q[65],q[10];
cz q[46],q[48];
cz q[46],q[41];
cz q[46],q[4];
cz q[48],q[72];
cz q[48],q[60];
cz q[34],q[37];
cz q[34],q[47];
cz q[34],q[3];
cz q[37],q[39];
cz q[70],q[86];
cz q[70],q[88];
cz q[17],q[85];
cz q[17],q[82];
cz q[17],q[97];
cz q[85],q[45];
cz q[85],q[84];
cz q[89],q[25];
cz q[89],q[29];
cz q[72],q[16];
cz q[72],q[99];
cz q[15],q[69];
cz q[15],q[57];
cz q[15],q[31];
cz q[69],q[22];
cz q[69],q[1];
cz q[66],q[32];
cz q[66],q[90];
cz q[7],q[74];
cz q[7],q[76];
cz q[7],q[82];
cz q[74],q[24];
cz q[74],q[35];
cz q[25],q[60];
cz q[25],q[55];
cz q[45],q[27];
cz q[45],q[96];
cz q[22],q[28];
cz q[28],q[12];
cz q[28],q[56];
cz q[43],q[60];
cz q[43],q[8];
cz q[43],q[95];
cz q[38],q[55];
cz q[38],q[27];
cz q[38],q[76];
cz q[55],q[76];
cz q[14],q[33];
cz q[14],q[71];
cz q[14],q[63];
cz q[33],q[44];
cz q[33],q[41];
cz q[18],q[58];
cz q[18],q[62];
cz q[58],q[64];
cz q[58],q[73];
cz q[61],q[93];
cz q[61],q[96];
cz q[61],q[44];
cz q[93],q[30];
cz q[93],q[99];
cz q[24],q[6];
cz q[24],q[9];
cz q[35],q[6];
cz q[35],q[49];
cz q[19],q[32];
cz q[19],q[91];
cz q[19],q[49];
cz q[32],q[63];
cz q[26],q[80];
cz q[26],q[82];
cz q[26],q[40];
cz q[80],q[13];
cz q[80],q[20];
cz q[27],q[71];
cz q[42],q[98];
cz q[42],q[12];
cz q[42],q[83];
cz q[98],q[59];
cz q[98],q[3];
cz q[30],q[41];
cz q[30],q[83];
cz q[88],q[95];
cz q[88],q[31];
cz q[31],q[44];
cz q[96],q[52];
cz q[59],q[83];
cz q[59],q[50];
cz q[52],q[92];
cz q[52],q[94];
cz q[92],q[54];
cz q[92],q[78];
cz q[16],q[21];
cz q[16],q[3];
cz q[8],q[63];
cz q[8],q[20];
cz q[54],q[9];
cz q[0],q[2];
cz q[0],q[79];
cz q[0],q[56];
cz q[2],q[73];
cz q[50],q[79];
cz q[50],q[97];
cz q[79],q[39];
cz q[57],q[90];
cz q[57],q[62];
cz q[40],q[56];
cz q[40],q[4];
cz q[62],q[29];
cz q[97],q[91];
cz q[94],q[78];
cz q[91],q[13];
cz q[12],q[71];
cz q[20],q[64];
cz q[1],q[84];
cz q[1],q[4];
cz q[13],q[49];
cz q[21],q[99];
cz q[81],q[86];
cz q[86],q[75];
cz q[10],q[6];
cz q[39],q[29];
cz q[90],q[47];
cz q[75],q[9];
cz q[18],q[81];
cz q[18],q[0];
cz q[18],q[77];
cz q[81],q[69];
cz q[81],q[90];
cz q[30],q[46];
cz q[30],q[1];
cz q[30],q[8];
cz q[46],q[17];
cz q[46],q[7];
cz q[82],q[93];
cz q[82],q[0];
cz q[82],q[55];
cz q[93],q[6];
cz q[93],q[41];
cz q[47],q[71];
cz q[47],q[16];
cz q[47],q[44];
cz q[71],q[38];
cz q[71],q[99];
cz q[2],q[48];
cz q[2],q[97];
cz q[2],q[23];
cz q[48],q[76];
cz q[48],q[33];
cz q[16],q[42];
cz q[16],q[8];
cz q[56],q[83];
cz q[56],q[67];
cz q[56],q[86];
cz q[83],q[33];
cz q[83],q[8];
cz q[0],q[60];
cz q[60],q[34];
cz q[60],q[97];
cz q[45],q[92];
cz q[45],q[15];
cz q[45],q[17];
cz q[92],q[33];
cz q[92],q[29];
cz q[14],q[22];
cz q[14],q[40];
cz q[14],q[84];
cz q[22],q[21];
cz q[22],q[61];
cz q[29],q[50];
cz q[29],q[3];
cz q[50],q[1];
cz q[50],q[64];
cz q[28],q[94];
cz q[28],q[61];
cz q[28],q[27];
cz q[94],q[96];
cz q[94],q[12];
cz q[40],q[36];
cz q[40],q[74];
cz q[68],q[75];
cz q[68],q[79];
cz q[68],q[74];
cz q[75],q[11];
cz q[75],q[52];
cz q[38],q[25];
cz q[38],q[52];
cz q[66],q[87];
cz q[66],q[37];
cz q[66],q[31];
cz q[87],q[15];
cz q[87],q[80];
cz q[55],q[59];
cz q[55],q[79];
cz q[59],q[9];
cz q[59],q[13];
cz q[10],q[91];
cz q[10],q[20];
cz q[10],q[76];
cz q[91],q[63];
cz q[91],q[37];
cz q[34],q[58];
cz q[34],q[73];
cz q[58],q[61];
cz q[58],q[70];
cz q[15],q[90];
cz q[67],q[76];
cz q[67],q[89];
cz q[44],q[24];
cz q[44],q[6];
cz q[24],q[78];
cz q[24],q[86];
cz q[63],q[99];
cz q[63],q[39];
cz q[96],q[80];
cz q[96],q[3];
cz q[20],q[25];
cz q[20],q[9];
cz q[19],q[32];
cz q[19],q[54];
cz q[19],q[49];
cz q[32],q[6];
cz q[32],q[51];
cz q[88],q[95];
cz q[88],q[51];
cz q[88],q[89];
cz q[95],q[78];
cz q[95],q[72];
cz q[79],q[62];
cz q[57],q[98];
cz q[57],q[35];
cz q[57],q[99];
cz q[98],q[35];
cz q[98],q[41];
cz q[42],q[23];
cz q[42],q[77];
cz q[69],q[62];
cz q[69],q[39];
cz q[9],q[85];
cz q[85],q[65];
cz q[85],q[36];
cz q[4],q[43];
cz q[4],q[17];
cz q[4],q[21];
cz q[43],q[84];
cz q[43],q[1];
cz q[13],q[21];
cz q[13],q[73];
cz q[62],q[12];
cz q[11],q[39];
cz q[11],q[36];
cz q[25],q[73];
cz q[5],q[7];
cz q[5],q[64];
cz q[5],q[26];
cz q[7],q[27];
cz q[31],q[35];
cz q[31],q[37];
cz q[78],q[27];
cz q[54],q[49];
cz q[54],q[26];
cz q[64],q[65];
cz q[77],q[80];
cz q[52],q[53];
cz q[23],q[97];
cz q[84],q[49];
cz q[3],q[51];
cz q[53],q[65];
cz q[53],q[90];
cz q[72],q[70];
cz q[72],q[26];
cz q[12],q[74];
cz q[70],q[89];
cz q[41],q[86];
cz q[7],q[72];
cz q[7],q[76];
cz q[7],q[98];
cz q[72],q[42];
cz q[72],q[76];
cz q[41],q[49];
cz q[41],q[30];
cz q[41],q[39];
cz q[49],q[80];
cz q[49],q[39];
cz q[19],q[55];
cz q[19],q[97];
cz q[19],q[62];
cz q[55],q[18];
cz q[55],q[80];
cz q[25],q[50];
cz q[25],q[47];
cz q[25],q[9];
cz q[50],q[13];
cz q[50],q[64];
cz q[1],q[58];
cz q[1],q[35];
cz q[1],q[5];
cz q[58],q[71];
cz q[58],q[15];
cz q[0],q[60];
cz q[0],q[40];
cz q[0],q[6];
cz q[60],q[17];
cz q[60],q[18];
cz q[36],q[89];
cz q[36],q[23];
cz q[36],q[34];
cz q[89],q[93];
cz q[89],q[69];
cz q[10],q[70];
cz q[10],q[94];
cz q[10],q[62];
cz q[70],q[43];
cz q[70],q[9];
cz q[77],q[90];
cz q[77],q[59];
cz q[77],q[95];
cz q[90],q[52];
cz q[90],q[34];
cz q[11],q[78];
cz q[11],q[8];
cz q[11],q[96];
cz q[78],q[24];
cz q[78],q[57];
cz q[2],q[75];
cz q[2],q[45];
cz q[2],q[94];
cz q[75],q[73];
cz q[75],q[44];
cz q[37],q[42];
cz q[37],q[83];
cz q[37],q[84];
cz q[42],q[33];
cz q[66],q[87];
cz q[66],q[53];
cz q[66],q[44];
cz q[87],q[91];
cz q[87],q[22];
cz q[80],q[4];
cz q[13],q[26];
cz q[13],q[93];
cz q[26],q[16];
cz q[26],q[48];
cz q[79],q[99];
cz q[79],q[92];
cz q[79],q[98];
cz q[99],q[74];
cz q[99],q[32];
cz q[16],q[95];
cz q[16],q[12];
cz q[95],q[29];
cz q[17],q[43];
cz q[17],q[8];
cz q[22],q[74];
cz q[22],q[65];
cz q[74],q[69];
cz q[45],q[76];
cz q[45],q[54];
cz q[91],q[81];
cz q[91],q[71];
cz q[8],q[56];
cz q[14],q[24];
cz q[14],q[46];
cz q[14],q[4];
cz q[24],q[31];
cz q[23],q[3];
cz q[23],q[48];
cz q[52],q[43];
cz q[52],q[65];
cz q[30],q[3];
cz q[30],q[38];
cz q[47],q[57];
cz q[47],q[65];
cz q[57],q[61];
cz q[92],q[31];
cz q[92],q[67];
cz q[35],q[33];
cz q[35],q[59];
cz q[82],q[88];
cz q[82],q[9];
cz q[82],q[63];
cz q[88],q[40];
cz q[88],q[96];
cz q[28],q[71];
cz q[28],q[46];
cz q[28],q[84];
cz q[83],q[12];
cz q[83],q[86];
cz q[61],q[5];
cz q[61],q[54];
cz q[93],q[54];
cz q[27],q[29];
cz q[27],q[3];
cz q[27],q[20];
cz q[29],q[38];
cz q[18],q[63];
cz q[21],q[68];
cz q[21],q[56];
cz q[21],q[4];
cz q[68],q[32];
cz q[68],q[85];
cz q[53],q[84];
cz q[53],q[46];
cz q[56],q[62];
cz q[33],q[44];
cz q[81],q[86];
cz q[81],q[32];
cz q[69],q[39];
cz q[38],q[51];
cz q[48],q[12];
cz q[96],q[20];
cz q[73],q[98];
cz q[73],q[63];
cz q[31],q[34];
cz q[5],q[59];
cz q[86],q[15];
cz q[20],q[67];
cz q[67],q[6];
cz q[40],q[51];
cz q[51],q[85];
cz q[94],q[64];
cz q[97],q[64];
cz q[97],q[15];
cz q[6],q[85];
