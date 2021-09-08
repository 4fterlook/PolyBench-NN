#define LOAD_CNN_FORWARD(CS,RS,SS,N,K,P,Q,CT,RT,ST)\
  DATA_TYPE sum = 0;\
  for (int i = CS; i < CS+CT; ++i){\
    for (int j = RS; j < RS+RT; ++j){\
      for (int k = SS; k < SS+ST; ++k){\
        sum += W[K][i][j][k];\
        sum += inp_F[N][i][NU * P + NR - j - 1][NU * Q + NS - k - 1];\
      }\
    }\
  }\
  FILE* nf = fopen("/dev/null","w");\
  fprintf(nf,"%lf",sum);\
  
