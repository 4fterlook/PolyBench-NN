/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"

#include <prem.h>
#include <m5ops.h>

#define gemm_tile_i 100 /*max size is 1000 in large*/
#define gemm_tile_k 20 /*max size is 1200 in large*/
#define gemm_tile_j 100 /*max size is 1100 in large*/

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))


static int gemm_tile_count = 0;

/* Array initialization. */
static void init_array(int ni, int nj, int nk,
                       DATA_TYPE *alpha,
                       DATA_TYPE *beta,
                       DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                       DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE)((i * j + 1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE)(i * (j + 1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE)(i * (j + 2) % nj) / nj;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj,
                        DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
    {
      if ((i * ni + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_gemm(int ni, int nj, int nk,
                        DATA_TYPE alpha,
                        DATA_TYPE beta,
                        DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                        DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                        DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj))
{

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
LKMC_M5OPS_RESETSTATS;
  for (int it = 0; it < _PB_NI; it += gemm_tile_i)
    for (int kt = 0; kt < _PB_NK; kt += gemm_tile_k)
      for (int jt = 0; jt < _PB_NJ; jt += gemm_tile_j){
        LKMC_M5OPS_DUMPSTATS;
        if(gemm_tile_count++ > 4){
          LKMC_M5OPS_EXIT;
        }
        LKMC_M5OPS_RESETSTATS;
        for (int i = it; i < MIN(_PB_NI, it + gemm_tile_i); i++)
          for (int k = kt; k < MIN(_PB_NK, kt + gemm_tile_k); k++)
            for (int j = jt; j < MIN(_PB_NJ, jt + gemm_tile_j); j++){
              C[i][j] += alpha * A[i][k] * B[k][j];
            }
      }
LKMC_M5OPS_DUMPSTATS;
#pragma endscop
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);

  /* Initialize array(s). */
  init_array(ni, nj, nk, &alpha, &beta,
             POLYBENCH_ARRAY(C),
             POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm(ni, nj, nk,
              alpha, beta,
              POLYBENCH_ARRAY(C),
              POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
