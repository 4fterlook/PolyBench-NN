/**
 * This version is stamped on Apr. 14, 2015
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 *
 *
 *
 *
 *
 *	Reference : cuDNN paper.
 *
 *
 */
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <prem.h>
#include <m5ops.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is N=1024. */
#include "cnn.h"

#define CNN_FORWARD
#define CNN_FORWARD_TIMER

static int cnn_forward_tile_count = 0;
static int cnn_backward_tile_count = 0;

#define cnn_backward_tile_n 10 /*max size is 50 in large*/
#define cnn_backward_tile_c 25 /*max size is 75 in large*/
#define cnn_backward_tile_h 25 /*max size is 50 in large*/
#define cnn_backward_tile_w 25 /*max size is 50 in large*/
#define cnn_backward_tile_k 20 /*max size is 40 in large*/
#define cnn_backward_tile_r 1  /*max size is 6 in large*/
#define cnn_backward_tile_s 3  /*max size is 6 in large*/
#define cnn_backward_tile_p 6  /*max size is 9 in large*/
#define cnn_backward_tile_q 6  /*max size is 9 in large*/

#define cnn_forward_tile_n 25 /*max size is 50 in large, 10 in small, 5 in mini*/
#define cnn_forward_tile_k 10 /*max size is 40 in large, 15 in small, 7 in mini*/
#define cnn_forward_tile_p 9 /*max size is 9 in large, 6 in small, 4 in mini*/
#define cnn_forward_tile_q 9 /*max size is 9 in large, 6 in small, 4 in mini*/
#define cnn_forward_tile_c 15 /*max size is 75 in large, 20 in small, 10 in mini*/
#define cnn_forward_tile_r 6 /*max size is 6 in large, 4 in small, 3 in mini*/
#define cnn_forward_tile_s 6 /*max size is 6 in large, 4 in small, 3 in mini*/


/* Array initialization. */
static void
init_array(int nn, int nk, int np, int nq, int nc, int nr, int ns, int nw,
           int nh,
           DATA_TYPE POLYBENCH_4D(out_F, NN, NK, NP, NQ, nn, nk, np, nq),
           DATA_TYPE POLYBENCH_4D(W, NK, NC, NR, NS, nk, nc, nr, ns),
           DATA_TYPE POLYBENCH_4D(inp_F, NN, NC, NH, NW, nn, nc, nh, nw),
           DATA_TYPE POLYBENCH_4D(err_in, NN, NC, NH, NW, nn, nc, nh, nw),
           DATA_TYPE POLYBENCH_4D(err_out, NN, NK, NP, NQ, nn, nk, np, nq)) {
  int a, b, e, d;

  for (a = 0; a < nn; a++)
    for (b = 0; b < nk; b++)
      for (e = 0; e < np; e++)
        for (d = 0; d < nq; d++) {
          out_F[a][b][e][d] = (DATA_TYPE)((a * b) % nn);
          err_out[a][b][e][d] = (DATA_TYPE)((e * d) % nk);
        }

  for (a = 0; a < nk; a++)
    for (b = 0; b < nc; b++)
      for (e = 0; e < nr; e++)
        for (d = 0; d < ns; d++)
          W[a][b][e][d] = (DATA_TYPE)((a * b) % nc) / (10 * nc);

  for (a = 0; a < nn; a++)
    for (b = 0; b < nc; b++)
      for (e = 0; e < nh; e++)
        for (d = 0; d < nw; d++) {
          inp_F[a][b][e][d] = (DATA_TYPE)((a * b) % nc);
          err_in[a][b][e][d] = (DATA_TYPE)((a * b) % nc);
        }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nn, int nk, int np, int nq,
                        DATA_TYPE POLYBENCH_4D(out_F, NN, NK, NP, NQ, nn, nk,
                                               np, nq)) {
  int a, b, e, d;
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("out_F");
  for (a = 0; a < nn; a++)
    for (b = 0; b < nk; b++)
      for (e = 0; e < np; e++)
        for (d = 0; d < nq; d++) {
          fprintf(stderr, DATA_PRINTF_MODIFIER, out_F[a][b][e][d]);
          if ((a * nk * np * nq + b * np * nq + e * nq + d) % 20 == 0)
            fprintf(stderr, "\n");
        }
  POLYBENCH_DUMP_END("out_F");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
#if defined(CNN_ALL) || defined(CNN_FORWARD) 
void cnn_forward(int nn, int nk, int np, int nq, int nc, int nr, int ns, int nw,
            int nh, int u, int v,
            DATA_TYPE POLYBENCH_4D(out_F, NN, NK, NP, NQ, nn, nk, np, nq),
            DATA_TYPE POLYBENCH_4D(W, NK, NC, NR, NS, nk, nc, nr, ns),
            DATA_TYPE POLYBENCH_4D(inp_F, NN, NC, NH, NW, nn, nc, nh, nw)) {

#pragma scop

#ifdef CNN_FORWARD_TIMER
    LKMC_M5OPS_RESETSTATS;
#endif
  for (int nt = 0; nt < _PB_NN; nt += cnn_forward_tile_n)
    for (int kt = 0; kt < _PB_NK; kt += cnn_forward_tile_k)
      for (int pt = 0; pt < _PB_NP; pt += cnn_forward_tile_p)
        for (int qt = 0; qt < _PB_NQ; qt += cnn_forward_tile_q){
          for (int ct = 0; ct < _PB_NC; ct += cnn_forward_tile_c)
            for (int rt = 0; rt < _PB_NR; rt += cnn_forward_tile_r)
              for (int st = 0; st < _PB_NS; st += cnn_forward_tile_s){
          #ifdef CNN_FORWARD_TIMER
                  LKMC_M5OPS_DUMPSTATS;
                  // Load data here
                  if(cnn_forward_tile_count++ > 4){
                    LKMC_M5OPS_EXIT;
                  }
                  LKMC_M5OPS_RESETSTATS;
          #endif
                  for (int n = nt; n < MIN(_PB_NN, nt + cnn_forward_tile_n); n++)
                    for (int k = kt; k < MIN(_PB_NK, kt + cnn_forward_tile_k); k++)
                      for (int p = pt; p < MIN(_PB_NP, pt + cnn_forward_tile_p); p++)
                        for (int q = qt; q < MIN(_PB_NQ, qt + cnn_forward_tile_q); q++)
                          // for (int c = 0; c < _PB_NC; c++)
                          //   for (int r = 0; r < _PB_NR; r++)
                          //     for (int s = 0; s < _PB_NS; s++){
                          for (int c = ct; c < MIN(_PB_NC, ct + cnn_forward_tile_c); c++)
                            for (int r = rt; r < MIN(_PB_NR, rt + cnn_forward_tile_r); r++)
                              for (int s = st; s < MIN(_PB_NS, st + cnn_forward_tile_s); s++) {
                      out_F[n][k][p][q] += W[k][c][r][s] * inp_F[n][c][NU * p + NR - r - 1][NU * q + NS - s - 1];
                    }
          // #ifdef CNN_FORWARD_TIMER
          //         LKMC_M5OPS_RESETSTATS;
          // #endif
              }
        }
        #ifdef CNN_FORWARD_TIMER
          LKMC_M5OPS_DUMPSTATS;
        #endif
#pragma endscop
}
#endif

#if defined(CNN_ALL) || defined(CNN_BACKWARD)
static void cnn_backward(int nn, int nk, int np, int nq, int nc, int nr, int ns,
                  int nw, int nh, int u, int v,
                  DATA_TYPE POLYBENCH_4D(err_out, NN, NK, NP, NQ, nn, nk, np,
                                         nq),
                  DATA_TYPE POLYBENCH_4D(W, NK, NC, NR, NS, nk, nc, nr, ns),
                  DATA_TYPE POLYBENCH_4D(err_in, NN, NC, NH, NW, nn, nc, nh,
                                         nw)) {

#pragma scop
#ifdef CNN_BACKWARD_TIMER
    LKMC_M5OPS_RESETSTATS;
#endif
  for (int nt = 0; nt < _PB_NN; nt += cnn_backward_tile_n)
    for (int ct = 0; ct < _PB_NC; ct += cnn_backward_tile_c)
      for (int ht = 0; ht < _PB_NH; ht += cnn_backward_tile_h)
        for (int wt = 0; wt < _PB_NW; wt += cnn_backward_tile_w){

          for (int kt = 0; kt < _PB_NK; kt += cnn_backward_tile_k)
            for (int rt = 0; rt < _PB_NR; rt += cnn_backward_tile_r)
              for (int st = 0; st < _PB_NS; st += cnn_backward_tile_s)
                for (int pt = 0; pt < _PB_NP; pt += cnn_backward_tile_p)
                  for (int qt = 0; qt < _PB_NQ; qt += cnn_backward_tile_q) {
          #ifdef CNN_BACKWARD_TIMER
                  LKMC_M5OPS_DUMPSTATS;
                  // Load data here
                  if(cnn_backward_tile_count++ > 20){
                    LKMC_M5OPS_EXIT;
                  }
                  LKMC_M5OPS_RESETSTATS;
          #endif
                    for (int n = nt; n < MIN(_PB_NN, nt + cnn_backward_tile_n); n++)
                      for (int c = ct; c < MIN(_PB_NC, ct + cnn_backward_tile_c); c++)
                        for int h = ht; h < MIN(_PB_NH, ht + cnn_backward_tile_h); h++)
                          for (int w = wt; w < MIN(_PB_NW, wt + cnn_backward_tile_w); w++)
                            for (int k = kt; k < MIN(_PB_NK, kt + cnn_backward_tile_k); k++)
                              for (int r = rt; r < MIN(_PB_NR, rt + cnn_backward_tile_r); r++)
                                for (int s = st; s < MIN(_PB_NS, st + cnn_backward_tile_s); s++)
                                  for (int p = pt; p < MIN(_PB_NP, pt + cnn_backward_tile_p); p++)
                                    for (int q = qt; q < MIN(_PB_NQ, qt + cnn_backward_tile_q); q++){
                                        if ((NU * p - (h - NR + r + 1) == 0) &&
                                            (NU * q - (w - NS + s + 1) == 0)) {
                                          err_in[n][c][h][w] +=
                                              W[k][c][r][s] * err_out[n][k][p][q];
                                          kernel_count++;
                                        }
                            }
                  }
        }
    #ifdef CNN_BACKWARD_TIMER
        LKMC_M5OPS_DUMPSTATS;
    #endif
#pragma endscop
}
#endif

int main(int argc, char **argv) {
  /* Retrieve problem size.
     nn -> Batch size
     nk -> Number of output feature maps
     np -> Output matrix height
     nq -> Output matrix width
     nc -> Number of input feature maps
     nr -> Filter height
     ns -> Filter width
     nh -> Input matrix height
     nw -> Input matrix width
   */
  int nn = NN;
  int nk = NK;
  int np = NP;
  int nq = NQ;
  int nc = NC;
  int nr = NR;
  int ns = NS;
  int nw = NW;
  int nh = NH;
  int nu = NU;
  int nv = NV;

  /* Variable declaration/allocation. */
  POLYBENCH_4D_ARRAY_DECL(out_F, DATA_TYPE, NN, NK, NP, NQ, nn, nk, np, nq);
  POLYBENCH_4D_ARRAY_DECL(W, DATA_TYPE, NK, NC, NR, NS, nk, nc, nr, ns);
  POLYBENCH_4D_ARRAY_DECL(inp_F, DATA_TYPE, NN, NC, NH, NW, nn, nc, nh, nw);
  POLYBENCH_4D_ARRAY_DECL(err_in, DATA_TYPE, NN, NC, NH, NW, nn, nc, nh, nw);
  POLYBENCH_4D_ARRAY_DECL(err_out, DATA_TYPE, NN, NK, NP, NQ, nn, nk, np, nq);

  /* Initialize array(s). */
  init_array(nn, nk, np, nq, nc, nr, ns, nw, nh, POLYBENCH_ARRAY(out_F),
             POLYBENCH_ARRAY(W), POLYBENCH_ARRAY(inp_F),
             POLYBENCH_ARRAY(err_in), POLYBENCH_ARRAY(err_out));

#if defined(CNN_ALL) || defined(CNN_FORWARD)
  /* Run kernel. */
  cnn_forward(nn, nk, np, nq, nc, nr, ns, nw, nh, nu, nv,
              POLYBENCH_ARRAY(out_F), POLYBENCH_ARRAY(W),
              POLYBENCH_ARRAY(inp_F));
#endif

#if defined(CNN_ALL) || defined(CNN_BACKWARD)
  /* Run kernel. */
  cnn_backward(nn, nk, np, nq, nc, nr, ns, nw, nh, nu, nv,
               POLYBENCH_ARRAY(err_out), POLYBENCH_ARRAY(W),
               POLYBENCH_ARRAY(err_in));
#endif

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nn, nk, np, nq, POLYBENCH_ARRAY(out_F)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(out_F);
  POLYBENCH_FREE_ARRAY(W);
  POLYBENCH_FREE_ARRAY(inp_F);
  POLYBENCH_FREE_ARRAY(err_out);
  POLYBENCH_FREE_ARRAY(err_in);

  return 0;
}
