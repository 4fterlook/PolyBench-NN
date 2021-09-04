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
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define LKMC_M5OPS_ENABLE 0
#include <m5ops.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is N=1024. */
#include "cnn.h"

/* Array initialization. */
static void init_array(int nn, int nk, int np, int nq, int nc, int nr, int ns, int nw, int nh,
											 DATA_TYPE POLYBENCH_4D(out_F, NN, NK, NP, NQ, nn, nk, np, nq),
											 DATA_TYPE POLYBENCH_4D(W, NK, NC, NR, NS, nk, nc, nr, ns),
											 DATA_TYPE POLYBENCH_4D(inp_F, NN, NC, NH, NW, nn, nc, nh, nw),
											 DATA_TYPE POLYBENCH_4D(err_in, NN, NC, NH, NW, nn, nc, nh, nw),
											 DATA_TYPE POLYBENCH_4D(err_out, NN, NK, NP, NQ, nn, nk, np, nq))
{
	int a, b, e, d;

	for (a = 0; a < nn; a++)
		for (b = 0; b < nk; b++)
			for (e = 0; e < np; e++)
				for (d = 0; d < nq; d++)
				{
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
				for (d = 0; d < nw; d++)
				{
					inp_F[a][b][e][d] = (DATA_TYPE)((a * b) % nc);
					err_in[a][b][e][d] = (DATA_TYPE)((a * b) % nc);
				}
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nn, int nk, int np, int nq, DATA_TYPE POLYBENCH_4D(out_F, NN, NK, NP, NQ, nn, nk, np, nq))
{
	int a, b, e, d;
	POLYBENCH_DUMP_START;
	POLYBENCH_DUMP_BEGIN("out_F");
	for (a = 0; a < nn; a++)
		for (b = 0; b < nk; b++)
			for (e = 0; e < np; e++)
				for (d = 0; d < nq; d++)
				{
					fprintf(stderr, DATA_PRINTF_MODIFIER, out_F[a][b][e][d]);
					if ((a * nk * np * nq + b * np * nq + e * nq + d) % 20 == 0)
						fprintf(stderr, "\n");
				}
	POLYBENCH_DUMP_END("out_F");
	POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void cnn_forward(int nn, int nk, int np, int nq, int nc, int nr, int ns, int nw, int nh, int u, int v,
												DATA_TYPE POLYBENCH_4D(out_F, NN, NK, NP, NQ, nn, nk, np, nq),
												DATA_TYPE POLYBENCH_4D(W, NK, NC, NR, NS, nk, nc, nr, ns),
												DATA_TYPE POLYBENCH_4D(inp_F, NN, NC, NH, NW, nn, nc, nh, nw))
{
#pragma scop
	for (int n = 0; n < _PB_NN; n++)
		for (int k = 0; k < _PB_NK; k++)
			for (int p = 0; p < _PB_NP; p++)
				for (int q = 0; q < _PB_NQ; q++)
					for (int c = 0; c < _PB_NC; c++)
						for (int r = 0; r < _PB_NR; r++)
							for (int s = 0; s < _PB_NS; s++)
								out_F[n][k][p][q] += W[k][c][r][s] * inp_F[n][c][NU * p + NR - r - 1][NU * q + NS - s - 1];
#pragma endscop
}

void cnn_backward(int nn, int nk, int np, int nq, int nc, int nr, int ns, int nw, int nh, int u, int v,
									DATA_TYPE POLYBENCH_4D(err_out, NN, NK, NP, NQ, nn, nk, np, nq),
									DATA_TYPE POLYBENCH_4D(W, NK, NC, NR, NS, nk, nc, nr, ns),
									DATA_TYPE POLYBENCH_4D(err_in, NN, NC, NH, NW, nn, nc, nh, nw))
{
#pragma scop
	for (int n = 0; n < _PB_NN; n++)
		for (int c = 0; c < _PB_NC; c++)
			for (int h = 0; h < _PB_NH; h++)
				for (int w = 0; w < _PB_NW; w++)
					for (int k = 0; k < _PB_NK; k++)
						for (int r = 0; r < _PB_NR; r++)
							for (int s = 0; s < _PB_NS; s++)
								for (int p = 0; p < _PB_NP; p++)
									for (int q = 0; q < _PB_NQ; q++)
										if ((NU * p - (h - NR + r + 1) == 0) && (NU * q - (w - NS + s + 1) == 0))
											err_in[n][c][h][w] += W[k][c][r][s] * err_out[n][k][p][q];
#pragma endscop
}

int main(int argc, char **argv)
{
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

	puts("start running cnn\n");
	
	/* Variable declaration/allocation. */
	POLYBENCH_4D_ARRAY_DECL(out_F, DATA_TYPE, NN, NK, NP, NQ, nn, nk, np, nq);
	POLYBENCH_4D_ARRAY_DECL(W, DATA_TYPE, NK, NC, NR, NS, nk, nc, nr, ns);
	POLYBENCH_4D_ARRAY_DECL(inp_F, DATA_TYPE, NN, NC, NH, NW, nn, nc, nh, nw);
	POLYBENCH_4D_ARRAY_DECL(err_in, DATA_TYPE, NN, NC, NH, NW, nn, nc, nh, nw);
	POLYBENCH_4D_ARRAY_DECL(err_out, DATA_TYPE, NN, NK, NP, NQ, nn, nk, np, nq);

	/* Initialize array(s). */
	init_array(nn, nk, np, nq, nc, nr, ns, nw, nh,
						 POLYBENCH_ARRAY(out_F),
						 POLYBENCH_ARRAY(W),
						 POLYBENCH_ARRAY(inp_F),
						 POLYBENCH_ARRAY(err_in),
						 POLYBENCH_ARRAY(err_out));

	/* Start timer. */
	polybench_start_instruments;

#if(LKMC_M5OPS_ENABLE)
  LKMC_M5OPS_RESETSTATS;
#endif
	/* Run kernel. */
	cnn_forward(nn, nk, np, nq, nc, nr, ns, nw, nh, nu, nv,
							POLYBENCH_ARRAY(out_F),
							POLYBENCH_ARRAY(W),
							POLYBENCH_ARRAY(inp_F));

	cnn_backward(nn, nk, np, nq, nc, nr, ns, nw, nh, nu, nv,
							 POLYBENCH_ARRAY(err_out),
							 POLYBENCH_ARRAY(W),
							 POLYBENCH_ARRAY(err_in));

#if(LKMC_M5OPS_ENABLE)
  LKMC_M5OPS_DUMPSTATS;
#endif

	/* Stop and print timer. */
	polybench_stop_instruments;
	polybench_print_instruments;

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
