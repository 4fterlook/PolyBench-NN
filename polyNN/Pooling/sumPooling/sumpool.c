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
 *	FIXME : Update reference link.
 *
 *
 */
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <m5ops.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is N=1024. */
#include "sumpool.h"

#include <limits.h>

#define SUMPOOL2D_FORWARD
#define SUMPOOL2D_FORWARD_TIMER

static int sumpool2d_forward_tile_count = 0;

#define sumpool2d_forward_tile_n 22 /*max size is 150 in large*/
#define sumpool2d_forward_tile_d 14 /*max size is 120 in large*/
#define sumpool2d_forward_tile_r 3 /*max size is 9 in large*/
#define sumpool2d_forward_tile_c 3 /*max size is 9 in large*/

/* Array initialization. */
static void init_array(int nn, int nd, int ih, int iw, int oh, int ow,
											 DATA_TYPE POLYBENCH_4D(out_F, NN, ND, OH, OW, nn, nd, oh, ow),
											 DATA_TYPE POLYBENCH_4D(inp_F, NN, ND, IH, IW, nn, nd, ih, iw)
											//  DATA_TYPE POLYBENCH_4D(err_in, NN, ND, IH, IW, nn, nd, ih, iw),
											//  DATA_TYPE POLYBENCH_4D(err_out, NN, ND, OH, OW, nn, nd, oh, ow)
											 )
{
	int a, b, d, e;

	for (a = 0; a < nn; a++)
		for (b = 0; b < nd; b++)
			for (d = 0; d < oh; d++)
				for (e = 0; e < ow; e++)
				{
					out_F[a][b][d][e] = (DATA_TYPE)(a * b + d * e % nn);
					// err_out[a][b][d][e] = (DATA_TYPE)(a + b + d + e % nn);
				}

	for (a = 0; a < nn; a++)
		for (b = 0; b < nd; b++)
			for (d = 0; d < iw; d++)
				for (e = 0; e < ih; e++)
				{
					inp_F[a][b][d][e] = (DATA_TYPE)(a * b + d * e % nd);
					// err_in[a][b][d][e] = (DATA_TYPE)(a + b + d + e % nd);
				}
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array_fwd(int nn, int nd, int oh, int ow, DATA_TYPE POLYBENCH_4D(out_F, NN, ND, OH, OW, nn, nd, oh, ow))
{
	int a, b, e, d;

	for (a = 0; a < nn; a++)
		for (b = 0; b < nd; b++)
			for (e = 0; e < oh; e++)
				for (d = 0; d < ow; d++)
				{
					fprintf(stderr, DATA_PRINTF_MODIFIER, out_F[a][b][e][d]);
					if ((a * nd * oh * ow + b * oh * ow + e * ow + d) % 20 == 0)
						fprintf(stderr, "\n");
				}
	fprintf(stderr, "\n");
}

static void print_array_bwd(int nn, int nd, int ih, int iw, DATA_TYPE POLYBENCH_4D(err_in, NN, ND, IH, IW, nn, nd, ih, iw))
{
	int a, b, e, d;

	for (a = 0; a < nn; a++)
		for (b = 0; b < nd; b++)
			for (e = 0; e < ih; e++)
				for (d = 0; d < iw; d++)
				{
					fprintf(stderr, DATA_PRINTF_MODIFIER, err_in[a][b][e][d]);
					if ((a * nd * ih * iw + b * ih * iw + e * iw + d) % 20 == 0)
						fprintf(stderr, "\n");
				}
	fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,

   including the call and return. */
static void sumpool2d_forward(int nn, int nd, int ih, int iw, int ow, int oh, int dh, int dw, int sh, int sw,
															DATA_TYPE POLYBENCH_4D(inp_F, NN, ND, IH, IW, nn, nd, ih, iw),
															DATA_TYPE POLYBENCH_4D(out_F, NN, ND, OH, OW, nn, nd, oh, ow))
{

	DATA_TYPE val;
#pragma scop
#ifdef SUMPOOL2D_FORWARD_TIMER
		LKMC_M5OPS_RESETSTATS;
#endif
for (int nt = 0; nt < _PB_NN; nt+=sumpool2d_forward_tile_n)
	for (int dt = 0; dt < _PB_ND; dt+=sumpool2d_forward_tile_d)
		for (int rt = 0; rt < _PB_NR; rt+=sumpool2d_forward_tile_r)
			for (int ct = 0; ct < _PB_NC; ct+=sumpool2d_forward_tile_c){
				#ifdef SUMPOOL2D_FORWARD_TIMER
					LKMC_M5OPS_DUMPSTATS;
					if (sumpool2d_forward_tile_count++ > 4){
						printf("Tile number: %d\n", sumpool2d_forward_tile_count);
						LKMC_M5OPS_EXIT;
					}
					LKMC_M5OPS_RESETSTATS;
				#endif
				for (int n = nt; n < MIN(_PB_NN, nt+sumpool2d_forward_tile_n); n++)
					for (int d = dt; d < MIN(_PB_ND,dt+sumpool2d_forward_tile_d); d++)
						for (int r = rt; r < MIN(_PB_NR,rt+sumpool2d_forward_tile_r); r++)
						{
							for (int c = ct; c < MIN(_PB_NC,ct+sumpool2d_forward_tile_c); c++)
								{
									DATA_TYPE val = 0;
									for (int h = sh * r; h < MIN(sh * r + dh, ih); h++)
										for (int w = sw * c; w < MIN(sw * c + dw, iw); w++)
											val += inp_F[n][d][h][w];
									out_F[n][d][r][c] = val;
								}
							}
	}

#pragma endscop
}

static void sumpool2d_backward(int nn, int nd, int ih, int iw, int ow, int oh, int dh, int dw, int sh, int sw,
															 DATA_TYPE POLYBENCH_4D(inp_F, NN, ND, IH, IW, nn, nd, ih, iw),
															 DATA_TYPE POLYBENCH_4D(out_F, NN, ND, OH, OW, nn, nd, oh, ow),
															 DATA_TYPE POLYBENCH_4D(err_in, NN, ND, IH, IW, nn, nd, ih, iw),
															 DATA_TYPE POLYBENCH_4D(err_out, NN, ND, OH, OW, nn, nd, oh, ow))
{

#pragma scop

	for (int n = 0; n < _PB_NN; n++)
		for (int d = 0; d < _PB_ND; d++)
			for (int r = 0; r < _PB_NR; r++)
			{
				for (int c = 0; c < _PB_NC; c++)
				{
					for (int h = sh * r; h < MIN(sh * r + dh, ih); h++)
						for (int w = sw * c; w < MIN(sw * c + dw, iw); w++)
							err_in[n][d][h][w] += err_out[n][d][r][c];
				}
			}

#pragma endscop
}

int main(int argc, char **argv)
{
	/* Retrieve problem size. 
	   inp - 4d Input matrix nn x nd x ih x iw
	   (dh,dw) - pool size
	   (sh,sw) - stride values
	   out - 4d output matrix nn x nd x oh x ow
	 */
	int nn = NN;
	int nd = ND;
	int ih = IH;
	int iw = IW;
	int dh = DH;
	int dw = DW;
	int sh = SH;
	int sw = SW;
	int oh = OH;
	int ow = OW;

	/* Variable declaration/allocation. */
	POLYBENCH_4D_ARRAY_DECL(inp_F, DATA_TYPE, NN, ND, IH, IW, nn, nd, ih, iw);
	POLYBENCH_4D_ARRAY_DECL(out_F, DATA_TYPE, NN, ND, OH, OW, nn, nd, oh, ow);
	POLYBENCH_4D_ARRAY_DECL(err_in, DATA_TYPE, NN, ND, IH, IW, nn, nd, ih, iw);
	POLYBENCH_4D_ARRAY_DECL(err_out, DATA_TYPE, NN, ND, OH, OW, nn, nd, oh, ow);

	/* Initialize array(s). */
	init_array(nn, nd, ih, iw, oh, ow,
						 POLYBENCH_ARRAY(out_F),
						 POLYBENCH_ARRAY(inp_F)
						//  POLYBENCH_ARRAY(err_in),
						//  POLYBENCH_ARRAY(err_out)
						);

	/* Run kernel. */

#if defined(SUMPOOL2D_FORWARD)
	sumpool2d_forward(nn, nd, ih, iw, oh, ow, dh, dw, sh, sw,
										POLYBENCH_ARRAY(inp_F),
										POLYBENCH_ARRAY(out_F));
#endif

#if defined(SUMPOOL2D_BACKWARD)
	sumpool2d_backward(nn, nd, ih, iw, oh, ow, dh, dw, sh, sw,
										 POLYBENCH_ARRAY(inp_F),
										 POLYBENCH_ARRAY(out_F),
										 POLYBENCH_ARRAY(err_in),
										 POLYBENCH_ARRAY(err_out));
#endif

	/* Prevent dead-code elimination. All live-out data must be printed
	   by the function call in argument. */
	polybench_prevent_dce(print_array_fwd(nn, nd, ow, oh, POLYBENCH_ARRAY(out_F)));
	// polybench_prevent_dce(print_array_bwd(nn, nd, iw, ih, POLYBENCH_ARRAY(err_in)));

	/* Be clean. */
	POLYBENCH_FREE_ARRAY(out_F);
	POLYBENCH_FREE_ARRAY(inp_F);
	// POLYBENCH_FREE_ARRAY(err_in);
	// POLYBENCH_FREE_ARRAY(err_out);

	return 0;
}
