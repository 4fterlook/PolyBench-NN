/**
 * This version is stamped on Apr. 14, 2015
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#ifndef _LSTM_H
# define _LSTM_H

/* Default to LARGE_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define LARGE_DATASET
# endif

# if !defined(NT) && !defined(NP) && !defined(NQ) && !defined(NS)
/* Define sample dataset sizes. */
#  ifdef MINI_DATASET
#   define NT 2
#   define NP 3
#   define NQ 4
#   define NS 5
#  endif 

#  ifdef SMALL_DATASET

#  endif 

#  ifdef MEDIUM_DATASET

#  endif 

#  ifdef LARGE_DATASET

#  endif 

#  ifdef EXTRALARGE_DATASET

#  endif 


#endif /* !(NI NJ NK) */
# define _PB_T POLYBENCH_LOOP_BOUND(NT,nt)
# define _PB_P POLYBENCH_LOOP_BOUND(NP,np)
# define _PB_Q POLYBENCH_LOOP_BOUND(NQ,nq)
# define _PB_S POLYBENCH_LOOP_BOUND(NS,ns)


/* Default data type */
# if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) && !defined(DATA_TYPE_IS_DOUBLE)
#  define DATA_TYPE_IS_DOUBLE
# endif

#ifdef DATA_TYPE_IS_INT
#  define DATA_TYPE int
#  define DATA_PRINTF_MODIFIER "%d "
#endif 

#ifdef DATA_TYPE_IS_FLOAT
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2f "
#  define SCALAR_VAL(x) x##f
#  define SQRT_FUN(x) sqrtf(x)
#  define EXP_FUN(x) expf(x)
#  define POW_FUN(x,y) powf(x,y)
# endif

#ifdef DATA_TYPE_IS_DOUBLE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)
# endif

#endif /* !_LSTM_H */

