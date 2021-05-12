#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <mpi.h>


void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

double evaluate(double* A, double* x, double* y, long N) {
  double* prediction = (double*) calloc(N, sizeof(double));

  MMult0(N, 1, N, A, x, prediction);

  double err = 0.0;
  for (long i = 0; i < N; i++) err = err + std::pow(prediction[i] - y[i],2);

  free(prediction);

  return err;
}

void nnls_seq(double* x, const double* Q, const double* inv_Qdiag, const double* c, long N, long max_iters) {
  //x_new = (double*) malloc(N * sizeof(double));

  double x_old;
  double* mu = (double*) malloc(N * sizeof(double));

  for (long i = 0; i < N; i++) {
    x[i] = 0.0;
    mu[i] = c[i];
  }

  for (long iter = 0; iter < max_iters; iter++) {
    for (long j = 0; j < N; j++) {
      x_old = x[j];
      x[j] = std::max(0.0, x_old - mu[j] * inv_Qdiag[j]);
      
      for (long i = 0; i < N; i++) {
        mu[i] = mu[i] + (x[j] - x_old)*Q[i+N*j];
      }
    }

  }

  free(mu);
}

void nnls_mpi(double* x, const double* Q, const double* inv_Qdiag, const double* c, long N, long max_iters, MPI_Comm comm, int mpirank, int mpisize, double& tt) {
  
  //Compute scan with MPI, save to B1
  long N_proc = long(N / mpisize);


  double* Q_proc = (double*) malloc(N_proc * N * sizeof(double));
  double* inv_Qdiag_proc = (double*) malloc(N_proc * sizeof(double));
  double* c_proc = (double*) malloc(N_proc * sizeof(double));

  MPI_Scatter(Q, N_proc*N, MPI_DOUBLE, Q_proc, N_proc*N, MPI_DOUBLE, 0, comm);
  MPI_Scatter(inv_Qdiag, N_proc, MPI_DOUBLE, inv_Qdiag_proc, N_proc, MPI_DOUBLE, 0, comm);
  MPI_Scatter(c, N_proc, MPI_DOUBLE, c_proc, N_proc, MPI_DOUBLE, 0, comm);
  MPI_Barrier(comm);

  tt = MPI_Wtime();

  //Compute scan within each compute node
  double* x_proc = (double*) calloc(N_proc, sizeof(double));
  double x_old;
  double* mu_proc = c_proc;
  
  long block_start = N_proc * mpirank;
  long block_end = N_proc * (mpirank + 1);

  double delta_x;


  for (long iters = 0; iters < max_iters; iters++) {
    for (long j = 0; j < N; j++) {
      if ( mpirank == j / N_proc ) {
        x_old = x_proc[j % N_proc];
        x_proc[j % N_proc] = std::max(0.0, x_old - mu_proc[j % N_proc] * inv_Qdiag_proc[j % N_proc]);
        delta_x = x_proc[j % N_proc] - x_old;
      }

      MPI_Bcast(&delta_x, 1, MPI_DOUBLE, j / N_proc, comm);
      
      for (long i = 0; i < N_proc; i++) {
        mu_proc[i] = mu_proc[i] + delta_x * Q_proc[N*i+j]; //Q is symmetric
      }
    }
  }

  //Send results to node 0
  MPI_Gather(x_proc, N_proc, MPI_DOUBLE, x, N_proc, MPI_DOUBLE, 0, comm);

  tt = MPI_Wtime() - tt;

  free(Q_proc);
  free(inv_Qdiag_proc);
  free(c_proc);

  free(x_proc);

}





int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int mpirank, mpisize;
  MPI_Comm_rank(comm, &mpirank);
  MPI_Comm_size(comm, &mpisize);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, mpisize, processor_name);


  MPI_Barrier(comm);

  long N = 1000;
  long max_iters = 2000;

  //Initialize
  double* A = NULL;
  double* y = NULL;

  double* Q = NULL;
  double* inv_Qdiag = NULL;
  double* c = NULL;
  double* x_seq = NULL;
  double* x_mpi = NULL;
  


  if (mpirank == 0) {
    double* A_T = NULL;

    A = (double*) malloc(N * N * sizeof(double));
    A_T = (double*) malloc(N * N * sizeof(double));
    y = (double*) malloc(N * sizeof(double));
    for (long i = 0; i < N; i++) {
      for (long j = 0; j < N; j++) {
        A[i+N*j] = (double)rand() / RAND_MAX;
        A_T[j+N*i] = A[i+N*j];
      }
    }

    for (long i = 0; i < N; i++) y[i] = (double)rand() / RAND_MAX;

    Q = (double*) calloc(N * N, sizeof(double));
    c = (double*) calloc(N, sizeof(double));

    MMult0(N, N, N, A_T, A, Q);
    MMult0(N, 1, N, A_T, y, c);
    for (long i = 0; i < N; i++) c[i] = -c[i];

    inv_Qdiag = (double*) malloc(N * sizeof(double));
    for (long i = 0; i < N; i++) inv_Qdiag[i] = 1/Q[i + N*i];

  }

  double tt = 0.0;
  if (mpirank == 0) {
    x_seq = (double*) malloc(N * sizeof(double));
    x_mpi = (double*) malloc(N * sizeof(double));

    tt = MPI_Wtime();
    nnls_seq(x_seq, Q, inv_Qdiag, c, N, max_iters);
    printf("sequential-nnls = %fs\n", MPI_Wtime() - tt);
  }

  
  //Compute scan with MPI, save to B1
  nnls_mpi(x_mpi, Q, inv_Qdiag, c, N, max_iters, comm, mpirank, mpisize, tt);
  if (mpirank == 0) printf("mpi-nnls = %fs\n", tt);

  if (mpirank == 0) {
    double err = 0.0;
    for (long i = 0; i < N; i++) err = std::max(err, std::abs(x_seq[i] - x_mpi[i]));
    printf("error = %f\n", err);

    printf("objective_seq = %f\n", evaluate(A,x_seq,y,N));
    printf("objective_mpi = %f\n", evaluate(A,x_mpi,y,N));


    printf("Q\tc\tseq\tmpi\n");
    for (long i = 0; i < 100; i++) printf("%f\t%f\t%f\t%f\n", inv_Qdiag[i], c[i], x_seq[i], x_mpi[i]);

  }



  MPI_Finalize();

  return 0;
}