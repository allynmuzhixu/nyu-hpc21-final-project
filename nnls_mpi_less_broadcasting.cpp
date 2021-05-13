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
  double x_old;
  double delta_x;
  double* mu = (double*) malloc(N * sizeof(double));

  //Initialize x and mu
  for (long i = 0; i < N; i++) {
    x[i] = 0.0;
    mu[i] = c[i];
  }

  //Main loop
  for (long iter = 0; iter < max_iters; iter++) {
    for (long j = 0; j < N; j++) {
      x_old = x[j];
      x[j] = std::max(0.0, x_old - mu[j] * inv_Qdiag[j]);
      delta_x = x[j] - x_old;

      //Don't update if x[j] didn't change
      if ((delta_x > 1e-10) | (delta_x < -1e-10)) {
        for (long i = 0; i < N; i++) {
          mu[i] = mu[i] + (x[j] - x_old)*Q[i+N*j];
        }
      }
    }
  }

  free(mu);
}


void nnls_mpi_less_broadcasting(double* x, const double* Q, const double* inv_Qdiag, const double* c, long N, long max_iters, MPI_Comm comm, int mpirank, int mpisize, double& tt) {
  long N_proc = long(N / mpisize);

  //Node 0 scatters data to nodes
  double* Q_proc = (double*) malloc(N_proc * N * sizeof(double));
  double* inv_Qdiag_proc = (double*) malloc(N_proc * sizeof(double));
  double* c_proc = (double*) malloc(N_proc * sizeof(double));

  MPI_Scatter(Q, N_proc*N, MPI_DOUBLE, Q_proc, N_proc*N, MPI_DOUBLE, 0, comm);
  MPI_Scatter(inv_Qdiag, N_proc, MPI_DOUBLE, inv_Qdiag_proc, N_proc, MPI_DOUBLE, 0, comm);
  MPI_Scatter(c, N_proc, MPI_DOUBLE, c_proc, N_proc, MPI_DOUBLE, 0, comm);
  MPI_Barrier(comm);
  tt = MPI_Wtime();

  //Initialize x and mu
  double* x_proc = (double*) calloc(N_proc, sizeof(double));
  double x_old;
  double* mu_proc = c_proc;

  //Main loop
  double* delta_x = (double*) malloc(N_proc * sizeof(double));
  long j;
  for (long iters = 0; iters < max_iters; iters++) {
    for (int root = 0; root < mpisize; root++) {
      //Root indicates which processor is responsible for broadcasting delta_x
      if (mpirank == root) {
        for (j = 0; j < N_proc; j++) {
          //Update x locally
          x_old = x_proc[j];
          x_proc[j] = std::max(0.0, x_old - mu_proc[j] * inv_Qdiag_proc[j]);
          delta_x[j] = x_proc[j] - x_old;

          //If x[j] changed, update mu
          if ((delta_x[j] > 1e-10) | (delta_x[j] < -1e-10)) {
            for (long i = 0; i < N_proc; i++) {
              mu_proc[i] = mu_proc[i] + delta_x[j] * Q_proc[N*i+N_proc*root+j]; //Q is symmetric
            }
          }
        }
        //Broadcast history of delta_x
        MPI_Bcast(delta_x, N_proc, MPI_DOUBLE, root, comm);
      
      } else {

        //Recieve history of delta_x
        MPI_Bcast(delta_x, N_proc, MPI_DOUBLE, root, comm);
        
        //Update mu accordingly
        for (long j = 0; j < N_proc; j++) {
          if ((delta_x[j] > 1e-10) | (delta_x[j] < -1e-10)) {
            for (long i = 0; i < N_proc; i++) {
              mu_proc[i] = mu_proc[i] + delta_x[j] * Q_proc[N*i+N_proc*root+j]; //Q is symmetric
            }
          }
        }

      }
    }
  }

  //Send results to node 0
  MPI_Gather(x_proc, N_proc, MPI_DOUBLE, x, N_proc, MPI_DOUBLE, 0, comm);

  tt = MPI_Wtime() - tt;

  free(Q_proc);
  free(inv_Qdiag_proc);
  free(c_proc);
  free(delta_x);

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

  long N = 10000;
  long max_iters = 100;

  //Initialize
  double* A = NULL;
  double* y = NULL;

  double* Q = NULL;
  double* inv_Qdiag = NULL;
  double* c = NULL;
  double* x_seq = NULL;
  double* x_mpi = NULL;

  if (mpirank == 0) printf("nnls_mpi_less_broadcasting\n");
  if (mpirank == 0) printf("N = %ld, max_iters = %ld\n", N, max_iters);


  if (mpirank == 0) {
    //Node 0 computes random A, y and Q, c
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

    //Precompute 1/Q_jj
    inv_Qdiag = (double*) malloc(N * sizeof(double));
    for (long i = 0; i < N; i++) inv_Qdiag[i] = 1/Q[i + N*i];
  }

  double tt = 0.0;
  if (mpirank == 0) {
    x_seq = (double*) malloc(N * sizeof(double));
    x_mpi = (double*) malloc(N * sizeof(double));

    //Compute NNLS sequentially
    tt = MPI_Wtime();
    nnls_seq(x_seq, Q, inv_Qdiag, c, N, max_iters);
    tt = MPI_Wtime() - tt;
    printf("sequential-nnls = %fs\n", tt);
    printf("flop-rate = %fGb/s\n", (double)max_iters*(3*N+2*N*N) / tt / 1e9);

  }

  //Compute NNLS with MPI + less broadcasting
  nnls_mpi_less_broadcasting(x_mpi, Q, inv_Qdiag, c, N, max_iters, comm, mpirank, mpisize, tt);
  if (mpirank == 0) printf("mpi-less-broadcasting nnls = %fs\n", tt);
  if (mpirank == 0) printf("flop-rate = %fGb/s\n", (double)max_iters*(3*N+2*N*N) / tt / 1e9);

  if (mpirank == 0) {
    double err = 0.0;
    for (long i = 0; i < N; i++) err = std::max(err, std::abs(x_seq[i] - x_mpi[i]));
    printf("error = %f\n", err);

    printf("objective_seq = %f\n", evaluate(A,x_seq,y,N));
    printf("objective_mpi = %f\n", evaluate(A,x_mpi,y,N));


    printf("seq\tmpi\n");
    for (long i = 0; i < 100; i++) printf("%f\t%f\n", x_seq[i], x_mpi[i]);

  }

  MPI_Finalize();

  return 0;
}