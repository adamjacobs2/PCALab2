#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
int rank, size;
int *sendcounts, *recvcounts;
int *inputDispls, *outputDispls;




// Timing functions
/*************************************************** */


/*   ttype: type to use for representing time */
typedef double ttype;
ttype tdiff(struct timespec a, struct timespec b)
/* Find the time difference. */
{
  ttype dt = (( b.tv_sec - a.tv_sec ) + ( b.tv_nsec - a.tv_nsec ) / 1E9);
  return dt;
}

struct timespec now()
/* Return the current time. */
{
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t;
}




struct timespec begin, end;
double time_spent;
/*************************************************** */



//Lab 2 functions
/*************************************************** */
void initialize_data(uint8_t *A, int N) {
    srand(1);
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 256;
    }
}

uint8_t* distribute_data(uint8_t *A, int N) {
    sendcounts = malloc(sizeof(int) * size);
    recvcounts = malloc(sizeof(int) * size);
    inputDispls = malloc(sizeof(int) * size);
    outputDispls = malloc(sizeof(int) * size);
    
    int rem = (N - 2) % size;
 
    int sum = 0;
    for (int i = 0; i < size; i++) {
        int outputRows = (N-2) / size + (i < rem ? 1 : 0);
        int inputRows = outputRows + 2;
        sendcounts[i] = inputRows * N;
        recvcounts[i] = outputRows * N;
        inputDispls[i] = sum;
        outputDispls[i] = sum + N;
        sum += outputRows* N;
    }

    if (rank == 0) {
        initialize_data(A, N);

#ifndef TIMING
        printf ("\n");
        printf ("%d X %d data matrix:", N, N);
        printf ("\n");       
        uint8_t (*data)[N] = (uint8_t (*)[N]) A;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf ("%d\t", data [i][j]);
            }
            printf ("\n");
        }
        printf ("\n");
        fflush(stdout);
#endif
    }

    uint8_t* local_buf = malloc(sendcounts[rank] * sizeof(uint8_t));
    
    // Distribute data from A (on rank 0) to all local_bufs
    
    MPI_Scatterv(A, sendcounts, inputDispls, MPI_UINT8_T, local_buf, sendcounts[rank], MPI_UINT8_T, 0, MPI_COMM_WORLD);


#ifndef TIMING

    if(rank == 0){
        printf ("\n");
        printf ("ScatterV send_counts[0-3] and displs[0-3]:\n");
        printf ("sendcounts[0] = %d", sendcounts[0]);
        printf ("sendcounts[1] = %d", sendcounts[1]);
        printf ("sendcounts[2] = %d", sendcounts[2]);
        printf ("sendcounts[3] = %d", sendcounts[3]);
        printf ("displs[0] = %d", inputDispls[0]);
        printf ("displs[1] = %d", inputDispls[1]);
        printf ("displs[2] = %d", inputDispls[2]);
        printf ("displs[3] = %d", inputDispls[3]);

    }
    

    printf ("\n");
    printf ("Rank %d recv buff:\n", rank);

    uint8_t (*data)[N] = (uint8_t (*)[N]) local_buf;
        for (int i = 0; i < sendcounts[rank] / N; i++) {
            for (int j = 0; j < N; j++) {
                printf ("%d\t", data [i][j]);
            }
            printf ("\n");
        }
    fflush(stdout);
#endif
    return local_buf;
}

uint8_t* mask_operation(uint8_t *recv_buff, int N) {
    //edge workers will send an extra unprocessed row
    int local_rows = (sendcounts[rank] / N) - 2;
    uint8_t (*data)[N] = (uint8_t (*)[N]) recv_buff;
    
    //alocate the buffer that we send back
    uint8_t* ptr = calloc(recvcounts[rank], sizeof(uint8_t)); 
    uint8_t (*result)[N] = (uint8_t (*)[N]) ptr;

    int sum; 
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 0; j < N; j++) {
	    // Handle vertical boundaries (left and right columns)
        if (j == 0 || j == N - 1) {
            result[i-1][j] = data[i][j]; 
            continue;
        }
            sum = data[i-1][j-1] +   data[i-1][j] +   data[i-1][j+1] +
                      data[i][j-1]  + 2*data[i][j] + data[i][j+1]   +
                      data[i+1][j-1] + data[i+1][j] + data[i+1][j+1];
            result[i-1][j] = (uint8_t)(sum / 10);
        }
    }

#ifndef TIMING
    printf ("\n");
    printf ("Updated values for rank %d\n", rank);

    
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            printf ("%d\t", result [i][j]);
        }
        printf ("\n");
    }
    printf("\n");
    fflush(stdout);
#endif
    return ptr;
}

void collect_results(uint8_t *updated_buff, int N, uint8_t *Ap, uint8_t* A) {
    
    MPI_Gatherv(updated_buff, recvcounts[rank], MPI_UINT8_T, 
                Ap, recvcounts, outputDispls, MPI_UINT8_T, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        uint8_t (*result)[N] = (uint8_t (*)[N]) Ap;
        uint8_t (*Aptr)[N] = (uint8_t (*)[N]) A;

        //copy first and last row of input matrix into Ap 
        for(int j = 0; j < N; j++){
            result[0][j] = Aptr[0][j];
            result[N-1][j] = Aptr[0][j];
        }

        end = now();
        time_spent = tdiff(begin, end);

        
#ifndef TIMING

if(rank == 0){
        printf ("\n");
        printf ("GatherV send_counts[0-3] and displs[0-3]:\n");
        printf ("sendcounts[0] = %d", recvcounts[0]);
        printf ("sendcounts[1] = %d", recvcounts[1]);
        printf ("sendcounts[2] = %d", recvcounts[2]);
        printf ("sendcounts[3] = %d", recvcounts[3]);
        printf ("displs[0] = %d", outputDispls[0]);
        printf ("displs[1] = %d", outputDispls[1]);
        printf ("displs[2] = %d", outputDispls[2]);
        printf ("displs[3] = %d", outputDispls[3]);

    }
        printf("Updated Data Matrix\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // Formatting with \t ensures columns stay aligned
                printf("%d\t", result[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        fflush(stdout);
#endif

    printf("Total time taken: %f seconds\n", time_spent);
    fflush(stdout);
    }
}
/*************************************************** */


int main(int argc, char **argv) {



    begin = now();
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = atoi(argv[1]);
    uint8_t *A = NULL;
    uint8_t *Ap = NULL;

    // ONLY Rank 0 allocates the master memory
    if (rank == 0) {
        A = malloc(N * N * sizeof(uint8_t));
        Ap = calloc(N * N, sizeof(uint8_t));
    }

    uint8_t *temp1 = distribute_data(A, N);
    uint8_t *temp2 = mask_operation(temp1, N);
    collect_results(temp2, N, Ap, A);

    // Cleanup
    if (rank == 0) {
        free(A);
        free(Ap);
    }
    free(sendcounts);
    free(recvcounts);
    free(inputDispls);
    free(outputDispls);
    free(temp1);
    free(temp2);

    MPI_Finalize();
    return 0;
}
