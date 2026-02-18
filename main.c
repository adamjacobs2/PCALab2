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
        printf ("\n");
        printf ("%d X %d data array:", N, N);
        printf ("\n");       
        uint8_t (*data)[N] = (uint8_t (*)[N]) A;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf ("%d\t", data [i][j]);
            }
            printf ("\n");
        }
        printf ("\n");
    }

    uint8_t* local_buf = malloc(sendcounts[rank] * sizeof(uint8_t));
    
    // Distribute data from A (on rank 0) to all local_bufs
    
    MPI_Scatterv(A, sendcounts, inputDispls, MPI_UINT8_T, local_buf, sendcounts[rank], MPI_UINT8_T, 0, MPI_COMM_WORLD);


    printf ("\n");
    printf ("rank %d recv buff:\n", rank);

    uint8_t (*data)[N] = (uint8_t (*)[N]) local_buf;
        for (int i = 0; i < sendcounts[rank] / N; i++) {
            for (int j = 0; j < N; j++) {
                printf ("%d\t", data [i][j]);
            }
            printf ("\n");
        }
    
    return local_buf;
}

uint8_t* mask_operation(uint8_t *recv_buff, int N) {
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


    printf ("\n");
    printf ("rank %d result:\n", rank);

    
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            printf ("%d\t", result [i][j]);
        }
        printf ("\n");
    }
    return ptr;
}

void collect_results(uint8_t *updated_buff, int N, uint8_t *Ap) {
    
    MPI_Gatherv(updated_buff, recvcounts[rank], MPI_UINT8_T, 
                Ap, recvcounts, outputDispls, MPI_UINT8_T, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        uint8_t (*result)[N] = (uint8_t (*)[N]) Ap;
        
        printf("Updated Data Matrix\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // Formatting with \t ensures columns stay aligned
                printf("%d\t", result[i][j]);
            }
            printf("\n");
        }
    }
}
/*************************************************** */


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <N>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

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
    collect_results(temp2, N, Ap);

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
