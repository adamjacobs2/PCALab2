#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[])
{
    int N = atof (argv[1]);
    int rank, size;     // for storing this process' rank, and the number of processes
    int rem;            // elements remaining after division among processes
    int sum = 0; 
    int i, j; 
    int k = 0;              // Sum of counts. Used to calculate displacements
    int rec_buf[100];          // buffer where the received data should be stored

    int data [N][N];
    
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // added
    rem = (N*N)%size;

    int *sendcounts = malloc(sizeof(int)*size);
         // array describing how many elements to send to each process
    int *displs = malloc(sizeof(int)*size);
         // array describing the displacements where each segment begins


    // calculate send counts and displacements
    for (i = 0; i < size; i++) {
        sendcounts[i] = (N*N)/size;
        if (rem > 0) {
          sendcounts[i]++;
          rem--;
        }

        displs[i] = sum;
        sum += sendcounts[i];
    }

    // print calculated send counts and displacements for each process
    if (0 == rank) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                data [i][j] = k++;
            }
        }

        printf ("\n");
        printf ("%d X %d data array:", N, N);
        printf ("\n");       
 
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf ("%d\t", data [i][j]);
            }
        }
        printf ("\n");

        for (i = 0; i < size; i++) {
            printf("sendcounts[%d] = %d\tdispls[%d] = %d\n", i, sendcounts[i], i, displs[i]);
        }
    }

    // divide the data among processes as described by sendcounts and displs
    MPI_Scatterv(&data, sendcounts, displs, MPI_INT, &rec_buf, 100, MPI_INT, 0, MPI_COMM_WORLD);

    // print what each process received
    printf("%d: ", rank);
    for (i = 0; i < sendcounts[rank]; i++) {
        printf("%d\t", rec_buf[i]);
    }
    printf("\n");


for (i=0; i < sendcounts[rank]; i++) {
         rec_buf[i] = rec_buf[i] * 10;
    }
    printf("updated %d: ", rank);   
    for (i = 0; i < sendcounts[rank]; i++) {
        printf("%d\t", rec_buf[i]);        
    }
    printf("\n");
       
    
    MPI_Gatherv(&rec_buf, sendcounts[rank], MPI_INT, &data, sendcounts, displs, MPI_INT,  0, MPI_COMM_WORLD); 
         
    if (0 == rank) {
        printf ("Updated data array: \n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf ("%d\t", data [i][j]);
            }
        }
        printf ("\n");
    }   







    MPI_Finalize();
    free(sendcounts);
    free(displs);
    return 0;
}