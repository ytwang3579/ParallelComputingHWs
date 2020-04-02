#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;

	int myrank, total;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int mystart = (r / total) * myrank;
	int mywork = (myrank == (total-1)) ? r : mystart + (r/total);

	for (unsigned long long x = myrank; x < r; x+=total) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
		if(pixels >= k) pixels %= k;
	}

	unsigned long long pixelsfromprev;
	if(myrank == 0) {
		if(total != 1) MPI_Send(&pixels, 1, MPI_UNSIGNED_LONG_LONG, myrank+1, 0, MPI_COMM_WORLD);
	} else {
		MPI_Recv(&pixelsfromprev, 1, MPI_UNSIGNED_LONG_LONG, myrank-1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		pixels += pixelsfromprev;
		if(pixels >= k) pixels %= k;
		if(myrank != (total-1)) MPI_Send(&pixels, 1, MPI_UNSIGNED_LONG_LONG, myrank+1, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();

	if(myrank == (total-1)) printf("%llu\n", (4 * pixels) % k);
}
