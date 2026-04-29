#include <assert.h>
#include <stdio.h>

/*
These unit tests aim at checking the correct implementation of block matrices,
specifically for binary sparse matrices.
*/

void print_matrix(float **M, int rows, int cols) {

    for(int i = 0; i < rows; i++) {

        for(int j = 0; j < cols; j++) {
            printf("%2.0f ", M[i][j]);
        }

        printf("\n");
    }
}


void main() {

        int row_dimension = 20;
        int column_dimension = 20;
        int n_connections = 4;
        int n_blocks = 5;

        float **M = binary_sparse(row_dimension, column_dimension, n_connections, 42, n_blocks);

        int block_size = column_dimension/n_blocks;
	for(int i=0; i<row_dimension; i++){
		int block_id = i/block_size;
		int start = block_id*block_size;
		int nnz = 0; //non zero elements, used to check the number of connections is respected

		for(int j=0; j<column_dimension; j++){
			assert(M[i][j]==0 || M[i][j]==1 || M[i][j]==-1);
			if(M[i][j]!=0){
				nnz++;
				assert(j>=start);
				assert(j<start + block_size);	
			}
		}
		assert(nnz<=n_connections);
	}
	printf("PASS\n");
	printf("MATRIX:\n");
    print_matrix(M, rows, cols);
}
