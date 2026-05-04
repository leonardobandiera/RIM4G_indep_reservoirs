#include <assert.h>
#include <stdio.h>

#include "../lib.h"

/*
These unit tests aim at checking the correct implementation of block matrices,
specifically for binary sparse matrices.
*/


void main() {

        int row_dimension = 20;
        int n_connections = 3;
        int n_blocks = 4;

        float **M = binary_symmetric_sparse_blocks(row_dimension, n_connections, n_blocks, random_state=42);
		
    	// print matrix
    	printf("Matrix M:\n");
    	for(int i = 0; i < row_dimension; i++){
        	for(int j = 0; j < row_dimension; j++){
            	printf("%2.0f ", M[i][j]); 
        	}
       	    printf("\n");
    	}
    	printf("\n");

        int block_size = row_dimension/n_blocks;
	for(int i=0; i<row_dimension; i++){
		int block_id = i/block_size;
		int start = block_id*block_size;
		int nnz = 0; //non zero elements, used to check the number of connections is respected

		for(int j=0; j<row_dimension; j++){
			assert(M[i][j]==M[j][i]);
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
}
