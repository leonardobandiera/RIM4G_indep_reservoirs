
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h> 
#include <time.h>
#include <omp.h>



#ifndef lib_h
#define lib_h 


int rng_int(int upper_bound);
void rng_threads_init(int nthreads, long base_seed);

// Estrae un uniforme in [0,1) dallo stato del **thread corrente**
// (usa omp_get_thread_num() dentro)
double rng_uniform01(void);

// Libera la memoria degli stati RNG
void rng_threads_free(void);

typedef struct Sparse_element{
    int dimension;
    int * indexes;
    float * values;
} Sparse_element;

// Struttura che rappresenta una matrice sparsa.
typedef struct Sparse_matrix {
    int dimension;
    Sparse_element * head;
} Sparse_matrix;


//double total_energy(Sparse_matrix * A, Sparse_matrix * W, int ** embedding, float bias_coupling, float ** bias, int a)
 

void init_generator(int,int); 
float** generate_dense_matrix(int,int);
float** binary_sparse(int ,int , int , int, int ); //added one int for the newly added n_blocks
float** uniform_sparse(int, int, int, int); 
float** binary_symmetric_sparse(int, int, int);
void print_matrix(float**, int,int);
void print_int_matrix(int **, int, int);
void free_matrix(float**, int); 
void free_int_matrix(int**, int); 
float** uniform_symmetric_sparse(int, int, int);
float** uniform_symmetric_traceless_sparse(int, int, int);
float** binary_antisymmetric_sparse(int, int, int); 
float** uniform_antisymmetric_sparse(int, int, int); 
Sparse_matrix* Dense_to_Sparse( float **, int, int); 
ssize_t getline(char **, size_t *, FILE *); 
float** read_matrix(const char* , int, int);
int** read_int_matrix(const char*, int, int);
int **  Initialize_embedding(int, int);
void print_embedding_file(char *, int **, int, int);
float** alloc_and_copy_float_matrix_from_np(float*, int, int); 
int** alloc_and_copy_int_matrix_from_np(int*, int, int);  
void free_sparse( Sparse_matrix * ); 
Sparse_matrix * read_sparse(const char*, float);
double total_energy(Sparse_matrix * , Sparse_matrix *, int **, float, float **); 
double simulate(Sparse_matrix *, Sparse_matrix *, int **, float, float **, double initial_energy, int j, int nthreads,double);
double parallel_update(Sparse_matrix *, Sparse_matrix *, int **, float, float **, double initial_energy, int j, int nthreads);

//parallel_update
double simulate_no_parallel(Sparse_matrix *, Sparse_matrix *, int **, float, float **, double initial_energy, int j);

double simulate_2hops_no_parallel(Sparse_matrix *, Sparse_matrix *, int **, float, float **, double initial_energy, int);
double simulate_2hops(Sparse_matrix *, Sparse_matrix *, int **, float, float **, double initial_energy, int,int);

float prova_bias(float **, int, int);




#endif
