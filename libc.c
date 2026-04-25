#include "lib.h"
#include <stdlib.h> 
#include <stdint.h> 
#include <stdio.h>
#include "mt19937-64/mt64.h"
#include <sys/types.h>
#include <string.h> 
#include <time.h>
#include <omp.h>
#include <math.h> 
#define MAX_NEIGHBORS 10000 



static struct drand48_data *g_rng = NULL;
static int g_rng_threads = 0;

void rng_threads_init(int nthreads, long base_seed) {
    // Se già inizializzato con lo stesso numero di thread, non rifare nulla
    if (g_rng && g_rng_threads == nthreads) return;

    // Se cambia nthreads, rialloca (evita leak)
    free(g_rng);
    g_rng = (struct drand48_data*)malloc((size_t)nthreads * sizeof *g_rng);
    if (!g_rng) {
        perror("malloc g_rng");
        abort();
    }
    g_rng_threads = nthreads;

    // Semina uno stato indipendente per ciascun thread
    for (int t = 0; t < nthreads; ++t) {
        long seed_t = base_seed ^ (long)(0x9E3779B9u * (unsigned)(t + 1));
        srand48_r(seed_t, &g_rng[t]);
    }
}

double rng_uniform01(void) {
    int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    double u = 0.0;
    drand48_r(&g_rng[tid], &u);   // uniforme in [0,1)
    return u;
}

void rng_threads_free(void) {
    free(g_rng);
    g_rng = NULL;
    g_rng_threads = 0;
}



//////////////da qui è come prima //////////////




// Restituisce un intero in [0, upper_bound)
int rng_int(int upper_bound) {
    double u = 0.0;
    drand48_r(&g_rng[0], &u);
    return (int)(u * upper_bound);
}




//questo era solo una prova, dopo posso toglierlo 
float** generate_dense_matrix(int row,int col){ 
    float ** M = malloc(row*sizeof(float*)); 
    
    for (int i=0; i<row; i++){ 
    M[i] = calloc(row, sizeof(float)); 
    }
    
    for (int i= 0; i<row; i++){ 
        for (int j=0; j<col; j++){ 
        
        M[i][j]= i*j+0.333; 
        }
    }
    
    return M; 
}


void init_generator(int random_state_iteration,int nthreads){

 unsigned long long init[4]={0x12345ULL, 0x23456ULL, 0x34567ULL, 0x45678ULL}, length=4;
 
 init_by_array64(init, length);  //inizializza il generatore 
 
 
 if (random_state_iteration != -1) srand(random_state_iteration);   //per le altre iterazioni 
    else srand(time(NULL));
    
 long seed =(long)random_state_iteration;     //  14.0;  // (long)time(NULL);        // o un seed fisso per riproducibilità
 rng_threads_init(nthreads, seed);
 } 

///introduzione delle matrici a blocchi, così da simulare reservoir indipendenti.
///aggiunta tra i parametri della funzione int n_blocks, che definisce il numero di blocchi della matrice.
float** binary_sparse(int row_dimension, int column_dimension, int n_connections, int random_state, int n_blocks)
{ 
    float **M = malloc(row_dimension*sizeof(float*)); 
    
    if (random_state != -1) srand(random_state);
    else srand(time(NULL));

    /*
    for (int row=0; row<row_dimension; row++){ 
         M[row] = calloc(column_dimension, sizeof(float)); 
         //M[row] = malloc(column_dimension*sizeof(float)); 
         
         // Genera n_connessioni non nulli casuali (+1,-1) per la riga corrente 
         
         for (int k=0; k<n_connections; k++)
         {
          int j = (genrand64_int64()%column_dimension); 
          M[row][j] = 2*(int)(genrand64_int64()%2) -1;
         } 
         
      }
    */

    int block_size = column_dimension / n_blocks;
    for (int row=0; row<row_dimension; row++){
        M[row] = calloc(column_dimension, sizeof(float));
        int block_id = row / block_size; //contrassegno ogni blocco con un id
        int start = block_id * block_size;
        for (int k=0; k<n_connections; k++){
            int j = start + (genrand64_int64()%column_dimension); //connessioni solo dentro il blocco
            M[row][j] = 2*(int)(genrand64_int64()%2) -1;
        }
    }
      
      return M; 
} 

float** uniform_sparse(int row_dimension, int column_dimension, int n_connections, int random_state)
{ 
    float **M= malloc(row_dimension*sizeof(float*)); 
    int j; 
    if(random_state != -1) srand(random_state); 
    else srand (time(NULL)); 
    
    for (int row=0; row <row_dimension; row++)
    {
     //alloca ogni riga della matrice e la inizializza a zero  
     M[row] = calloc(column_dimension,sizeof(float)); 
     
     for (int k=0; k<n_connections; k++)
     { 
       j= (genrand64_int64()%column_dimension); 
       M[row][j] = 2*genrand64_real2() -1;
     } 
    }
    
    return M; 
   
}


float** binary_symmetric_sparse(int row_dimension, int n_connections, int random_state)
{ 
    float **M = malloc(row_dimension*sizeof(float*)); 
    int j;
    if (random_state != -1) srand(random_state);
    else srand(time(NULL));
    
    for (int row=0; row < row_dimension; row++)
    {
     M[row] = calloc(row_dimension,sizeof(float)); 
    } 
    
    for (int row =0; row < row_dimension; row++)
    { 
     for (int k= 0; k < n_connections ; k++)
     { j = (genrand64_int64()%row_dimension); 
            M[row][j] = 2*(int)(genrand64_int64()%2) -1;
            M[j][row] = M[row][j];
        }
    }
    
    return M; 
    
}


float** uniform_symmetric_sparse(int row_dimension, int n_connections, int random_state)
{ 
    float **M = malloc(row_dimension*sizeof(float*)); 
    int j;
    if (random_state != -1) srand(random_state);
    else srand(time(NULL));
    
    for (int row =0; row < row_dimension; row++) 
    { 
     M[row] = calloc(row_dimension,sizeof(float)); 
    } 
    
    for (int row = 0; row<row_dimension; row++)
    { 
     for(int k=0; k<n_connections; k++)
     { 
      j = (genrand64_int64()%row_dimension); 
      M[row][j] = 2*genrand64_real2() -1;
      M[j][row] = M[row][j];
     } 
    }
    
    return M; 
    
} 


float** uniform_symmetric_traceless_sparse(int row_dimension, int n_connections, int random_state)
{
    float **M = malloc(row_dimension*sizeof(float*));
    int j; 
 
    if (random_state != -1) srand(random_state);
    else srand(time(NULL));
    for (int row =0; row < row_dimension; row++) 
    { 
     M[row] = calloc(row_dimension,sizeof(float)); 
    } 
    
    for (int row =0; row < row_dimension; row++)
    { 
     for (int k=0; k<n_connections; k++)
     { 
     
      j = (genrand64_int64()%row_dimension); 
      if (j == row) k--;
      else {
            M[row][j] = 2*genrand64_real2() -1;
            M[j][row] = M[row][j];
            } 
            
     }
     
    } 
    
    return M;
    
} 



     
 
float** binary_antisymmetric_sparse(int row_dimension, int n_connections, int random_state)
{ 

  float **M = malloc(row_dimension*sizeof(float*));
  int j;   
  
  if (random_state != -1) srand(random_state);
  else srand(time(NULL));
  
  for(int row=0; row<row_dimension; row++)
  { 
   M[row]= calloc(row_dimension,sizeof(float)); 
  }
  
  for (int row=0; row<row_dimension; row++)
  { 
   for(int k=0; k<n_connections; k++)
   {  j = (genrand64_int64()%row_dimension); 
            if (j == row) k--;
            else {
                M[row][j] = 2*(int)(genrand64_int64()%2) -1;
                M[j][row] = -M[row][j];
            }
   }
   }
   return M; 
   
}

float** uniform_antisymmetric_sparse(int row_dimension, int n_connections, int random_state)
{ 
  float **M = malloc(row_dimension*sizeof(float*));
  int j; 
  if (random_state != -1) srand(random_state);
  else srand(time(NULL));
  
  for (int row=0; row<row_dimension; row++)
  { 
   M[row] = calloc(row_dimension,sizeof(float));
  } 
  
  for(int row=0; row<row_dimension; row++)
  { 
   for(int k=0; k<n_connections; k++)
   { 
     j = (genrand64_int64()%row_dimension); 
     if (j == row) k--;
     else {
           M[row][j] = 2*genrand64_real2() -1;;
           M[j][row] = -M[row][j];
          } 
          
    }
   }
   
   return M; 
   
} 

Sparse_matrix * Dense_to_Sparse( float ** M, int row_dimension, int column_dimension )
{
    int row,col,count;
    Sparse_matrix *S = malloc(sizeof(Sparse_matrix));

    // Alloca due array temporanei per salvare gli indici e i valori non nulli di una riga
    int *temp_indexes_array = malloc(column_dimension*sizeof(int));
    float *temp_values_array = malloc(column_dimension*sizeof(float));

    // Imposta il numero di righe della matrice sparsa
    S->dimension = row_dimension;
    S->head = malloc(row_dimension*sizeof(Sparse_element));

    for ( row = 0;  row < row_dimension;    row++)
    {
        count = 0;
        for( col = 0;   col < column_dimension; col++)
        {
            // Se il valore è diverso da zero, lo salva nell'array temporaneo
            if ( M[row][col] != 0 )
            {
                temp_indexes_array[count] = col;
                temp_values_array[count] = M[row][col];
                count++;
            }
        }
        // Alloca la memoria per gli indici e i valori non nulli della riga corrente
        S->head[row].dimension = count;
        S->head[row].indexes = malloc(count*sizeof(int));
        S->head[row].values = malloc(count*sizeof(float));

        // Copia gli indici e i valori non nulli dall'array temporaneo alla matrice sparsa
        for (col = 0; col < count; col++) S->head[row].indexes[col] = temp_indexes_array[col];
        for (col = 0; col < count; col++) S->head[row].values[col] = temp_values_array[col];
    }
    // Libera la memoria allocata per gli array temporanei
    free(temp_indexes_array);
    free(temp_values_array);

    return S;
    
}

float prova_bias(float **bias, int n, int m){
    printf("%g\n",bias[n][m]);
    fflush(stdout);
    return bias[n][m]; 
}



ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
    char *buf = NULL;
    size_t bufsize = 0;
    int ch;
    size_t len = 0;

    if (lineptr == NULL || n == NULL || stream == NULL) return -1;

    buf = *lineptr;
    bufsize = *n;

    while ((ch = fgetc(stream)) != EOF) {
        if (len + 1 >= bufsize) {
            bufsize = bufsize ? bufsize * 2 : 128;
            buf = realloc(buf, bufsize);
            if (!buf) return -1;
        }
        buf[len++] = ch;
        if (ch == '\n') break;
    }
    if (len == 0) return -1;
    buf[len] = '\0';

    *lineptr = buf;
    *n = bufsize;
    return len;
}


float** read_matrix(const char* filename, int rows, int cols) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Errore nell'aprire il file %s\n", filename);
        return NULL;
    }

    // Alloca la memoria per la matrice
    float** matrix = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(cols * sizeof(float));
    }

    // Legge i valori della matrice dal file
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%f", &matrix[i][j]);
        }
    }

    // Chiudi il file
    fclose(file);

    return matrix;
}

int** read_int_matrix(const char* filename, int rows, int cols) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Errore nell'aprire il file %s\n", filename);
        return NULL;
    }

    // Alloca la memoria per la matrice
    int** matrix = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
    }

    // Legge i valori della matrice dal file
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%d", &matrix[i][j]);
        }
    }

    // Chiudi il file
    fclose(file);

    return matrix;
}


int **  Initialize_embedding(int vertices, int dimension)
{
    int **embedding = malloc(sizeof(int*)*vertices);
    if (embedding == NULL) {
    perror("Impossibile allocare memoria per embedding");
    exit(EXIT_FAILURE);
    }
    srand(time(NULL) );
    int i,j;

    
    for (i=0;   i<vertices; i++)
    {
        embedding[i] = malloc(sizeof(int)*dimension);
        if (embedding[i] == NULL) {
        perror("Impossibile allocare memoria per embedding");
        exit(EXIT_FAILURE);
        }

        for (j=0;   j<dimension;    j++)
        {
            embedding[i][j] = 2*(genrand64_int64()%2) -1;
        }
    }
    return embedding;

}


void print_embedding_file(char * nome, int ** embedding, int vertices, int dimension)
{
    int i,j,step,iterazioni;
    FILE *file;
    file = fopen(nome, "w"); // Apri il file per scrittura

    if (file == NULL)   perror("Errore nell'aprire il file!\n");

    for (i=0;   i<vertices; i++)
    {
        for (j=0;   j<dimension-1;    j++)
        {
            fprintf(file,"%d ",embedding[i][j]);
        }
        fprintf(file,"%d\n",embedding[i][j]);
    }
    fclose(file); // Chiudi il file

}

void free_sparse( Sparse_matrix * S)
{
    int i,j;

    for (i=0; i < S->dimension; i++)
    {
        free(S->head[i].indexes);
        free(S->head[i].values);
    }
    free(S->head);
    free(S);
}




void print_matrix(float **M, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%.6f ", M[i][j]);
        }
        printf("\n");
    }
}

void print_int_matrix(int **M, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", M[i][j]);
        }
        printf("\n");
    }
}

float** alloc_and_copy_float_matrix_from_np(float* np_array, int rows, int cols){ 
    float** M = malloc(rows*sizeof(float*)); 
    
    
    for (int i=0; i<rows; i++){ 
        M[i] = calloc(cols,sizeof(float)); 
        
        for (int j=0; j<cols;j++){ 
            M[i][j]=(np_array[i*cols+j]); 
        }
    }
    return M; 
}

int** alloc_and_copy_int_matrix_from_np(int* np_array, int rows, int cols){ 
    int** M = malloc(rows*sizeof(int*)); 
    
    
    for (int i=0; i<rows; i++){ 
        M[i] = calloc(cols,sizeof(int)); 
        
        for (int j=0; j<cols;j++){ 
            M[i][j]=np_array[i*cols+j]; 
        }
    }
    return M; 
}












void free_matrix(float** M, int row){ 
    for (int i=0; i<row ;i++){
        free(M[i]); 
        } 
    free(M); 
} 

void free_int_matrix(int** M, int row){ 
    for (int i=0; i<row ;i++){
        free(M[i]); 
        } 
    free(M); 
} 




Sparse_matrix * read_sparse(const char* filename, float interaction)
{
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char* token;
    int* vettore_indici;
    int i=0;
    int j=0;
    int k;
    Sparse_matrix* S;

    FILE* file = fopen(filename, "r");

    if (file == NULL) {
        perror("Errore nell'apertura del file");
        exit(EXIT_FAILURE);
    }
    S = malloc(sizeof(Sparse_matrix));
    if (S == NULL) {
        perror("Impossibile allocare memoria per Graph");
        exit(EXIT_FAILURE);
    }

    int V = 0;
    read = getline(&line, &len, file);
    V = atoi(line);
    S->dimension = V;
    S->head = malloc(V*sizeof(Sparse_element));

    if (S->head == NULL) {
        perror("Impossibile allocare memoria per array");
        exit(EXIT_FAILURE);
    }


    // Leggi le righe del file
    i=0;
    while ((read = getline(&line, &len, file)) != -1 && i<V) {
        j=0;
        token = strtok(line, " ");
        vettore_indici = malloc(atoi(token)*sizeof(int));
        if (vettore_indici == NULL) {
            perror("Impossibile allocare memoria per vettore");
            exit(EXIT_FAILURE);
        }
        (S->head[i]).dimension = atoi(token);
        (S->head[i]).indexes = vettore_indici;
        S->head[i].values = malloc(atoi(token)*sizeof(float));
        for ( k=0; k< S->head[i].dimension; k++) 
        {
            if( genrand64_real2() < interaction) S->head[i].values[k] = 1;
            else S->head[i].values[k] = -1;
        }

        while( token != NULL && j<(S->head[i]).dimension) {
            token = strtok(NULL, " ");
            vettore_indici[j] = atoi(token);
            j++;
            //printf("Riga letta: %s", line);
            }
        i++;
    }
    // Controlla se si è verificato un errore durante la lettura
    if (ferror(file)) {
        perror("Errore nella lettura del file");
    }

    // Libera la memoria allocata per la linea
    free(line);

    // Chiudi il file
    fclose(file);
    return S;
}

double total_energy(Sparse_matrix * A, Sparse_matrix * W, int ** embedding, float bias_coupling, float ** bias)
{ 

 //int s = A->dimension; 
 //printf("Dimensioni di W : %d\n",s);
 //for (int kk=0; kk<10; kk++){
 //    printf("bias: %g\n",bias[kk][kk]);
 //    }
     



   double total_Energy=0; 
   for(int nn=0; nn<A->dimension;nn++)
   { 
    for(int mm=0; mm<W->dimension;mm++)
    { 
     double E_prov = 0; 
     double count= 0; 
     for(int l=0; l<A->head[nn].dimension;l++)
     {for(int k=0; k<W->head[mm].dimension;k++)
     { 
      count+=((float) W->head[mm].values[k])* embedding[A->head[nn].indexes[l]][W->head[mm].indexes[k]];
      }
     }
     E_prov = -((count*embedding[nn][mm])/2 +bias_coupling*bias[nn][mm]*embedding[nn][mm]); 
     total_Energy += E_prov;
     }
     }
     
     
     
     
   return total_Energy; 
   
} 

//metto int perche rendo solo la variazione di energia dato che dell'embedding ho il riferimento void simulate(Sparse_matrix * A, Sparse_matrix * W, int ** embedding, float bias_coupling, float ** bias, double initial_energy) 
//{ 


double parallel_update(Sparse_matrix * A, Sparse_matrix * W, int ** embedding, float bias_coupling, float ** bias, double initial_energy, int steps, int nthreads)
{ 

  omp_set_num_threads(nthreads); 
  
  int rows = A->dimension; 
  int cols = W->dimension;
  
  int** emb2 = malloc(rows*sizeof(int*)); 
    
    
    for (int i=0; i<rows; i++){ 
        emb2[i] = calloc(cols,sizeof(float)); }
        
  
  for (int j=0; j<steps; j++)
  { 
    #pragma omp parallel for schedule(auto)
    for (int n=0; n<A->dimension; n++)
    { 
     for(int m=0; m<W->dimension; m++)
     {
     
       double count = 0; 
    
    //int my_thread_id = omp_get_thread_num();
    
    double var_energia; 
    
    for (int l=0; l<A->head[n].dimension; l++)
   { 
    for (int k=0; k<W->head[m].dimension; k++)
    { 
     count += ((float) W->head[m].values[k])* embedding[A->head[n].indexes[l]][W->head[m].indexes[k]];
    }
  
  
   } 
    
    
    var_energia = count*embedding[n][m] +bias_coupling*bias[n][m]*embedding[n][m];
    
     if (var_energia<0)
    { 
     
     
     //
     emb2[n][m] = embedding[n][m]*-1; 
     double variazione = 2*var_energia; 
     //var_locals[my_thread_id]+=variazione; 
     
    } 
    
  }
  
  }
  
  //qui copio emb2 in embedding --> se devo consegnare questo megli oscambiare indirizzi 
  
  for(int aa =0; aa<rows;aa++)
  { 
   for(int bb=0; bb<cols; bb++)
   { 
   
    embedding[aa][bb]=emb2[aa][bb]; 
   }
   
  }
  
  }
  
  free_int_matrix(emb2,rows); 
  
  return 0; 
  
  }  
  
  
    
    
   
    
  
  

double simulate(Sparse_matrix * A, Sparse_matrix * W, int ** embedding, float bias_coupling, float ** bias, double initial_energy, int steps, int nthreads, double T)
{ 


  double total_energy=initial_energy;
  double var_locals[120] = {0}; 
  int loc_events[120] = {0};
  int total_events = 0;
  omp_set_num_threads(nthreads);
  
  int nn_t = omp_get_max_threads(); 
  // printf("Numero threads = %d, ma ne ho mandati %d\n",nn_t,nthreads);
  for (int j=0; j<steps; j++)
  { 
    int n = rng_int(A->dimension);
  // int n=rand()%(A->dimension);  //scelta casuale del vertice 
  
   
   
   

     
      
   
   #pragma omp parallel for schedule(auto)
   for(int m=0;m<W->dimension; m++)
   {
    
    //printf("Thread %d esegue m = %d\n", omp_get_thread_num(), m);
    //srand(time(NULL)+omp_get_thread_num());  //togliere questo per la versione deterministica 
    int my_thread_id = omp_get_thread_num();
    //printf("I'm process %d\n:",my_thread_id);
    double count = 0; 
    
    //int my_thread_id = omp_get_thread_num();
    
    double var_energia; 
    
    for (int l=0; l<A->head[n].dimension; l++)
   { 
    for (int k=0; k<W->head[m].dimension; k++)
    { 
     count += ((float) W->head[m].values[k])* embedding[A->head[n].indexes[l]][W->head[m].indexes[k]];
    }
  
  
   } 
    
    
    var_energia = count*embedding[n][m] +bias_coupling*bias[n][m]*embedding[n][m];
    //var_energia = count*embedding[n][m] +bias_coupling*bias[n][m]*embedding[n][m];
    //printf("Posizione %i and %i , bias: %g\n",n,m,bias[n][m]);
    //fflush(stdout);
    
    
    if (var_energia<0)
    { 
     
     
     //
     embedding[n][m]*=-1;
     double variazione = 2*var_energia; 
     var_locals[my_thread_id]+=variazione; 
     loc_events[my_thread_id]+=1;
     
    } 
    else if (T > 0.000001)
            {
                //if (u < expf(-(float)var_energia / Temperature)) embedding[n][m] *= -1;
                double u = rng_uniform01(); 
                
                if (u<exp(-var_energia/T)){ 
                                                 embedding[n][m]*=-1; 
                                                 double var = 2*var_energia;
                                                 var_locals[my_thread_id]+=var;
                                                 loc_events[my_thread_id]+=1;
                                                 }
                
               // if (var_energia <= 4999 &&  (float)rand()/RAND_MAX < exponential[(int)var_energia] ) embedding[n][m] *= -1; 
               // if (var_energia > 4999 && (float)rand()/RAND_MAX < exp(-var_energia/Temperature) )  embedding[n][m] *= -1; 
            }
    
 
    } 
    }
    
    
    for (int qq=0; qq<120; qq++){
      total_energy += var_locals[qq]; 
      total_events +=loc_events[qq]; 
      }
      
    //  printf("%d\n",total_events); 
    //  fflush(stdout);
      
      
      return total_energy; 
 
   
   
 
    
    
   } 
   

double simulate_no_parallel(Sparse_matrix * A, Sparse_matrix * W, int ** embedding, float bias_coupling, float ** bias, double initial_energy, int steps)
{ 


  double total_energy=initial_energy;
  double var_local=0;
  for (int j=0; j<steps; j++)
  { 
   int n=rand()%(A->dimension);  //scelta casuale del vertice 
   for(int m=0;m<W->dimension; m++)
   {
    double count = 0; 
    double var_energia; 
    
    for (int l=0; l<A->head[n].dimension; l++)
   { 
    for (int k=0; k<W->head[m].dimension; k++)
    { 
     count += ((float) W->head[m].values[k])* embedding[A->head[n].indexes[l]][W->head[m].indexes[k]];
    }
    } 
    var_energia = count*embedding[n][m] +bias_coupling*bias[n][m]*embedding[n][m];
   
    if (var_energia<0)
    { 
     
     embedding[n][m]*=-1;
     double variazione = 2*var_energia; 
     var_local+=variazione; 
     
    } 
    
    } 
    }
    total_energy += var_local; 
    return total_energy; 
    } 
   
int contains(int *array, int size, int value)
{ 

for (int i = 0; i < size; i++) {
        if (array[i] == value) return 1;
    }
    return 0;
}
  
    
 
double simulate_2hops_no_parallel(Sparse_matrix * A, Sparse_matrix * W, int ** embedding, float bias_coupling, float ** bias, double initial_energy, int steps)  //,int max_neigh)
{ 

  double total_energy=initial_energy;
  double var_local=0;
  for (int j=0; j<steps; j++)
  { 
  
    int n=rand()%(A->dimension);  //scelta casuale del nodo 
    
    int second_neighbors[MAX_NEIGHBORS];
    int second_size = 0;

        // 1. raccogli tutti i secondi vicini
        for (int l = 0; l < A->head[n].dimension; l++) {
            int first_neighbor = A->head[n].indexes[l];

            for (int l2 = 0; l2 < A->head[first_neighbor].dimension; l2++) {
                int second_neighbor = A->head[first_neighbor].indexes[l2];

                if (second_neighbor != n &&
                    !contains(A->head[n].indexes, A->head[n].dimension, second_neighbor) &&
                    !contains(second_neighbors, second_size, second_neighbor)) {

                    second_neighbors[second_size++] = second_neighbor;
                }
            }
        }
    
    for(int m=0; m<W->dimension; m++)
    { 
     double count=0.0; 
     double var_energia =0.0; 
     for(int l=0; l<A->head[n].dimension; l++) 
     { 
      int neighbor = A->head[n].indexes[l];
      for (int k=0;k<W->head[m].dimension; k++)
      {
         count += ((float) W->head[m].values[k])* embedding[neighbor][W->head[m].indexes[k]];
      
      }
     }
     
     
     double second_neighbor_weight =1/3;   //dopo spostarl tra i parametri 
     for(int i=0; i<second_size; i++)
     { 
      int s = second_neighbors[i]; 
      for (int k=0; k<W->head[m].dimension;k++)
      { count += second_neighbor_weight * ((float) W->head[m].values[k]) * embedding[s][W->head[m].indexes[k]];
      } 
      }
      var_energia = count*embedding[n][m] +bias_coupling*bias[n][m]*embedding[n][m];
      
      if (var_energia<0)
    { 
     
     embedding[n][m]*=-1;
     double variazione = 2*var_energia; 
     var_local+=variazione; 
     
    } 
      
     }
     }
     
      total_energy += var_local; 
    return total_energy; 
    } 
     
     
      
       
     
     
   

 double simulate_2hops(Sparse_matrix * A, Sparse_matrix * W, int ** embedding, float bias_coupling, float ** bias, double initial_energy, int steps,int nthreads)  //,int max_neigh)
{ 

  omp_set_num_threads(nthreads);

  double total_energy=initial_energy;
  double var_locals[120] = {0}; 
  for (int j=0; j<steps; j++)
  { 
  
    int n=rand()%(A->dimension);  //scelta casuale del nodo 
    
    int second_neighbors[MAX_NEIGHBORS];
    int second_size = 0;

        // 1. raccogli tutti i secondi vicini
        for (int l = 0; l < A->head[n].dimension; l++) {
            int first_neighbor = A->head[n].indexes[l];

            for (int l2 = 0; l2 < A->head[first_neighbor].dimension; l2++) {
                int second_neighbor = A->head[first_neighbor].indexes[l2];

                if (second_neighbor != n &&
                    !contains(A->head[n].indexes, A->head[n].dimension, second_neighbor) &&
                    !contains(second_neighbors, second_size, second_neighbor)) {

                    second_neighbors[second_size++] = second_neighbor;
                }
            }
        }
    #pragma omp parallel for schedule(auto)
    for(int m=0; m<W->dimension; m++)
    { 
    
     int my_thread_id = omp_get_thread_num();
     double count=0.0; 
     double var_energia =0.0; 
     for(int l=0; l<A->head[n].dimension; l++) 
     { 
      int neighbor = A->head[n].indexes[l];
      for (int k=0;k<W->head[m].dimension; k++)
      {
         count += ((float) W->head[m].values[k])* embedding[neighbor][W->head[m].indexes[k]];
      
      }
     }
     
     
     double second_neighbor_weight =1/3 ;   //dopo spostarl tra i parametri 
     for(int i=0; i<second_size; i++)
     { 
      int s = second_neighbors[i]; 
      for (int k=0; k<W->head[m].dimension;k++)
      { count += second_neighbor_weight * ((float) W->head[m].values[k]) * embedding[s][W->head[m].indexes[k]];
      } 
      }
      var_energia = count*embedding[n][m] +bias_coupling*bias[n][m]*embedding[n][m];
      
      if (var_energia<0)
    { 
     
     embedding[n][m]*=-1;
     double variazione = 2*var_energia; 
     var_locals[my_thread_id]+=variazione;
     
    } 
      
     }
     }
     
      for (int qq=0; qq<120; qq++){
      total_energy += var_locals[qq]; 
      }
      
    
      
     
      
    return total_energy; 
    } 
     
     
      
       
      
  
  


 
 


 
 










    
    


