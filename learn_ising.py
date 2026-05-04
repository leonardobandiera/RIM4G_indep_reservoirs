import ctypes 
import numpy as np 
from argparse import ArgumentParser 
import sys 
import os 
import inspect


    
class SparseElement(ctypes.Structure): 
    _fields_ = [ 
        ("dimension", ctypes.c_int), 
        ("indexes", ctypes.POINTER(ctypes.c_int)), 
        ("values",ctypes.POINTER(ctypes.c_float))
        ] 
        
class SparseMatrix(ctypes.Structure): 
    _fields_ = [
        ("dimension", ctypes.c_int), 
        ("head",ctypes.POINTER(SparseElement))
        ]
        
        

                       
def get_initializer_w_matrix(lib_ising): 
    return {1:lib_ising.binary_sparse,
            2:lib_ising.uniform_sparse,
            3:lib_ising.binary_symmetric_sparse,
            4:lib_ising.uniform_symmetric_sparse,
            5:lib_ising.uniform_symmetric_traceless_sparse,
            6:lib_ising.binary_antisymmetric_sparse,
            7:lib_ising.uniform_antisymmetric_sparse,
            8:lib_ising.binary_symmetric_sparse_blocks
            }
            
            
def get_float_matrix_from_c(ptr, row, col): 
    result = np.zeros((row,col),dtype=np.float32) 
    for i in range(row): 
        for j in range(col): 
            result[i,j] = ptr[i][j] 
    return result 
    
def get_int_matrix_from_c(ptr, row, col): 
    result = np.zeros((row,col),dtype=np.int32) 
    for i in range(row): 
        for j in range(col): 
            result[i,j] = ptr[i][j] 
    return result 

def initialization(arr):
    return np.where(arr == 0, -1, arr)

            
            
    
        
        
class Ising(): #per ora lascio perdere iterations e termalization
               #aggiungo dopo inter steps 
    
    def __init__(self,data,adj_matrix_name,vertices,steps,temperature,w_connections,random_state_w,random_state_iteration,dimension,bias_coupling,nthreads=6,matrix_type=3, continue_= False, root_ = "./results",data_="Photo"):
        
        self.nthreads = nthreads
        self.initialize_generator(random_state_iteration) 
        
        self.data_=data_ 
        self.data = data
        
        self.root = root_
        
        self.vertices = vertices 
    
        self.steps = steps #non lo sto usando , lo controllo da fuori 
        
        self.temperature = temperature 
        
        self.w_connections = w_connections #numero connsessioni del reservoir 
        
        self.random_state_w = random_state_w   #seed del reservoir 
        
        self.random_state_iteration = random_state_iteration  #seed dell'inizializzazione 
        
        self.dimension = dimension 
        
        self.bias_coupling = bias_coupling 
        
        self.numpy_bias = None 
        
        self.matrix_w = None   #reservoir denso 
        
        self.sparse_matrix_w = None #reservoir sparso 
        
        self.embedding = None 
        
        self.initializer = get_initializer_w_matrix(lib_ising) 
       
        self.A = lib_ising.read_sparse(adj_matrix_name,1) #mettere il nome della matrice tra gli argomenti 
     
        self.bias = None 
        
        
         
        if (not continue_): #inizializza matrice w , la sua forma sparsa e l'embedding 
        
            name_bias = os.path.join(self.root,"input_bias.txt") 
            numpy_bias =self.data.x.numpy() @ (2*np.random.rand(self.data.x.shape[1],self.dimension)-1).astype(np.float32)
            self.numpy_bias = np.ascontiguousarray(numpy_bias, dtype=np.float32)
            
            np.savetxt(name_bias,self.numpy_bias)        #self.data.x.numpy() @ (2*np.random.rand(self.data.x.shape[1],self.dimension)-1))
            #name_bias =name_bias.encode('utf-8')
            #self.bias = lib_ising.read_matrix(name_bias,self.vertices,self.dimension);
            flat_ptr = self.numpy_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) 
            self.bias= lib_ising.alloc_and_copy_float_matrix_from_np(flat_ptr,self.numpy_bias.shape[0],self.numpy_bias.shape[1])
            
            self.matrix_w = self.generate_w_matrix(matrix_type) 
            
         
            self.sparse_matrix_w= self.generate_sparse_matrix(self.matrix_w,self.dimension,self.dimension) 
            self.embedding =self.initialize_embedding()
           # self.sparse_matrix_w= self.generate_sparse_matrix(self.matrix_w,self.dimension,self.dimension)
           # self.sparse_matrix_w= self.generate_sparse_matrix(self.matrix_w,self.dimension,self.dimension)
            #new_init = initialization(data.x.numpy().astype(np.int32))
            #new_init=new_init.astype(np.int32)
            #flat_ptr = new_init.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) 
            #self.embedding=lib_ising.alloc_and_copy_int_matrix_from_np(flat_ptr,self.vertices,self.dimension) 
    
        
            
        else: 
            print("continue") 
            name_bias = os.path.join(self.root,"input_bias.txt") 
            name_bias =name_bias.encode('utf-8')
            self.bias = lib_ising.read_matrix(name_bias,self.vertices,self.dimension);
            self.embedding = self.copy_embedding_to_c()
            self.matrix_w = self.copy_matrix_w_to_c()
            #self.matrix_w = self.generate_w_matrix(matrix_type) 
            self.sparse_matrix_w= self.generate_sparse_matrix(self.matrix_w,self.dimension,self.dimension) 
            #self.sparse_matrix_w= self.generate_sparse_matrix(self.matrix_w,self.dimension,self.dimension) 
            #self.sparse_matrix_w= self.generate_sparse_matrix(self.matrix_w,self.dimension,self.dimension)
             
   
               
    def initialize_embedding(self): 
       
        return lib_ising.Initialize_embedding(self.vertices,self.dimension) 
        
    def initialize_generator(self,random_state): 
        lib_ising.init_generator(random_state,self.nthreads) 
        
        
    def calculate_energy(self): 
        #total_energy(Sparse_matrix * A, Sparse_matrix * W, int ** embedding, float bias_coupling, float ** bias)
        return lib_ising.total_energy(self.A,self.sparse_matrix_w,self.embedding,self.bias_coupling,self.bias) 
        #return 0 
        
        
        
    def clean_memory(self): 
        self.clean_matrix(self.matrix_w,self.dimension) 
        self.matrix_w = None 
        self.clean_sparse_matrix(self.sparse_matrix_w) 
        self.sparse_matrix_w = None 
        self.clean_int_matrix(self.embedding,self.vertices) 
        self.embedding= None 
        self.clean_sparse_matrix(self.A) 
        self.A= None 
        
        
    
    
            
    
    def generate_w_matrix(self,int_type): 
        
        init_ = self.initializer[int_type] 
        if int_type == 8: #caso della matrice a blocchi
            n_blocks = 16
            return init_(self.dimension, self.w_connections, n_blocks, self.random_state_w)
            
        n_params = len(init_.argtypes)
        if n_params==4: 
            return init_(self.dimension,self.dimension,self.w_connections, self.random_state_w) 
        
        else: 
            return init_(self.dimension,self.w_connections, self.random_state_w)
    
        #return lib_ising.generate_dense_matrix(10,10)
        
    def generate_sparse_matrix(self,ptr,dim1,dim2): 
        return lib_ising.Dense_to_Sparse(ptr,dim1,dim2)
        
    def parallel_update(self,j,initial_energy=None): 
        if initial_energy is None: 
            initial_energy = 0 
        return lib_ising.parallel_update(self.A,self.sparse_matrix_w,self.embedding,self.bias_coupling,self.bias,initial_energy,j,self.nthreads)
        
    
    #questa è quella usata principalmetne per simulare     
    def simulate(self, j,initial_energy=None,temperature=None): 
    #(Sparse_matrix * A, Sparse_matrix * W, int ** embedding, float bias_coupling, float ** bias, double initial_energy, int steps, int nthreads)
        #initial_energy = self.calculate_energy()
        #initial_energy =0  #provvisorio e falso 
        if initial_energy is None: 
            initial_energy = 0 
        
        if temperature is None: 
            temperature= self.temperature   #nel caso cambi per annealing
        #print(initial_energy) 
      #  temperature = temperature*1.380649e-23 
        return lib_ising.simulate(self.A,self.sparse_matrix_w,self.embedding,self.bias_coupling,self.bias,initial_energy,j,self.nthreads,temperature) 
        
    def simulate_no_parallel(self,j,initial_energy=None): 
        if initial_energy is None: 
            initial_energy=0
        return lib_ising.simulate_no_parallel(self.A,self.sparse_matrix_w,self.embedding,self.bias_coupling,self.bias,initial_energy,j) 
        
   #simulate_2hops_no_parallel
    def simulate_2hops_no_parallel(self,j,initial_energy=None): 
        if initial_energy is None: 
            initial_energy=0
        return lib_ising.simulate_2hops_no_parallel(self.A,self.sparse_matrix_w,self.embedding,self.bias_coupling,self.bias,initial_energy,j) 
        
    def simulate_2hops(self,j,initial_energy=None): 
        if initial_energy is None: 
            initial_energy=0
        return lib_ising.simulate_2hops_no_parallel(self.A,self.sparse_matrix_w,self.embedding,self.bias_coupling,self.bias,initial_energy,j,self.nthreads) 
        
    def clean_matrix(self,ptr,dim): 
        #passare anche la dimensione delle righe per pulire la matrice 
        lib_ising.free_matrix(ptr,dim)
        #self.matrix_w = None 
    def clean_int_matrix(self,ptr,dim): 
        lib_ising.free_int_matrix(ptr,dim)
        #magari questa gestirla nel caso precedente 
        
    def clean_sparse_matrix(self,ptr): 
        #aggiungere if puntatore diverso da None 
        lib_ising.free_sparse(ptr)
        self.sparse_matrix = None 
        
    def init_embedding(self,folder_="./"): 
        self.clean_int_matrix(self.embedding,self.vertices) 
        self.embedding= None 
        self.embedding =self.copy_embedding_to_c(root_=folder_)

    def copy_embedding_to_c(self, root_=None):
        if root_ is None: 
            result = np.load(os.path.join(self.root,"embedding.npy"))
        else:
            name_init = "embedding_"+self.data_+".npy"
            result = np.load(os.path.join(root_,name_init))
        flat_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) 
        return lib_ising.alloc_and_copy_int_matrix_from_np(flat_ptr,self.vertices,self.dimension) 
    
    def copy_matrix_w_to_c(self): 
        result = np.load(os.path.join(self.root,"numpy_w.npy"))
        flat_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) 
        return lib_ising.alloc_and_copy_float_matrix_from_np(flat_ptr,self.dimension,self.dimension)  
        return 
    def save_to_numpy(self,save_file=False): 
        if self.matrix_w is not None:
            numpy_w = get_float_matrix_from_c(self.matrix_w, self.dimension, self.dimension)
        if self.embedding is not None: 
            numpy_emb = get_int_matrix_from_c(self.embedding,self.vertices, self.dimension)
        
        if save_file: 
            if not os.path.exists(self.root):
                os.makedirs(self.root)
       
            np.save(os.path.join(self.root,"numpy_w.npy"),numpy_w)
            np.save(os.path.join(self.root,"embedding.npy"),numpy_emb)
            
        return numpy_w,numpy_emb
        
    def print_int_matrix(self,ptr,dim1,dim2): 
        lib_ising.print_int_matrix(ptr,dim1,dim2)
            
        
    def __str__(self):
    
        return (
            f"IsingModel:\n"
            f"  Dimensione: {self.dimension}\n"
            f"  Dataset: {self.data}\n"
            f"  Bias coupling: {self.bias_coupling:.4f}"
        )
        
    
    

def setup_lib(lib_path): 
    lib = ctypes.cdll.LoadLibrary(lib_path) 
    lib.generate_dense_matrix.argtypes = [ctypes.c_int,ctypes.c_int]
    lib.generate_dense_matrix.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    lib.print_matrix.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),ctypes.c_int,ctypes.c_int]
    lib.free_matrix.argtypes =  [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int]
    lib.free_int_matrix.argtypes =  [ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), ctypes.c_int]
    lib.binary_sparse.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.binary_sparse.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    lib.uniform_sparse.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.uniform_sparse.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    lib.binary_symmetric_sparse.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.binary_symmetric_sparse.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    lib.uniform_symmetric_sparse.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.uniform_symmetric_sparse.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    lib.uniform_symmetric_traceless_sparse.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.uniform_symmetric_traceless_sparse.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    lib.binary_antisymmetric_sparse.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.binary_antisymmetric_sparse.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    lib.uniform_antisymmetric_sparse.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.uniform_antisymmetric_sparse.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    lib.binary_symmetric_sparse_blocks.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.binary_symmetric_sparse_blocks.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    lib.Dense_to_Sparse.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int]
    lib.Dense_to_Sparse.restype = ctypes.POINTER(SparseMatrix)
    lib.free_sparse.argtypes = [ctypes.POINTER(SparseMatrix)]
    lib.read_matrix.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.read_matrix.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float)) 
   
    lib.read_int_matrix.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.read_int_matrix.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_int)) 
    lib.Initialize_embedding.argtypes = [ctypes.c_int,ctypes.c_int]
    lib.Initialize_embedding.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_int)) 
    lib.print_embedding_file.argtypes = [ctypes.c_char_p,ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),ctypes.c_int, ctypes.c_int]
    lib.print_int_matrix.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),ctypes.c_int, ctypes.c_int]
    lib.alloc_and_copy_float_matrix_from_np.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]#non usare questa per gli embedding 
    lib.alloc_and_copy_float_matrix_from_np.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    lib.alloc_and_copy_int_matrix_from_np.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]#non usare questa per gli embedding 
    lib.alloc_and_copy_int_matrix_from_np.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
    lib.read_sparse.argtypes = [ctypes.c_char_p, ctypes.c_float]
    lib.read_sparse.restype = ctypes.POINTER(SparseMatrix)
    lib.total_energy.argtypes =[ctypes.POINTER(SparseMatrix),ctypes.POINTER(SparseMatrix),ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),ctypes.c_float,ctypes.POINTER(ctypes.POINTER(ctypes.c_float))]
    lib.total_energy.restype= ctypes.c_double
    lib.init_generator.argtypes = [ctypes.c_int,ctypes.c_int]
    lib.init_generator.restype = None
    lib.simulate.argtypes = [ctypes.POINTER(SparseMatrix),ctypes.POINTER(SparseMatrix),ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),ctypes.c_float,ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_double]
    lib.simulate.restype = ctypes.c_double
    lib.simulate_no_parallel.argtypes = [ctypes.POINTER(SparseMatrix),ctypes.POINTER(SparseMatrix),ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),ctypes.c_float,ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),ctypes.c_double,ctypes.c_int]
    lib.simulate_no_parallel.restype = ctypes.c_double
    
    lib.simulate_2hops_no_parallel.argtypes = [ctypes.POINTER(SparseMatrix),ctypes.POINTER(SparseMatrix),ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),ctypes.c_float,ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),ctypes.c_double,ctypes.c_int]
    lib.simulate_2hops_no_parallel.restype = ctypes.c_double
    
    lib.prova_bias.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),ctypes.c_int,ctypes.c_int]
    lib.prova_bias.restype =  ctypes.c_float
    
    lib.parallel_update.argtypes = [ctypes.POINTER(SparseMatrix),ctypes.POINTER(SparseMatrix),ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),ctypes.c_float,ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),ctypes.c_double,ctypes.c_int,ctypes.c_int]
    lib.parallel_update.restype = ctypes.c_double
    
    return lib
    





lib_ising=setup_lib("./libc.so") 
