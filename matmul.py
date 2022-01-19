from torch import Tensor
import torch
from torch_sparse.storage import SparseStorage
@torch.jit.script
def coalesce(index, value, m:int, n:int, op:str="add"):
    '''
    copied from torch-sparse
    '''
    #print(index.shape,value.shape)
    storage = SparseStorage(row=index[0], col=index[1], value=value, sparse_sizes=(m, n), is_sorted=False)
    storage = storage.coalesce(reduce=op)
    return torch.stack([storage.row(), storage.col()], dim=0), storage.value()
#@torch.jit.script
def matmul(A: Tensor, B: Tensor, D: Tensor):
    '''
    A: (I,K)
    B: (J,K)
    D: (3,L), L is the number of non-zero elements in D. Three rows of D contain i, j, k respectively
    return : idx, val form a coo sparse matrix
    '''
    I = A.shape[0]
    J = B.shape[0]
    K = A.shape[1]
    idx_A = D[0]*K+D[2]
    val_A = A.reshape(-1,A.shape[2])[idx_A]
    idx_B = D[1]*K+D[2]
    val_B = B.reshape(-1,A.shape[2])[idx_B]
    val = val_A*val_B
    idx, val = coalesce(D[:2], val, I, J)
    return torch.sparse_coo_tensor(idx, val, (I, J, A.shape[2])).to_dense()

'''
A = torch.rand((4,5,6))
B = torch.rand((3,5,6))
D = torch.tensor([[1,3,3],[2,1,1],[3,4,3]])
S = matmul(A, B, D)
print(S.shape)
'''