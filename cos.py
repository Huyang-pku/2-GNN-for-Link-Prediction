import torch
n = 10
edge_index0 = torch.tensor([[1,2],[1,4],[1,6],[1,8],[2,3],[2,4],[2,5],[3,5],[3,7],[3,0],[4,6],[4,8],
                            [5,7],[5,0],[6,9],[6,0],[7,8],[7,9],[8,9],[9,0]])
edge_index1 = torch.tensor([[1,2],[1,4],[1,8],[1,7],[2,3],[2,4],[2,5],[3,5],[3,6],[3,0],[4,6],[4,8],
                            [5,7],[5,0],[6,8],[6,9],[7,9],[7,0],[8,9],[9,0]])
adj0 = torch.zeros(n, n)
adj1 = torch.zeros(n, n)
adj0[edge_index0[:,0],edge_index0[:,1]] = 1
adj0[edge_index0[:,1],edge_index0[:,0]] = 1
adj1[edge_index1[:,0],edge_index1[:,1]] = 1
adj1[edge_index1[:,1],edge_index1[:,0]] = 1
import pdb
pdb.set_trace()