import torch

from torch_hep.graph import GraphBuilder


def test_GraphBuilder():
    g1 = 1
    g2 = 2
    g3 = 3
    n1 = [1,3,4]
    n2 = [0.1,0.2,0.4]
    n3 = [0,0,0]
    ei = [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]
    e1 = [2,4,2,3,9,8]
    e2 = [1,5,4,3,4,7]
    e3 = [1,5,4,3,4,7]

    G = GraphBuilder()
    G.add_asGlobal('u',g1=g1,g2=g2)
    G.add_asNode('x',n1=n1,n2=n2)
    G.add_asEdge('e',index=ei,e1=e1,e2=e2)
    G.add_asNode('x',n3=n3)
    G.add_asGlobal('u',g3=g3)
    G.add_asEdge('e',e3=e3)

    assert torch.all(torch.eq(G['u'], torch.tensor([1,2,3])))
    assert torch.all(torch.eq(G['x'], torch.tensor([[1,0.1,0],[3,0.2,0],[4,0.4,0]])))
    assert torch.all(torch.eq(G['e'], torch.tensor([[2,1,1],[4,5,5],[2,4,4],[3,3,3],[9,4,4],[8,7,7]])))
    assert torch.all(torch.eq(G['edge_index'], torch.tensor([[0,0,1,1,2,2],[1,2,0,2,0,1]])))

    for i in range(0,3):
        assert (G['u_features'][i] == ['g1','g2','g3'][i])
        assert (G['x_features'][i] == ['n1','n2','n3'][i])
        assert (G['e_features'][i] == ['e1','e2','e3'][i])
