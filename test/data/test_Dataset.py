import torch

from torch_geometric.data import Data

from torch_hep.data import GraphDataset


def test_dataset():
    x1 = torch.Tensor([[1], [1], [1]])
    x2 = torch.Tensor([[2], [2], [2]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    face = torch.tensor([[0], [1], [2]])

    data1 = Data(x1, edge_index, face=face, test_int=1, test_str='1')
    data1.num_nodes = 10

    data2 = Data(x2, edge_index, face=face, test_int=2, test_str='2')
    data2.num_nodes = 5

    dataset = GraphDataset(graphs=[data1, data2])
    assert str(dataset) == 'GraphDataset(2)'
    assert len(dataset) == 2

    assert len(dataset[0]) == 6
    assert dataset[0].num_nodes == 10
    assert dataset[0].x.tolist() == x1.tolist()
    assert dataset[0].edge_index.tolist() == edge_index.tolist()
    assert dataset[0].face.tolist() == face.tolist()
    assert dataset[0].test_int == 1
    assert dataset[0].test_str == '1'

    assert len(dataset[1]) == 6
    assert dataset[1].num_nodes == 5
    assert dataset[1].x.tolist() == x2.tolist()
    assert dataset[1].edge_index.tolist() == edge_index.tolist()
    assert dataset[1].face.tolist() == face.tolist()
    assert dataset[1].test_int == 2
    assert dataset[1].test_str == '2'

    dataset.save_to('./test/dataset/test_dataset')
    dataset_loaded = GraphDataset(root='./test/dataset/test_dataset')
    assert str(dataset_loaded) == 'GraphDataset(2)'
    assert len(dataset_loaded) == 2

    assert len(dataset_loaded[0]) == 6
    assert dataset_loaded[0].num_nodes == 10
    assert dataset_loaded[0].x.tolist() == x1.tolist()
    assert dataset_loaded[0].edge_index.tolist() == edge_index.tolist()
    assert dataset_loaded[0].face.tolist() == face.tolist()
    assert dataset_loaded[0].test_int == 1
    assert dataset_loaded[0].test_str == '1'

    assert len(dataset_loaded[1]) == 6
    assert dataset_loaded[1].num_nodes == 5
    assert dataset_loaded[1].x.tolist() == x2.tolist()
    assert dataset_loaded[1].edge_index.tolist() == edge_index.tolist()
    assert dataset_loaded[1].face.tolist() == face.tolist()
    assert dataset_loaded[1].test_int == 2
    assert dataset_loaded[1].test_str == '2'