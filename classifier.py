from utils import prepare_data

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = prepare_data()
    assert train_x.shape[0] == train_y.shape[0]
    assert test_x.shape[0] == test_y.shape[0]
