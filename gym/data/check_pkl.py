import pickle

if __name__ == "__main__":
    file_path = 'hopper-medium-v2-copy.pkl'

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Type of data: ", type(data))
    print("Length of data: ", len(data))
    print("Index 0 in data: ", data[0].keys())
    print("Observations: ", data[0]['observations'].shape)
    print("Observations: ", data[0]['next_observations'].shape)