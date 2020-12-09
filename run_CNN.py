def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_map = pickle.load(fo, encoding='bytes')
    return data_map


def main():
    file_path = "test_batch"
    img_data = unpickle(file_path)
    print(img_data.decode().keys())


if __name__ == "__main__":
    main()
