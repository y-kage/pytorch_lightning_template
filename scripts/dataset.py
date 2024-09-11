from torch.utils.data import Dataset


class RandomWeightDataset(Dataset):
    def __init__(self, data_dir: str = "../data", train=True):
        super().__init__()
        key = None
        if train == True:
            key = "train_inputs"
        else:
            key = "test_inputs"
        self.data = None
        self.weight_mean = None
        self.weight_std = None
        self.pose_mean = None
        self.pose_std = None
        with open(os.path.join(data_dir, "random_weight.json")) as f:
            data = json.load(f)
            self.weight_mean = data["weight_mean"]
            self.weight_std = data["weight_std"]
            self.pose_mean = data["pose_mean"]
            self.pose_std = data["pose_std"]
            self.data = np.array(data[key])

    def __getitem__(self, index: int):
        temp = self.data[index]
        # temp = (temp - self.mean) / self.std
        _input = torch.Tensor([temp])

        return _input, _input

    def __len__(self) -> int:
        return len(self.data)
