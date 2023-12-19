choices = ["bit", "eth", "ltc"]

dataset_paths = {
    "bit": "data/full_bitcoin.csv",
    "eth": "data/full_ethereum.csv",
    "ltc": "data/full_litecoin.csv"
}

batch_sizes = {
    "bit": 32,
    "eth": 32,
    "ltc": 32
}

epochs = {
    "bit": 100,
    "eth": 100,
    "ltc": 100
}

durations = {
    "bit": 50,
    "eth": 50,
    "ltc": 50
}