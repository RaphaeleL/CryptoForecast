choices = ["bit", "eth", "ltc"]
dataset_names = ["investing", "coinmarketcap", "yahoo"]

dataset_paths = {
    "bit": {
        "investing": "data/bitcoin_investing.csv",
        "coinmarketcap": "data/bitcoin_coinmarketcap.csv",
        "yahoo": "data/bitcoin_yahoo.csv"
    },
    "eth": {
        "investing": "data/ethereum_investing.csv",
        "coinmarketcap": "data/ethereum_coinmarketcap.csv",
        "yahoo": "data/ethereum_yahoo.csv"
    },
    "ltc": {
        "investing": "data/litecoin_investing.csv",
        "coinmarketcap": "data/litecoin_coinmarketcap.csv",
        "yahoo": "data/litecoin_yahoo.csv"
    }
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