choices = ["bit", "eth", "ltc"]
dataset_names = ["investing", "coinmarketcap", "yahoo", "gecko"]

dataset_paths = {
    "bit": {
        "investing": "data/bitcoin_investing.csv",
        "coinmarketcap": "data/bitcoin_coinmarketcap.csv",
        "yahoo": "data/bitcoin_yahoo.csv",
        "gecko": "data/bitcoin_gecko.csv"
    },
    "eth": {
        "investing": "data/ethereum_investing.csv",
        "coinmarketcap": "data/ethereum_coinmarketcap.csv",
        "yahoo": "data/ethereum_yahoo.csv",
        "gecko": "data/ethereum_gecko.csv"
    },
    "ltc": {
        "investing": "data/litecoin_investing.csv",
        "coinmarketcap": "data/litecoin_coinmarketcap.csv",
        "yahoo": "data/litecoin_yahoo.csv",
        "gecko": "data/litecoin_gecko.csv"
    }
}
