from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig, start_server
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

# ============================
# Metric Aggregation
# ============================
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


# ============================
# Strategy and Config
# ============================
strategy = FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_available_clients=3,  # require at least 3 clients
)

config = ServerConfig(num_rounds=20)  # 20 FL rounds


# ============================
# Flower ServerApp
# ============================
app = ServerApp(
    config=config,
    strategy=strategy,
)


# ============================
# Legacy Mode
# ============================
if __name__ == "__main__":
    start_server(
        server_address="0.0.0.0:5006",
        config=config,
        strategy=strategy,
    )
