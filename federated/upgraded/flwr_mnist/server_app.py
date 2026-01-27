from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping, FedAvg
from flwr.server.workflow import DefaultWorkflow

from flwr_mnist.task import Net, get_weights

app = ServerApp()

# weighted average
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }

@app.main()
def main(grid: Grid, context: Context) -> None:

    # Initialize global model
    model_weights = get_weights(Net())
    parameters = ndarrays_to_parameters(model_weights)
    
    # get inputs from config 
    num_sampled_clients = context.run_config['num-sampled-clients']
    num_rounds = context.run_config['num-server-rounds']
    fraction_fit = 0.5 # TODO: make this configurable as well
    min_fit_clients = int(num_sampled_clients * fraction_fit)

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=min_fit_clients,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters)

    # Construct Legacy Context
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

    # Create train/eval workflow
    workflow = DefaultWorkflow()

    # Execute the workflow!
    workflow(grid, context)