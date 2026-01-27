import torch
from flwr.client import NumPyClient
from flwr.client.mod import fixedclipping_mod, secaggplus_mod
from flwr.clientapp import ClientApp
from flwr.common import Context

from flwr_mnist.task import Net, get_weights, load_data, set_weights, test, train, get_device



class FlowerClient(NumPyClient):
    def __init__(self, trainloader, testloader) -> None:
        self.net = Net()
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = get_device()

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.testloader,
            epochs=4, # TODO: make this configable
            device=self.device
        )
        return get_weights(self.net), len(self.trainloader.dataset), results
    
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {'accuracy': accuracy}
    
def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    trainloader, testloader = load_data(
        partition_id=partition_id, num_partitions=context.node_config["num-partitions"]
    )

    return FlowerClient(trainloader=trainloader, testloader=testloader).to_client()
    
# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[]
)