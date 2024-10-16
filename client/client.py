import flwr as fl
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
import pytorch_lightning as pl
import os

# Definicja modelu
class MNISTModel(pl.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

# Przygotowanie danych
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    return train_loader

# Definicja klienta Flower
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = MNISTModel()
        self.train_loader = load_data()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False)
        trainer.fit(self.model, self.train_loader)
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Można dodać kod ewaluacji
        return 0.0, len(self.train_loader.dataset), {}

def main():
    # Ścieżka do certyfikatu CA
    ca_cert_path = "/certs/ca.crt"

    # Sprawdzenie istnienia certyfikatu
    if not os.path.exists(ca_cert_path):
        raise FileNotFoundError("Brak certyfikatu CA.")

    client = FlowerClient()
    fl.client.start_client(
        server_address="server:8080",
        client=client,
        root_certificates=open(ca_cert_path, "rb").read()
    )

if __name__ == "__main__":
    main()
