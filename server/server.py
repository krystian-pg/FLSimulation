import flwr as fl
from flwr.server.strategy import FedAvg
import torch

# Definicja modelu (zgodna z klientem)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

# Definicja strategii
class SaveModelStrategy(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # Wywołanie metody nadrzędnej FedAvg.aggregate_fit
        super_result = super().aggregate_fit(rnd, results, failures)
        
        # Sprawdzenie, czy super().aggregate_fit zwróciło krotkę
        if isinstance(super_result, tuple):
            aggregated_parameters = super_result[0]
            metrics = super_result[1]
        else:
            aggregated_parameters = super_result
            metrics = {}

        if aggregated_parameters is not None:
            # Konwersja Parameters na listę ndarray
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Tworzenie nowego modelu
            model = Net()
            
            # Tworzenie state_dict z aggregated_weights
            state_dict = {}
            for (k, v) in zip(model.state_dict().keys(), ndarrays):
                state_dict[k] = torch.tensor(v)
            
            # Ładowanie state_dict do modelu
            model.load_state_dict(state_dict, strict=True)
            
            # Zapisanie modelu
            torch.save(model.state_dict(), f"model_round_{rnd}.pth")
        
        # Zwrócenie aggregated_parameters oraz metrics
        return aggregated_parameters, metrics

# Uruchomienie serwera Flower
def main():
    # Ścieżki do certyfikatów
    cert_path = "/certs/server.crt"
    key_path = "/certs/server.key"
    ca_cert_path = "/certs/ca.crt"

    # Odczyt certyfikatów jako bajty
    with open(ca_cert_path, "rb") as ca_cert_file, \
         open(cert_path, "rb") as cert_file, \
         open(key_path, "rb") as key_file:
        ca_cert = ca_cert_file.read()
        server_cert = cert_file.read()
        server_key = key_file.read()

    strategy = SaveModelStrategy()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        certificates=(ca_cert, server_cert, server_key)
    )

if __name__ == "__main__":
    main()
