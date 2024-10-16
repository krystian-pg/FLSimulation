# FLSimulation

## **Spis Treści**

1. [Wprowadzenie](#wprowadzenie)
2. [Wymagania Wstępne](#wymagania-wstępne)
3. [Struktura Projektu](#struktura-projektu)
4. [Konfiguracja SSL](#konfiguracja-ssl)
5. [Przygotowanie Środowiska](#przygotowanie-środowiska)
6. [Budowanie i Uruchamianie Kontenerów](#budowanie-i-uruchamianie-kontenerów)
7. [Opis Plików](#opis-plików)
    - [docker-compose.yml](#docker-composeyml)
    - [Dockerfile dla Serwera](#dockerfile-dla-serwera)
    - [Dockerfile dla Klientów](#dockerfile-dla-klientów)
    - [server.py](#serverpy)
    - [client.py](#clientpy)
    - [requirements.txt](#requirementstxt)
8. [Monitorowanie i Logowanie](#monitorowanie-i-logowanie)
9. [Rozwiązywanie Problemów](#rozwiązywanie-problemów)
10. [Dodatkowe Informacje](#dodatkowe-informacje)
11. [Licencja](#licencja)

## **Wprowadzenie**

Ten projekt demonstruje symulację Federated Learning (FL) z wykorzystaniem frameworka Flower (`flwr`) oraz Docker Compose. Umożliwia uruchomienie serwera FL oraz dwóch klientów FL w izolowanych kontenerach Docker, komunikujących się poprzez sieć Docker. Dodatkowo, konfiguracja obejmuje zabezpieczenie komunikacji za pomocą certyfikatów SSL.

## **Wymagania Wstępne**

Przed rozpoczęciem upewnij się, że masz zainstalowane następujące narzędzia na swojej maszynie:

- **Docker**: [Instalacja Dockera](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Instalacja Docker Compose](https://docs.docker.com/compose/install/)
- **Git**

## **Struktura Projektu**

```
FLSimulation/
├── docker-compose.yml
├── server/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── server.py
│   ├── ca.crt
│   ├── server.crt
│   └── server.key
└── client/
    ├── Dockerfile
    ├── requirements.txt
    ├── client.py
```

- **docker-compose.yml**: Plik konfiguracji Docker Compose definiujący usługi serwera i klientów.
- **server/**: Katalog zawierający pliki serwera FL.
    - **Dockerfile**: Instrukcje do budowania obrazu Docker dla serwera.
    - **requirements.txt**: Lista zależności Pythona dla serwera.
    - **server.py**: Kod serwera Flower.
    - **ca.crt**, **server.crt**, **server.key**: Certyfikaty SSL.
- **client/**: Katalog zawierający pliki klientów FL.
    - **Dockerfile**: Instrukcje do budowania obrazu Docker dla klientów.
    - **requirements.txt**: Lista zależności Pythona dla klientów.
    - **client.py**: Kod klienta Flower.

## **Konfiguracja SSL**

Aby zapewnić bezpieczną komunikację między serwerem a klientami, używamy certyfikatów SSL. Upewnij się, że posiadasz odpowiednie pliki certyfikatów:

1. **ca.crt**: Certyfikat urzędu certyfikacji (CA).
2. **server.crt**: Certyfikat serwera.
3. **server.key**: Klucz prywatny serwera.

Umieść te pliki w katalogu `server/`. Upewnij się, że Docker ma dostęp do tych plików poprzez odpowiednią konfigurację wolumenów w `docker-compose.yml`.

## **Przygotowanie Środowiska**

1. **Klonowanie Repozytorium (opcjonalnie):**

    ```bash
    git clone https://github.com/twoje-repozytorium/FLSimulation.git
    cd FLSimulation
    ```

2. **Umieszczenie Certyfikatów SSL:**

    Umieść pliki `ca.crt`, `server.crt` i `server.key` w katalogu `server/`.

## **Budowanie i Uruchamianie Kontenerów**

Aby zbudować obrazy Docker i uruchomić kontenery, wykonaj następujące kroki:

1. **Usunięcie Istniejących Kontenerów (opcjonalnie):**

    ```bash
    docker-compose down
    ```

2. **Czyszczenie Niepotrzebnych Zasobów Dockera (opcjonalnie):**

    ```bash
    docker system prune -f
    ```

3. **Budowanie Obrazów Docker:**

    ```bash
    docker-compose build
    ```

4. **Uruchomienie Kontenerów:**

    ```bash
    docker-compose up --build
    ```

    Logi z kontenerów będą wyświetlane w konsoli. Aby przerwać działanie kontenerów, użyj `Ctrl+C`.

## **Opis Plików**

### **docker-compose.yml**

Plik `docker-compose.yml` definiuje trzy usługi: `server`, `client1` i `client2`. Usługi `Prometheus` i `Grafana` są zakomentowane i nie są uruchamiane.

```yaml
services:
  server:
    build: ./server
    ports:
      - "8080:8080"
    volumes:
      - ./server:/app
      - ./server/ca.crt:/certs/ca.crt
      - ./server/server.crt:/certs/server.crt
      - ./server/server.key:/certs/server.key
    networks:
      - fl_network
    command: python server.py

  client1:
    build: ./client
    volumes:
      - ./client:/app/client
      - ./server/ca.crt:/certs/ca.crt
    networks:
      - fl_network
    depends_on:
      - server
    command: python client.py

  client2:
    build: ./client
    volumes:
      - ./client:/app/client
      - ./server/ca.crt:/certs/ca.crt
    networks:
      - fl_network
    depends_on:
      - server
    command: python client.py

  # Prometheus i Grafana zostały zakomentowane

networks:
  fl_network:
    driver: bridge
```

#### **Wyjaśnienie Konfiguracji:**

- **server**:
  - **build**: Ścieżka do katalogu serwera, zawierającego Dockerfile.
  - **ports**: Przekierowanie portu 8080 hosta na port 8080 kontenera.
  - **volumes**: Montowanie katalogu serwera oraz certyfikatów do kontenera.
  - **networks**: Przyłączenie do sieci `fl_network`.
  - **command**: Komenda do uruchomienia serwera (`python server.py`).

- **client1** i **client2**:
  - **build**: Ścieżka do katalogu klienta, zawierającego Dockerfile.
  - **volumes**: Montowanie katalogu klienta oraz certyfikatu CA do kontenera.
  - **networks**: Przyłączenie do sieci `fl_network`.
  - **depends_on**: Upewnienie się, że serwer jest uruchomiony przed klientami.
  - **command**: Komenda do uruchomienia klienta (`python client.py`).

- **networks**:
  - Definicja sieci `fl_network` używanej przez wszystkie usługi.

### **Dockerfile dla Serwera**

Plik `Dockerfile` dla serwera znajduje się w katalogu `server/` i definiuje, jak zbudować obraz Docker dla serwera FL.

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "server.py"]
```

#### **Wyjaśnienie Instrukcji:**

1. **FROM python:3.9-slim**: Bazowy obraz Dockera z Pythonem 3.9 na bazie lekkiej dystrybucji.
2. **WORKDIR /app**: Ustawienie katalogu roboczego na `/app`.
3. **COPY requirements.txt requirements.txt**: Skopiowanie pliku `requirements.txt` do katalogu roboczego.
4. **RUN pip install --no-cache-dir -r requirements.txt**: Instalacja zależności Pythona.
5. **COPY . .**: Skopiowanie całego kodu serwera do katalogu roboczego.
6. **EXPOSE 8080**: Otwarcie portu 8080.
7. **CMD ["python", "server.py"]**: Domyślna komenda do uruchomienia serwera.

### **Dockerfile dla Klientów**

Plik `Dockerfile` dla klientów znajduje się w katalogu `client/` i definiuje, jak zbudować obraz Docker dla klientów FL.

```dockerfile
FROM python:3.9-slim

WORKDIR /app/client

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "client.py"]
```

#### **Wyjaśnienie Instrukcji:**

1. **FROM python:3.9-slim**: Bazowy obraz Dockera z Pythonem 3.9 na bazie lekkiej dystrybucji.
2. **WORKDIR /app/client**: Ustawienie katalogu roboczego na `/app/client`.
3. **COPY requirements.txt requirements.txt**: Skopiowanie pliku `requirements.txt` do katalogu roboczego.
4. **RUN pip install --no-cache-dir -r requirements.txt**: Instalacja zależności Pythona.
5. **COPY . .**: Skopiowanie całego kodu klienta do katalogu roboczego.
6. **CMD ["python", "client.py"]**: Domyślna komenda do uruchomienia klienta.

### **server.py**

Plik `server.py` zawiera kod serwera Flower, który zarządza federowanym uczeniem.

```python
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
```

#### **Wyjaśnienie Kodowania:**

1. **Definicja Modelu**: Model `Net` jest prostą siecią neuronową z dwiema warstwami liniowymi, kompatybilną z klientami.

2. **Definicja Strategii `SaveModelStrategy`**:
    - **extend FedAvg**: Klasa `SaveModelStrategy` dziedziczy po `FedAvg`, implementując własną metodę `aggregate_fit`.
    - **aggregate_fit**: Metoda ta agreguje wyniki od klientów, konwertuje je na `ndarray`, ładuje je do modelu i zapisuje model po każdej rundzie treningowej.

3. **Uruchomienie Serwera**:
    - **Certyfikaty SSL**: Odczytanie certyfikatów SSL z zamontowanego katalogu `/certs/`.
    - **Start Serwera**: Użycie `fl.server.start_server` do uruchomienia serwera FL na porcie `8080` z konfiguracją strategii i certyfikatami SSL.

### **client.py**

Plik `client.py` zawiera kod klienta Flower, który uczestniczy w federowanym uczeniu.

```python
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
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=9)
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
```

#### **Wyjaśnienie Kodowania:**

1. **Definicja Modelu**: Model `MNISTModel` jest prostą siecią neuronową kompatybilną z PyTorch Lightning, używaną do klasyfikacji MNIST.

2. **Przygotowanie Danych**: Funkcja `load_data` pobiera i przygotowuje dane MNIST, używając transformacji `ToTensor` i tworząc `DataLoader` z 9 workerami.

3. **Definicja Klienta `FlowerClient`**:
    - **get_parameters**: Pobiera aktualne parametry modelu jako listę `ndarray`.
    - **set_parameters**: Ustawia parametry modelu na podstawie otrzymanych wartości od serwera.
    - **fit**: Trenuje model na lokalnych danych przez jedną epokę i zwraca zaktualizowane parametry oraz liczbę przykładów.
    - **evaluate**: Funkcja ewaluacji (opcjonalna).

4. **Uruchomienie Klienta**:
    - **Certyfikat CA**: Sprawdzenie istnienia certyfikatu CA i jego załadowanie.
    - **Start Klienta**: Użycie `fl.client.start_client` do połączenia z serwerem FL na adresie `server:8080` z wykorzystaniem certyfikatów SSL.

### **requirements.txt**

Plik `requirements.txt` zawiera listę zależności Pythona niezbędnych do uruchomienia serwera i klientów FL.

#### **server/requirements.txt**

```plaintext
flwr==1.12.0
torch
pytorch-lightning
torchvision
lightning[extra]
```

#### **client/requirements.txt**

```plaintext
flwr==1.12.0
torch
pytorch-lightning
torchvision
lightning[extra]
```

## **Monitorowanie i Logowanie**

Po uruchomieniu kontenerów możesz monitorować ich działanie poprzez logi. Użyj poniższej komendy, aby śledzić logi wszystkich usług:

```bash
docker-compose logs -f
```

### **Opis Logów:**

- **Serwer (`server-1`)**:
    - Informacje o uruchomieniu serwera FL.
    - Zbieranie i agregacja parametrów od klientów.
    - Zapisanie modelu po każdej rundzie treningowej.

- **Klienci (`client1-1`, `client2-1`)**:
    - Informacje o połączeniu z serwerem FL.
    - Odbieranie i trenowanie parametrów modelu.
    - Wysyłanie zaktualizowanych parametrów z powrotem do serwera.

## **Rozwiązywanie Problemów**

### **1. Błąd `AttributeError: 'tuple' object has no attribute 'tensors'`**

**Przyczyna:** W metodzie `aggregate_fit` klasy `SaveModelStrategy` próbujesz iterować po obiekcie `Parameters`, który jest krotką (`tuple`), a nie instancją `Parameters`.

**Rozwiązanie:** Zmodyfikuj metodę `aggregate_fit`, aby poprawnie rozpakować krotkę zwracaną przez `super().aggregate_fit(...)`.

```python
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
```

### **2. Ostrzeżenie dotyczące `version` w `docker-compose.yml`**

**Przyczyna:** Atrybut `version` jest przestarzały w nowszych wersjach Docker Compose.

**Rozwiązanie:** Usuń linię `version: '3.8'` z pliku `docker-compose.yml`.

```yaml
services:
  # ... pozostała konfiguracja
networks:
  fl_network:
    driver: bridge
```

### **3. Ostrzeżenia dotyczące `tensorboardX`**

**Przyczyna:** `pytorch_lightning` usuwa `tensorboardX` jako zależność, co powoduje ostrzeżenia.

**Rozwiązanie:** Zainstaluj dodatkowe zależności, aby wyeliminować ostrzeżenia.

1. **Dodaj `lightning[extra]` do `requirements.txt` klientów:**

    ```plaintext
    lightning[extra]
    ```

2. **Zbuduj ponownie obrazy Docker:**

    ```bash
    docker-compose build
    docker-compose up
    ```

### **4. Ostrzeżenie dotyczące `DataLoader`**

**Przyczyna:** `DataLoader` nie ma wystarczającej liczby workerów, co może wpływać na wydajność.

**Rozwiązanie:** Zwiększ liczbę workerów w `DataLoader` w pliku `client.py`.

```python
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=9)
    return train_loader
```

## **Dodatkowe Informacje**

### **Testowanie Połączenia SSL**

Aby upewnić się, że serwer Flower działa poprawnie i obsługuje SSL, możesz przetestować połączenie SSL za pomocą `openssl`:

```bash
openssl s_client -connect localhost:8080 -CAfile ./server/ca.crt
```

Powinieneś zobaczyć informacje o poprawnym handshake SSL, podobne do tego:

```
CONNECTED(00000003)
...
Verify return code: 0 (ok)
```

### **Sprawdzenie Certyfikatów w Kontenerach**

Upewnij się, że pliki certyfikatów są poprawnie zamontowane w kontenerach serwera i klientów.

1. **W Kontenerze Serwera:**

    ```bash
    docker exec -it flsimulation-server-1 /bin/sh
    ls -l /certs/
    ```

    Powinieneś zobaczyć:

    ```
    -rw-r--r-- 1 root root ... /certs/ca.crt
    -rw-r--r-- 1 root root ... /certs/server.crt
    -rw------- 1 root root ... /certs/server.key
    ```

2. **W Kontenerze Klienta:**

    ```bash
    docker exec -it flsimulation-client1-1 /bin/sh
    ls -l /certs/ca.crt
    ```

    Powinieneś zobaczyć:

    ```
    -rw-r--r-- 1 root root ... /certs/ca.crt
    ```

### **Sprawdzenie Sieci Docker**

Upewnij się, że wszystkie usługi są połączone z tą samą siecią Docker (`fl_network`), co pozwala klientom rozpoznawać serwer pod nazwą `server`.

```bash
docker network inspect fl_network
```

Powinieneś zobaczyć, że wszystkie kontenery są połączone z tą samą siecią.

### **Sprawdzenie Wersji Flower**

Upewnij się, że zarówno serwer, jak i klienci używają tej samej wersji Flower (1.12.0).

**Wewnątrz Kontenerów:**

```bash
docker exec -it flsimulation-server-1 /bin/sh
pip show flwr
```

Powinieneś zobaczyć:

```
Name: flwr
Version: 1.12.0
...
```

Jeśli wersje się różnią, zaktualizuj Flower:

```bash
pip install --upgrade flwr
```

### **Sprawdzenie Innych Procesów na Porcie `8080`**

Upewnij się, że na porcie `8080` nasłuchuje tylko serwer Flower:

```bash
sudo lsof -i :8080
```

Powinieneś zobaczyć tylko proces związany z serwerem Flower.

## **Licencja**

Ten projekt jest objęty licencją Apache License 2.0. Zobacz plik [LICENSE](LICENSE) w repozytorium, aby uzyskać więcej informacji.

---

Dziękujemy za skorzystanie z naszego projektu Federated Learning z Flower i Docker Compose. Jeśli masz pytania lub napotkasz problemy, prosimy o zgłoszenie ich poprzez system Issues na GitHubie.