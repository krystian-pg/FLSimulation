# FLSimulation

---

## **Spis treści**

1. **Wymagania wstępne**
2. **Ogólny zarys architektury**
3. **Konfiguracja środowiska**
4. **Przygotowanie kodu aplikacji**
   - Serwer Flower
   - Klient Flower
5. **Tworzenie obrazów Dockera**
   - Dockerfile dla serwera
   - Dockerfile dla klienta
6. **Konfiguracja Docker Compose**
7. **Integracja monitoringu z Grafaną i Prometheusem**
   - Konfiguracja Prometheusa
   - Konfiguracja Grafany
8. **Uruchomienie symulacji**
9. **Monitorowanie sieci federacyjnej**
10. **Podsumowanie**

---

## **1. Wymagania wstępne**

Przed rozpoczęciem upewnij się, że masz zainstalowane następujące narzędzia:

- **Docker**: do konteneryzacji aplikacji.
- **Docker Compose**: do zarządzania wieloma kontenerami Dockera.
- **Python 3.8+**: do uruchamiania kodu lokalnie (opcjonalnie).
- **Git**: do klonowania repozytoriów (opcjonalnie).

---

## **2. Ogólny zarys architektury**

- **Serwer Flower**: Koordynuje proces federacyjnego uczenia maszynowego.
- **Klienci Flower**: Uruchamiają trening na lokalnych danych i komunikują się z serwerem.
- **Prometheus**: Zbiera metryki z serwera i klientów.
- **Grafana**: Wizualizuje zebrane metryki, umożliwiając monitorowanie stanu sieci.

---

## **3. Konfiguracja środowiska**

### **Instalacja Dockera i Docker Compose**

Jeśli nie masz jeszcze zainstalowanego Dockera i Docker Compose, możesz je pobrać z oficjalnej strony:

- **Docker**: [Instrukcja instalacji](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Instrukcja instalacji](https://docs.docker.com/compose/install/)

Upewnij się, że Docker działa poprawnie, uruchamiając:

```bash
docker run hello-world
```

### **Struktura projektu**

Stwórzmy katalog projektu i ustalmy jego strukturę:

```
federated-learning-project/
├── server/
│   ├── server.py
│   └── Dockerfile
├── client/
│   ├── client.py
│   └── Dockerfile
├── docker-compose.yml
├── prometheus/
│   └── prometheus.yml
└── grafana/
```

---

## **4. Przygotowanie kodu aplikacji**

### **4.1. Serwer Flower**

**Plik:** `server/server.py`

```python
import flwr as fl

def main():
    # Definiujemy strategię agregacji
    strategy = fl.server.strategy.FedAvg()

    # Uruchamiamy serwer Flower
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 3},
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
```

**Wyjaśnienie:**

- Importujemy bibliotekę Flower.
- Definiujemy główną funkcję `main`, w której:
  - Ustawiamy strategię federacyjną, np. FedAvg (średnia ważona wag modeli).
  - Uruchamiamy serwer na adresie `0.0.0.0:8080`, aby był dostępny dla kontenerów klientów.

### **4.2. Klient Flower**

**Plik:** `client/client.py`

```python
import flwr as fl
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch.nn.functional as F

# Definicja modelu
class MNISTModel(pl.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
    
    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Przygotowanie danych
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='/data', train=True, download=True, transform=transform)
    train_dataset, _ = random_split(mnist_train, [5000, len(mnist_train) - 5000])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    return train_loader

# Definicja klienta Flower
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False)
        trainer.fit(self.model, self.train_loader)
        return self.get_parameters(), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        # Opcjonalnie implementacja ewaluacji
        return 0.0, len(self.train_loader.dataset), {}

def main():
    # Inicjalizacja modelu i danych
    model = MNISTModel()
    train_loader = load_data()

    # Inicjalizacja klienta Flower
    client = FlowerClient(model, train_loader)

    # Uruchomienie klienta
    fl.client.start_numpy_client(server_address="server:8080", client=client)

if __name__ == "__main__":
    main()
```

**Wyjaśnienie:**

- Definiujemy model MNIST z użyciem PyTorch Lightning.
- Przygotowujemy dane treningowe (subset MNIST).
- Implementujemy klasę `FlowerClient`, która dziedziczy po `fl.client.NumPyClient`.
- W `main()` uruchamiamy klienta, łącząc się z serwerem o nazwie `server` (co zostanie zdefiniowane w Docker Compose).

---

## **5. Tworzenie obrazów Dockera**

### **5.1. Dockerfile dla serwera**

**Plik:** `server/Dockerfile`

```Dockerfile
# Używamy oficjalnego obrazu Python
FROM python:3.9-slim

# Ustawiamy katalog roboczy
WORKDIR /app

# Kopiujemy pliki serwera
COPY server.py /app/server.py

# Instalujemy zależności
RUN pip install flwr

# Eksponujemy port serwera
EXPOSE 8080

# Uruchamiamy serwer
CMD ["python", "server.py"]
```

**Wyjaśnienie:**

- Bazujemy na lekkim obrazie `python:3.9-slim`.
- Ustawiamy katalog roboczy `/app`.
- Kopiujemy plik `server.py`.
- Instalujemy potrzebne pakiety (w tym przypadku tylko `flwr`).
- Eksponujemy port `8080`, na którym działa serwer Flower.
- Definiujemy polecenie startowe.

### **5.2. Dockerfile dla klienta**

**Plik:** `client/Dockerfile`

```Dockerfile
# Używamy oficjalnego obrazu Python
FROM python:3.9-slim

# Ustawiamy katalog roboczy
WORKDIR /app

# Kopiujemy pliki klienta
COPY client.py /app/client.py

# Instalujemy zależności
RUN pip install flwr torch torchvision pytorch-lightning

# Upewniamy się, że katalog danych istnieje
RUN mkdir -p /data

# Uruchamiamy klienta
CMD ["python", "client.py"]
```

**Wyjaśnienie:**

- Podobnie jak w przypadku serwera, używamy obrazu `python:3.9-slim`.
- Kopiujemy `client.py`.
- Instalujemy wszystkie potrzebne pakiety.
- Tworzymy katalog `/data` dla danych MNIST.
- Definiujemy polecenie startowe.

---

## **6. Konfiguracja Docker Compose**

**Plik:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  server:
    build: ./server
    container_name: server
    ports:
      - "8080:8080"
    networks:
      - fl-network

  client1:
    build: ./client
    depends_on:
      - server
    networks:
      - fl-network

  client2:
    build: ./client
    depends_on:
      - server
    networks:
      - fl-network

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    networks:
      - fl-network

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    networks:
      - fl-network

networks:
  fl-network:
    driver: bridge
```

**Wyjaśnienie:**

- Definiujemy usługi:
  - **server**: buduje obraz z katalogu `./server`, eksponuje port `8080`.
  - **client1** i **client2**: dwa instancje klienta, budowane z `./client`, zależne od serwera.
  - **prometheus**: używa oficjalnego obrazu Prometheusa, ładuje konfigurację z lokalnego pliku.
  - **grafana**: używa oficjalnego obrazu Grafany.
- Wszystkie usługi są połączone w sieci `fl-network`, co umożliwia komunikację między nimi.

---

## **7. Integracja monitoringu z Grafaną i Prometheusem**

### **7.1. Konfiguracja Prometheusa**

**Plik:** `prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'flower_server'
    static_configs:
      - targets: ['server:8080']
  - job_name: 'flower_clients'
    static_configs:
      - targets: ['client1:8080', 'client2:8080']
```

**Wyjaśnienie:**

- Ustawiamy globalny interwał scrapowania na 5 sekund.
- Definiujemy dwa zadania:
  - **flower_server**: monitoruje serwer Flower.
  - **flower_clients**: monitoruje klientów Flower.
- Cele (targets) to nazwy usług zdefiniowanych w Docker Compose.

### **7.2. Konfiguracja Grafany**

- Po uruchomieniu Grafany, będziesz mógł uzyskać do niej dostęp pod adresem `http://localhost:3000`.
- Domyślne dane logowania to **admin/admin**.
- Po zalogowaniu należy:
  - Dodać źródło danych Prometheus, wskazując na `http://prometheus:9090`.
  - Importować gotowe dashboardy lub stworzyć własne.

**Uwaga:** Pełna konfiguracja Grafany wymaga interakcji z interfejsem webowym, więc nie jest w całości możliwa do zautomatyzowania w plikach konfiguracyjnych.

---

## **8. Uruchomienie symulacji**

Przejdź do katalogu głównego projektu i uruchom Docker Compose:

```bash
docker-compose up --build
```

**Wyjaśnienie:**

- **`--build`**: powoduje odbudowanie obrazów Dockera, jeśli zaszły zmiany.
- Docker Compose uruchomi wszystkie usługi zgodnie z konfiguracją.

---

## **9. Monitorowanie sieci federacyjnej**

### **9.1. Dostęp do Grafany**

- Otwórz przeglądarkę i wejdź na `http://localhost:3000`.
- Zaloguj się (domyślnie **admin/admin**).
- Dodaj źródło danych:
  - Przejdź do **Configuration** -> **Data Sources**.
  - Wybierz **Prometheus**.
  - Ustaw URL na `http://prometheus:9090`.
  - Zapisz.

### **9.2. Tworzenie dashboardu**

- Możesz teraz tworzyć własne dashboardy, dodając panele z metrykami z Prometheusa.
- Możesz monitorować takie metryki jak:
  - Liczba iteracji treningu.
  - Czas trwania rund.
  - Zużycie zasobów (CPU, RAM).

**Uwaga:** Aby monitorować specyficzne metryki z Flower, musielibyśmy zmodyfikować kod serwera i klientów, aby eksportować metryki w formacie zrozumiałym dla Prometheusa.

---

## **10. Podsumowanie**

Stworzyliśmy symulację federacyjnego uczenia maszynowego z użyciem Flower i PyTorch Lightning, uruchomioną w kontenerach Dockerowych. Użyliśmy Docker Compose do zarządzania wieloma kontenerami i zintegrowaliśmy monitoring z Prometheusem i Grafaną.

---

## **Dodatkowe wyjaśnienia i rozszerzenia**

### **Eksportowanie metryk z Flower do Prometheusa**

Aby eksportować metryki z aplikacji do Prometheusa, możemy użyć biblioteki **prometheus_client**.

**Instalacja:**

W plikach `Dockerfile` dodaj:

```Dockerfile
RUN pip install prometheus_client
```

**Modyfikacja kodu serwera:**

W `server/server.py`, zaimportuj biblioteki i rozpocznij eksportowanie metryk:

```python
from prometheus_client import start_http_server, Counter

# Definiujemy licznik dla liczby rund
ROUND_COUNTER = Counter('fl_rounds_total', 'Total number of federated learning rounds')

def main():
    # Startujemy serwer metryk Prometheusa
    start_http_server(8000)

    # Definiujemy strategię z wywołaniem zwrotnym po każdej rundzie
    class StrategyWithMetrics(fl.server.strategy.FedAvg):
        def aggregate_fit(self, rnd, results, failures):
            # Inkrementujemy licznik rund
            ROUND_COUNTER.inc()
            return super().aggregate_fit(rnd, results, failures)

    strategy = StrategyWithMetrics()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 3},
        strategy=strategy,
    )
```

**Wyjaśnienie:**

- Używamy `prometheus_client` do eksportowania metryk.
- Uruchamiamy serwer metryk na porcie `8000`.
- Tworzymy licznik `ROUND_COUNTER` i inkrementujemy go po każdej rundzie.

**Aktualizacja konfiguracji Prometheusa:**

Dodaj port `8000` do celów monitorowania serwera:

```yaml
scrape_configs:
  - job_name: 'flower_server'
    static_configs:
      - targets: ['server:8000']
```

**Modyfikacja kodu klienta:**

Podobnie możemy eksportować metryki z klienta, np. liczba epok treningu.

---

## **Uwagi końcowe**

- **Skalowalność**: Możesz dodać więcej klientów, kopiując sekcję klienta w `docker-compose.yml` i zmieniając nazwy usług.
- **Dane**: W obecnej konfiguracji każdy klient pobiera te same dane MNIST. W rzeczywistym scenariuszu federacyjnego uczenia klienci powinni mieć różne, nieudostępniane między sobą dane.
- **Bezpieczeństwo**: W produkcyjnych zastosowaniach należy zadbać o bezpieczeństwo komunikacji (np. TLS) i ochronę danych.

Jeśli masz pytania lub potrzebujesz dalszych wyjaśnień na którymkolwiek etapie, daj znać!
