# Se chegou aqui, é porque você está interessado em saber mais. Logo, de brinde, como rodar um código `Python` aqui

``` python exec="on" html="1"
--8<-- "./docs/roteiro4/limit.def.py"
```

``` python exec="on" html="1"
--8<-- "./docs/roteiro4/smc.py"
```

[Markdown-exec](https://pawamoy.github.io/markdown-exec/usage/){:target='_blank'} é uma extensão do [Markdown](https://www.markdownguide.org/){:target='_blank'} que permite executar código Python diretamente no Markdown. Isso é útil para gerar resultados dinâmicos ou executar scripts de forma interativa.

## Roteiro 1 - Data Preparation and Analysis for Neural Networks

This activity is designed to test your skills in generating synthetic datasets, handling real-world data challenges, and preparing data to be fed into neural networks.

## Excercise 1 - Exploring Class Separability in 2D

generate and visualize a two-dimensional dataset to explore how data distribution affects the complexity of the decision boundaries a neural network would need to learn.

### Generate the Data

Start by importing necessary libraries for this project:

<!-- termynal -->

``` bash
pip install matplotlib pandas scikit-learn numpy
```

Create a synthetic dataset with a total of 400 samples, divided equally among 4 classes (100 samples each). Use a Gaussian distribution to generate the points for each class based on the following parameters:

- Class 0: Mean = [2,3] , Standard Deviation = [0.8,2.5]
- Class 1: Mean = [5,6], Standard Deviation = [1.2,1.9]
- Class 2: Mean = [8,1], Standard Deviation = [0.9,0.9]
- Class 3: Mean = [15,4], Standard Deviation = [0.5,2.0]

``` pyodide install="pandas,matplotlib,scikit-learn,numpy"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

params = {
    0: {"mean": np.array([2.0, 3.0]), "std": np.array([0.8, 2.5])},
    1: {"mean": np.array([5.0, 6.0]), "std": np.array([1.2, 1.9])},
    2: {"mean": np.array([8.0, 1.0] ), "std": np.array([0.9, 0.9])},
    3: {"mean": np.array([15.0, 4.0]), "std": np.array([0.5, 2.0])},
}

n_samples_class = 100

data_list = []
for cls, p in params.items():
    samples = rng.normal(loc=p["mean"], scale=p["std"], size=(n_samples_class, 2))
    labels = np.full((n_samples_class, 1), cls, dtype=int)
    data_list.append(np.hstack([samples, labels]))

# Combine
data = np.vstack(data_list)
df = pd.DataFrame(data, columns=["x1", "x2", "label"]).astype({"label": int})

# Shuffle rows
df = df.sample(frac=1.0, random_state=123).reset_index(drop=True)

# Save to CSV
csv_path = "synthetic_gaussian_4class_400.csv"
df.to_csv(csv_path, index=False)

# Show class counts
counts = df["label"].value_counts().sort_index()

# Display a preview to the user
df.head(500)

```

![Tela do Dashboard do MAAS](./maas.png)
/// caption
Dashboard do MAAS
///

Conforme ilustrado acima, a tela inicial do MAAS apresenta um dashboard com informações sobre o estado atual dos servidores gerenciados. O dashboard é composto por diversos painéis, cada um exibindo informações sobre um aspecto específico do ambiente gerenciado. Os painéis podem ser configurados e personalizados de acordo com as necessidades do usuário.

### Tarefa 2

## App

### Tarefa 1

Exemplo de diagrama

```mermaid
architecture-beta
    group api(cloud)[API]

    service db(database)[Database] in api
    service disk1(disk)[Storage] in api
    service disk2(disk)[Storage] in api
    service server(server)[Server] in api

    db:L -- R:server
    disk1:T -- B:server
    disk2:T -- B:db
```

[Mermaid](https://mermaid.js.org/syntax/architecture.html){:target="_blank"}

## Questionário, Projeto ou Plano

Esse seção deve ser preenchida apenas se houver demanda do roteiro.

## Discussões

Quais as dificuldades encontradas? O que foi mais fácil? O que foi mais difícil?

## Conclusão

O que foi possível concluir com a realização do roteiro?

### Diagrama de Classes do Banco

``` mermaid
classDiagram
    class Conta {
        - String id
        # double saldo
        - Cliente cliente
        + sacar(double valor)
        + depositar(double valor)
    }
    class Cliente {
        - String id
        - String nome
        - List<Conta> contas
    }
    class PessoaFisica {
        - String cpf
    }
    class PessoaJuridica {
        - String cnpj
    }
    class ContaCorrente {
        - double limite
        + sacar(double valor)
    }
    class ContaPoupanca {
        + sacar(double valor)
    }
    Conta *-- Cliente
    Conta <|-- ContaCorrente
    Conta <|-- ContaPoupanca
    Cliente <|-- PessoaFisica
    Cliente <|-- PessoaJuridica
```

### Diagrama de Seqüência de Autorização

``` mermaid
sequenceDiagram
  autonumber
  actor User
  User->>Auth Service: request with token
  Auth Service->>Auth Service: decodes the token and extracts claims
  Auth Service->>Auth Service: verifies permissions
  critical allowed
    Auth Service->>Secured Resource: authorizes the request
    Secured Resource->>User: returns the response
  option denied
    Auth Service-->>User: unauthorized message
  end  
```

Running the code below in Browser (Woooooowwwwww!!!!!!). [^1]

``` pyodide install="pandas,ssl"
import ssl
import pandas as pd

df = pd.DataFrame()
df['AAPL'] = pd.Series([1, 2, 3])
df['MSFT'] = pd.Series([4, 5, 6])
df['GOOGL'] = pd.Series([7, 8, 9])

print(df)

```

[^1]: [Pyodide](https://pawamoy.github.io/markdown-exec/usage/pyodide/){target="_blank"}
