# Treinamento e Validação no MNIST com PyTorch Lightning (SuperLight)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aj1no/DL_Atividade_06_MNIST_Lightning_Lite/blob/main/06_01_Treino_Validacao_MNIST_Lightning_Lite.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/vision/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Este repositório contém a implementação e resolução prática do exercício de **Classificação de Dígitos Manuscritos (MNIST)** utilizando **Gradiente Descendente Estocástico (SGD) por Minibatches** e a arquitetura modular didática **SuperLight** (inspirada nos princípios de design do *PyTorch Lightning*).

---

## Informações Acadêmicas

* **Curso:** Ciência de Dados
* **Disciplina:** Aprendizado Profundo / Deep Learning
* **Autor:** Rodolfo Vinicius Cima Takemoto ([@aj1no](https://github.com/aj1no))

---

## Objetivos do Projeto

1. **Otimização por Minibatches:** Compreender a dinâmica do Gradiente Descendente Estocástico (SGD) operando em lotes de 50 amostras.
2. **Abstrações do PyTorch:** Empregar as classes fundamentais `torch.utils.data.Dataset`, `DataLoader` e `torch.nn.Module`.
3. **Padrão Arquitetural Lightning (SuperLight):** Desacoplar a modelagem matemática (`LightningModule`) da engenharia de execução e laços de treino/validação/teste (`Trainer`).
4. **Checkpointing e Prevenção de Overfitting:** Monitorar a perda de validação (*Validation Loss*) a cada época e persistir automaticamente os melhores pesos (`best_model.pt`).
5. **Avaliação sob Escassez de Dados:** Avaliar a capacidade de generalização da rede treinada com um subconjunto estrito de apenas **1.000 amostras** de treinamento (1,67% da base total do MNIST).

---

## Arquitetura do Modelo

A rede neural implementada consiste em um Perceptron Multicamadas (**MLP**):

```text
Entrada: Imagem MNIST 28x28 (784 dimensões linearizadas)
   │
   ▼
[Camada Linear: 784 -> 500 neurônios]
   │
   ▼
[Ativação Não-Linear: ReLU]
   │
   ▼
[Camada Linear: 500 -> 10 neurônios (Logits)]
   │
   ▼
Saída: 10 Classes (Dígitos 0 a 9 via CrossEntropyLoss)
```

* **Total de Parâmetros Treináveis:** $784 \times 500 + 500 + 500 \times 10 + 10 = \mathbf{397.510}$ parâmetros.
* **Função de Custo:** `CrossEntropyLoss` (combina `LogSoftmax` e `NLLLoss`).
* **Otimizador:** `SGD` (Taxa de Aprendizado $\eta = 0.1$).

---

## Estrutura do Framework SuperLight

```text
┌─────────────────────────────────────────────────────────────┐
│                    LightningModule                          │
│ ├─ forward(x)                                               │
│ ├─ training_step(batch, batch_idx)                          │
│ ├─ training_epoch_end(outputs)                             │
│ ├─ validation_step(batch, batch_idx)                        │
│ ├─ validation_epoch_end(outputs)                            │
│ ├─ test_step(batch, batch_idx)                              │
│ ├─ test_epoch_end(outputs)                                  │
│ └─ configure_optimizers()                                   │
└──────────────────────────────┬──────────────────────────────┘
                               │ orquestrado por
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                         Trainer                             │
│ ├─ fit(model, train_dataloader, val_dataloader)             │
│ │   ├─ Training Loop (ZeroGrad -> Backward -> Step)         │
│ │   ├─ Validation Loop (Avaliação sem cálculo de gradientes)│
│ │   └─ Auto-Checkpointing (Salva 'best_model.pt')           │
│ └─ test(model, test_dataloader)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Resultados Experimentais

O modelo foi treinado por **20 épocas** com batch size 50 em **1.000 amostras** de treino e avaliado em **1.000 amostras** de teste isoladas:

### Resumo das Métricas

| Etapa / Conjunto | Loss (Cross-Entropy) | Acurácia | Detalhes |
| :--- | :---: | :---: | :--- |
| **Treinamento** (20 épocas) | $1.7897 \rightarrow 0.1842$ | — | 1.000 amostras (SGD, $\eta=0.1$, batch=50) |
| **Validação** (Melhor modelo) | 0.4199 | 87.50% | Checkpoint automático salvo (`best_model.pt`) |
| **Teste** (Avaliação Final) | **0.4160** | **87.60%** | 1.000 amostras de teste isoladas |

### Relatório Detalhado por Classe no Conjunto de Teste

```text
              precision    recall  f1-score   support

           0     0.9429    0.9612    0.9519       103
           1     0.9449    0.9917    0.9677       121
           2     0.8878    0.8788    0.8832        99
           3     0.8529    0.8614    0.8571       101
           4     0.8252    0.8673    0.8458        98
           5     0.8351    0.8804    0.8571        92
           6     0.9208    0.9490    0.9347        98
           7     0.9167    0.8491    0.8817       106
           8     0.7957    0.8222    0.8087        90
           9     0.8242    0.8152    0.8197        92

    accuracy                         0.8760      1000
   macro avg     0.8746    0.8776    0.8758      1000
weighted avg     0.8773    0.8760    0.8763      1000
```

---

## Principais Conclusões e Aprendizados

1. **Eficiência da Amostragem por Minibatches:** O uso de $B=50$ permitiu passos de gradiente estáveis e convergência rápida em poucas épocas, evitando a lentidão do Batch Gradient Descent e a instabilidade excessiva do SGD com batch unitário ($B=1$).
2. **Alta Capacidade de Generalização com Poucos Dados:** Atingir **87,60%** com apenas 1.000 instâncias de treino demonstra a eficácia do MLP com ativação ReLU em extrair fronteiras de decisão discriminantes sobre o espaço dos pixels.
3. **Clareza de Código e Boas Práticas:** A modularização estilo *PyTorch Lightning* elimina código duplicado (*boilerplate*), separa claramente o pipeline de dados da definição do modelo e garante execução robusta tanto em CPU quanto em GPU (CUDA).

---

## Ferramentas Utilizadas

Em conformidade com a transparência acadêmica:

* **Ambiente de Desenvolvimento:** Google Colaboratory / Antigravity IDE (Python 3.14 / Jupyter Kernel).
* **Bibliotecas Principais:** `torch` & `torchvision` (construção e treinamento da rede neural), `scikit-learn` (métricas de avaliação, relatório de classificação e matriz de confusão), `numpy` (manipulação de tensores e arrays), `matplotlib` & `seaborn` (geração dos gráficos de perda, acurácia e inspeção visual de inferência).
* **Assistência de IA (Antigravity / Gemini 3.7 Flash High):** Apoio na formulação dos gráficos comparativos, estruturação da classe modular SuperLight e documentação técnica em Markdown.

---

## Como Executar

1. **Clone este repositório:**
   ```bash
   git clone https://github.com/aj1no/DL_Atividade_06_MNIST_Lightning_Lite.git
   cd DL_Atividade_06_MNIST_Lightning_Lite
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python -m venv venv
   # No Windows:
   .\venv\Scripts\activate
   # No Linux/macOS:
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Inicie o Jupyter Notebook ou abra no Google Colab:**
   ```bash
   jupyter notebook 06_01_Treino_Validacao_MNIST_Lightning_Lite.ipynb
   ```

---

## Licença

Este projeto está sob a licença [MIT](LICENSE).
