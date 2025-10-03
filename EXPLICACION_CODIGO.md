# Explicación Detallada del Código - TiDE Implementation

Este documento explica cada bloque de código del notebook `laboratorio2_tide.ipynb`, incluyendo la lógica implementada, conceptos de deep learning involucrados, librerías utilizadas y referencias útiles.

---

## Tabla de Contenidos

1. [Configuración Inicial y Librerías](#1-configuración-inicial-y-librerías)
2. [Configuración del Experimento](#2-configuración-del-experimento)
3. [Descarga y Carga de Datos](#3-descarga-y-carga-de-datos)
4. [Preprocesamiento: Features Temporales](#4-preprocesamiento-features-temporales)
5. [Reversible Instance Normalization (RevIN)](#5-reversible-instance-normalization-revin)
6. [Dataset para Series de Tiempo](#6-dataset-para-series-de-tiempo)
7. [Bloque Residual (ResidualBlock)](#7-bloque-residual-residualblock)
8. [Modelo TiDE Completo](#8-modelo-tide-completo)
9. [Funciones de Entrenamiento](#9-funciones-de-entrenamiento)
10. [Pipeline de Experimentación](#10-pipeline-de-experimentación)
11. [Visualizaciones y Análisis](#11-visualizaciones-y-análisis)

---

## 1. Configuración Inicial y Librerías

### Código
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```

### Explicación

**Librerías utilizadas:**
- **NumPy** ([docs](https://numpy.org/doc/)): Operaciones numéricas y arrays
- **Pandas** ([docs](https://pandas.pydata.org/docs/)): Manipulación de datos tabulares
- **Matplotlib/Seaborn** ([matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)): Visualización
- **PyTorch** ([docs](https://pytorch.org/docs/)): Framework de deep learning
- **tqdm** ([docs](https://tqdm.github.io/)): Barras de progreso

**Conceptos de Deep Learning:**
- **Device selection**: Soporta CUDA (NVIDIA GPUs), MPS (Apple Silicon), y CPU
- **Reproducibilidad**: Seeds fijas para resultados determinísticos

**Referencias:**
- [PyTorch Device Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [Reproducibility in PyTorch](https://pytorch.org/docs/stable/notes/randomness.html)

---

## 2. Configuración del Experimento

### Código
```python
CONFIG = {
    'datasets': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'],
    'look_back': 96,
    'horizons': [24, 48, 96, 192, 336, 720],
    'target': 'OT',
    'train_ratio': 0.7,
    'val_ratio': 0.1,
    'test_ratio': 0.2,
}

HPARAMS = {
    'ETTh1': {
        'hidden_size': 256,
        'num_encoder_layers': 2,
        ...
    },
    ...
}
```

### Explicación

**Configuración del experimento:**
- **look_back**: Ventana de contexto (96 time-steps)
- **horizons**: Múltiples horizontes de predicción para evaluar capacidad de pronóstico a corto, mediano y largo plazo
- **target**: Variable objetivo (OT = Oil Temperature)
- **Splits temporales**: 70% train, 10% val, 20% test (crucial para series de tiempo)

**Conceptos de Deep Learning:**
- **Temporal splitting**: En series de tiempo NO se usa shuffle; el tiempo fluye hacia adelante
- **Hiperparámetros específicos por dataset**: Cada dataset tiene características únicas que requieren configuraciones diferentes

**Referencias:**
- [Time Series Forecasting Best Practices](https://otexts.com/fpp3/)
- Paper TiDE Table 7-8 para hiperparámetros óptimos

---

## 3. Descarga y Carga de Datos

### Código
```python
def download_ett_data():
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    base_url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/'
    datasets = ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv']

    for dataset in datasets:
        file_path = data_dir / dataset
        if not file_path.exists():
            urllib.request.urlretrieve(url, file_path)
```

### Explicación

**Datasets ETT (Electricity Transformer Temperature):**
- **ETTh1/ETTh2**: Datos horarios (17,420 puntos)
- **ETTm1/ETTm2**: Datos cada 15 minutos (69,680 puntos)
- **Variables**: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT

**Conceptos:**
- **Multivariate Time Series**: 7 variables correlacionadas
- **Different Frequencies**: Permite evaluar el modelo en granularidades temporales distintas

**Referencias:**
- [ETT Dataset Paper](https://arxiv.org/abs/2012.07436)
- [ETT GitHub Repository](https://github.com/zhouhaoyi/ETDataset)

---

## 4. Preprocesamiento: Features Temporales

### Código
```python
def create_time_features(df):
    df['date'] = pd.to_datetime(df['date'])

    df['minute'] = ((df['date'].dt.minute / 59.0) - 0.5).astype(np.float32)
    df['hour'] = ((df['date'].dt.hour / 23.0) - 0.5).astype(np.float32)
    df['dayofweek'] = ((df['date'].dt.dayofweek / 6.0) - 0.5).astype(np.float32)
    df['day'] = ((df['date'].dt.day / 31.0) - 0.5).astype(np.float32)
    df['dayofyear'] = ((df['date'].dt.dayofyear / 365.0) - 0.5).astype(np.float32)
    df['month'] = ((df['date'].dt.month / 12.0) - 0.5).astype(np.float32)
    df['weekofyear'] = ((week_values / 52.0) - 0.5).astype(np.float32)
    df['is_weekend'] = ((df['date'].dt.dayofweek >= 5).astype(float) - 0.5).astype(np.float32)
```

### Explicación

**Lógica:**
1. Extraer componentes temporales de la fecha
2. Normalizar cada feature a rango [-0.5, 0.5]
3. Total de 8 features temporales

**Conceptos de Deep Learning:**
- **Temporal Covariates**: Features conocidas tanto en el pasado como en el futuro
- **Feature Scaling**: Normalización centrada en 0 para estabilidad numérica
- **Seasonal Patterns**: Capturan ciclos diarios, semanales, mensuales, anuales

**¿Por qué [-0.5, 0.5]?**
- Centrado en 0: Mejor para gradientes
- Rango simétrico: Previene bias hacia valores positivos
- Consistente con el paper TiDE

**Referencias:**
- [Feature Engineering for Time Series](https://machinelearningmastery.com/basic-feature-engineering-time-series/)
- [Temporal Features in Forecasting](https://otexts.com/fpp3/useful-predictors.html)

---

## 5. Reversible Instance Normalization (RevIN)

### Código
```python
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        x = x * self.std + self.mean
        return x
```

### Explicación

**Lógica:**
1. **Normalización**: Calcular mean/std del input, normalizar
2. **Transformación afín (opcional)**: Escala y shift aprendibles
3. **Denormalización**: Revertir la normalización en las predicciones

**Conceptos de Deep Learning:**
- **Instance Normalization**: Normaliza por instancia (cada serie de tiempo), no por batch
- **Reversibilidad**: Permite restaurar la escala original de las predicciones
- **Distribution Shift**: Ayuda cuando las distribuciones cambian entre train/test

**¿Cuándo usar RevIN?**
- ✅ Útil para: Series con tendencias no estacionarias
- ❌ No siempre mejora: Depende del dataset (ver HPARAMS)

**Matemática:**
```
Normalización:   x_norm = (x - μ) / σ
Afín:            x_affine = γ * x_norm + β  (γ, β aprendibles)
Denormalización: x_original = (x_affine - β) / γ * σ + μ
```

**Referencias:**
- [RevIN Paper](https://openreview.net/forum?id=cGDAkQo1C0p)
- [Instance Normalization](https://arxiv.org/abs/1607.08022)

---

## 6. Dataset para Series de Tiempo

### Código
```python
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_col, time_features, look_back, horizon,
                 scaler_mean=None, scaler_std=None):
        # Normalización
        if scaler_mean is None:
            self.scaler_mean = data[target_col].mean()
            self.scaler_std = data[target_col].std()
        else:
            self.scaler_mean = scaler_mean
            self.scaler_std = scaler_std

        self.data_norm = data.copy()
        self.data_norm[target_col] = (data[target_col] - self.scaler_mean) / self.scaler_std

    def __getitem__(self, idx):
        lookback_start = idx
        lookback_end = idx + self.look_back
        horizon_start = lookback_end
        horizon_end = horizon_start + self.horizon

        y_past = torch.FloatTensor(
            self.data_norm[self.target_col].iloc[lookback_start:lookback_end].values
        )
        y_future = torch.FloatTensor(
            self.data_norm[self.target_col].iloc[horizon_start:horizon_end].values
        )
        x_past = torch.FloatTensor(
            self.data_norm[self.time_features].iloc[lookback_start:lookback_end].values
        )
        x_future = torch.FloatTensor(
            self.data_norm[self.time_features].iloc[horizon_start:horizon_end].values
        )

        return {
            'y_past': y_past,
            'y_future': y_future,
            'x_past': x_past,
            'x_future': x_future
        }
```

### Explicación

**Lógica:**
1. **Sliding Window**: Crea ventanas deslizantes de tamaño `look_back + horizon`
2. **Normalización consistente**: Usa estadísticas del train set para val/test
3. **Retorna 4 tensores**:
   - `y_past`: Valores históricos del target (look_back)
   - `y_future`: Valores futuros a predecir (horizon)
   - `x_past`: Features temporales pasadas (look_back)
   - `x_future`: Features temporales futuras (horizon) - **conocidas de antemano**

**Conceptos de Deep Learning:**
- **Supervised Learning Setup**: Input (past) → Output (future)
- **Data Leakage Prevention**: Normalización solo con estadísticas de train
- **Autoregressive Features**: El modelo ve tanto el pasado como covariables futuras conocidas

**Diagrama:**
```
Timeline:  [------- look_back -------][------- horizon -------]
           t-96                    t  t+1                   t+H

y_past:    [y_{t-95}, ..., y_t]
y_future:                           [y_{t+1}, ..., y_{t+H}]
x_past:    [x_{t-95}, ..., x_t]
x_future:                           [x_{t+1}, ..., x_{t+H}]
```

**Referencias:**
- [PyTorch Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

---

## 7. Bloque Residual (ResidualBlock)

### Código
```python
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, use_layer_norm=True):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)

        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = self.dropout(self.linear2(out))
        out = out + self.skip(x)

        if self.use_layer_norm:
            out = self.layer_norm(out)

        return out
```

### Explicación

**Lógica:**
1. **MLP Path**: Linear → ReLU → Dropout → Linear
2. **Skip Connection**: Suma el input directamente al output
3. **Layer Normalization**: Estabiliza entrenamiento

**Conceptos de Deep Learning:**

**1. Residual Connections (Skip Connections):**
- Inventadas en ResNet (2015)
- Mitigan el problema de vanishing gradients
- Permiten entrenar redes más profundas
- Ecuación: `output = F(x) + x`

**2. Dropout:**
- Regularización: Apaga neuronas aleatoriamente durante entrenamiento
- Previene overfitting
- Típicamente 0.1-0.5

**3. Layer Normalization:**
- Normaliza activaciones por capa
- Estabiliza entrenamiento
- Diferencia con Batch Norm: normaliza por features, no por batch

**4. ReLU Activation:**
- `ReLU(x) = max(0, x)`
- No linealidad más común
- Evita vanishing gradient (vs sigmoid/tanh)

**Matemática:**
```
x ∈ ℝ^{input_dim}
h = ReLU(W₁x + b₁)           # hidden_dim
z = W₂h + b₂                  # output_dim
z = Dropout(z)
out = LayerNorm(z + Wₛx)     # Wₛ es proyección si dims no coinciden
```

**Referencias:**
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html)

---

## 8. Modelo TiDE Completo

### Código (Simplificado)
```python
class TiDEModel(nn.Module):
    def __init__(self, look_back, horizon, num_covariates, ...):
        # 1. Feature Projection
        self.feature_projection = ResidualBlock(
            num_covariates, hidden_size, temporal_width
        )

        # 2. Dense Encoder
        encoder_input_dim = look_back + (look_back + horizon) * temporal_width
        self.encoder = nn.Sequential(*encoder_layers)

        # 3. Dense Decoder
        self.decoder = nn.Sequential(*decoder_layers)

        # 4. Temporal Decoder
        self.temporal_decoder = ResidualBlock(
            decoder_output_dim + temporal_width, temporal_decoder_hidden, 1
        )

        # 5. Global Residual
        self.global_residual = nn.Linear(look_back, horizon)

    def forward(self, y_past, x_past, x_future):
        # 1. Project covariates
        x_past_proj = self.feature_projection(x_past)
        x_future_proj = self.feature_projection(x_future)

        # 2. Encode
        encoder_input = torch.cat([y_past, x_past_proj, x_future_proj], dim=1)
        encoding = self.encoder(encoder_input)

        # 3. Decode
        decoder_output = self.decoder(encoding)
        decoder_output = decoder_output.reshape(batch_size, horizon, decoder_output_dim)

        # 4. Temporal decode
        predictions = []
        for t in range(horizon):
            temporal_input = torch.cat([decoder_output[:, t, :], x_future_proj[:, t, :]], dim=1)
            pred_t = self.temporal_decoder(temporal_input)
            predictions.append(pred_t)
        predictions = torch.stack(predictions, dim=1)

        # 5. Add global residual
        residual = self.global_residual(y_past_flat)
        predictions = predictions + residual

        return predictions
```

### Explicación Detallada

**Arquitectura TiDE en 5 Pasos:**

#### 1. Feature Projection
**¿Qué hace?**
- Proyecta covariables de `num_covariates` (8) → `temporal_width` (típicamente 4)
- Se aplica **independientemente** a cada time-step
- Reduce dimensionalidad pero preserva información temporal

**Matemática:**
```
x_t ∈ ℝ^r (r = num_covariates)
x̃_t = FeatureProjection(x_t) ∈ ℝ^p (p = temporal_width)
```

**¿Por qué?**
- Reduce complejidad computacional
- Aprende representación compacta de patrones temporales

#### 2. Dense Encoder
**¿Qué hace?**
- Combina: valores pasados + covariables pasadas proyectadas + covariables futuras proyectadas
- Stack de Residual Blocks
- Genera un **encoding global** de toda la información

**Input:**
```
[y₁, y₂, ..., y_L,                           # look_back values
 x̃₁, x̃₂, ..., x̃_L,                          # past covariates projected
 x̃_{L+1}, x̃_{L+2}, ..., x̃_{L+H}]           # future covariates projected
```

**Output:**
```
e ∈ ℝ^{hidden_size}  # Encoding global
```

**¿Por qué incluir futuro?**
- Las covariables temporales (día, hora, etc.) **son conocidas de antemano**
- El modelo puede usar esta información para mejorar predicciones

#### 3. Dense Decoder
**¿Qué hace?**
- Transforma encoding global → representaciones por time-step futuro
- Output: `(horizon × decoder_output_dim)` valores
- Se reshape a `(horizon, decoder_output_dim)`

**Matemática:**
```
e ∈ ℝ^{hidden_size}
D = Decoder(e) ∈ ℝ^{H × d}  (d = decoder_output_dim)
D = [d₁, d₂, ..., d_H]  donde d_t ∈ ℝ^d
```

#### 4. Temporal Decoder
**¿Qué hace?**
- **Para cada time-step futuro**, combina:
  - Representación del decoder (d_t)
  - Covariables futuras proyectadas (x̃_{L+t})
- Genera predicción final para ese time-step

**Matemática:**
```
Para t = 1, 2, ..., H:
  ŷ_{L+t} = TemporalDecoder([d_t; x̃_{L+t}])
```

**¿Por qué por time-step?**
- Permite usar información específica de cada momento futuro
- Highway connection desde covariables futuras

#### 5. Global Residual Connection
**¿Qué hace?**
- Proyección lineal simple: `look_back` valores → `horizon` predicciones
- Se suma directamente a las predicciones del temporal decoder

**Matemática:**
```
r = Linear_residual([y₁, y₂, ..., y_L])
ŷ_final = ŷ_temporal_decoder + r
```

**¿Por qué?**
- Facilita aprendizaje de tendencias simples
- El modelo solo necesita aprender la parte difícil (residuos)
- Similar a ResNet: aprende diferencias, no mapeo completo

### Conceptos de Deep Learning

**1. Encoder-Decoder Architecture:**
- **Encoder**: Comprime información → representación latente
- **Decoder**: Expande representación → predicciones
- Usado en: Seq2Seq, Autoencoders, Transformers

**2. Dense (Fully Connected) Layers:**
- Contraste con: Convoluciones (locales), Attention (pairwise)
- Ventaja: Simplicidad, rapidez
- Desventaja: No explota estructura espacial/temporal directamente

**3. Multi-Scale Processing:**
- **Global**: Encoder/Decoder procesan toda la secuencia
- **Local**: Temporal Decoder procesa cada time-step

**4. Highway Connections:**
- Permiten flujo directo de información
- Temporal decoder tiene highway desde covariables futuras

### Complejidad Computacional

**TiDE:** O(L) - Lineal en longitud de secuencia
- Encoder: procesa secuencia flattened
- Decoder: genera todas predicciones en paralelo (excepto temporal loop)

**Transformer:** O(L²) - Cuadrática
- Self-attention: cada token atiende a todos los demás

**Resultado:**
- TiDE es **5-10x más rápido** que Transformers (según paper)

### Referencias
- [TiDE Paper](https://arxiv.org/abs/2304.08424)
- [Encoder-Decoder Tutorial](https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html)
- [Highway Networks](https://arxiv.org/abs/1505.00387)

---

## 9. Funciones de Entrenamiento

### train_epoch

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc='Training', leave=False):
        y_past = batch['y_past'].unsqueeze(-1).to(device)
        y_future = batch['y_future'].unsqueeze(-1).to(device)
        x_past = batch['x_past'].to(device)
        x_future = batch['x_future'].to(device)

        optimizer.zero_grad()
        predictions = model(y_past, x_past, x_future)
        loss = criterion(predictions, y_future)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)
```

**Conceptos:**
- **model.train()**: Activa dropout, batch norm en modo training
- **optimizer.zero_grad()**: Resetea gradientes acumulados
- **loss.backward()**: Backpropagation - calcula gradientes
- **optimizer.step()**: Actualiza pesos usando gradientes

**Algoritmo:**
```
Para cada batch:
  1. Forward pass: calcular predicciones
  2. Calcular loss (MSE)
  3. Backward pass: calcular ∂loss/∂weights
  4. Update: weights ← weights - lr * gradients
```

### evaluate

```python
def evaluate(model, data_loader, criterion, device, denorm_fn=None):
    model.eval()
    total_mse = 0
    total_mae = 0

    with torch.no_grad():
        for batch in data_loader:
            predictions = model(y_past, x_past, x_future)

            if denorm_fn is not None:
                predictions_denorm = denorm_fn(predictions)
                y_future_denorm = denorm_fn(y_future)

            mse = torch.mean((predictions_denorm - y_future_denorm) ** 2)
            mae = torch.mean(torch.abs(predictions_denorm - y_future_denorm))

            total_mse += mse.item()
            total_mae += mae.item()
```

**Conceptos:**
- **model.eval()**: Desactiva dropout, batch norm en modo inference
- **torch.no_grad()**: No calcular gradientes (ahorra memoria, acelera cómputo)
- **Denormalización**: Métricas en escala original

**Métricas:**

**MSE (Mean Squared Error):**
```
MSE = (1/n) Σ(ŷᵢ - yᵢ)²
```
- Penaliza errores grandes cuadráticamente
- Sensible a outliers

**MAE (Mean Absolute Error):**
```
MAE = (1/n) Σ|ŷᵢ - yᵢ|
```
- Penaliza errores linealmente
- Más robusto a outliers

### train_model

```python
def train_model(model, train_loader, val_loader, hparams, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams['epochs'])

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(hparams['epochs']):
        train_loss = train_epoch(...)
        val_mse, val_mae, _, _ = evaluate(...)
        scheduler.step()

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= hparams['patience']:
            print(f"Early stopping en época {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    return model
```

**Conceptos:**

**1. Adam Optimizer:**
- Adaptive Moment Estimation
- Combina momentum + adaptive learning rates
- Muy popular para deep learning
- Ecuaciones:
```
m_t = β₁ m_{t-1} + (1-β₁) g_t        # First moment (momentum)
v_t = β₂ v_{t-1} + (1-β₂) g_t²       # Second moment (adaptive)
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

**2. Cosine Annealing Learning Rate:**
- Reduce learning rate siguiendo curva coseno
- Fórmula:
```
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(πt/T))
```
- Ventajas:
  - Exploración inicial (lr alto)
  - Refinamiento final (lr bajo)
  - Puede escapar mínimos locales

**3. Early Stopping:**
- Monitorea métrica de validación
- Para entrenamiento si no mejora en `patience` épocas
- Previene overfitting
- Restaura mejor modelo (no el último)

**Referencias:**
- [Adam Paper](https://arxiv.org/abs/1412.6980)
- [Cosine Annealing](https://arxiv.org/abs/1608.03983)
- [Early Stopping](https://en.wikipedia.org/wiki/Early_stopping)

---

## 10. Pipeline de Experimentación

### Código
```python
for dataset_name in CONFIG['datasets']:
    for horizon in CONFIG['horizons']:
        # 1. Crear datasets
        train_dataset, val_dataset, test_dataset = create_datasets(...)

        # 2. DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        # 3. Crear modelo
        model = TiDEModel(...).to(device)

        # 4. Entrenar
        model, train_losses, val_losses = train_model(...)

        # 5. Evaluar en test
        test_mse, test_mae, predictions, targets = evaluate(...)

        # 6. Guardar resultados
        all_results.append({
            'Dataset': dataset_name,
            'Horizon': horizon,
            'MSE': test_mse,
            'MAE': test_mae
        })
```

### Explicación

**Loop Anidado:**
- Outer loop: Cada dataset (ETTh1, ETTh2, ETTm1, ETTm2)
- Inner loop: Cada horizonte (24, 48, 96, 192, 336, 720)
- Total: 4 × 6 = 24 experimentos

**DataLoader:**
```python
DataLoader(dataset, batch_size=512, shuffle=True)
```
- **batch_size=512**: Número de samples por batch (según paper)
- **shuffle=True** (train): Mezcla samples (pero respeta temporalidad dentro de cada window)
- **shuffle=False** (val/test): Evaluar en orden

**¿Por qué modelos separados por horizonte?**
- Cada horizonte es una tarea diferente
- 24 steps adelante ≠ 720 steps adelante
- Permite al modelo especializarse

**Conceptos:**
- **Hyperparameter Tuning**: Diferentes hparams por dataset
- **Grid Search**: Probar múltiples configuraciones
- **Model Selection**: Elegir mejor modelo basado en validation

---

## 11. Visualizaciones y Análisis

### MSE/MAE por Horizonte

```python
for idx, dataset_name in enumerate(CONFIG['datasets']):
    dataset_results = results_df[results_df['Dataset'] == dataset_name]
    axes[idx].plot(dataset_results['Horizon'], dataset_results['MSE'], marker='o')
```

**Análisis esperado:**
- ✅ MSE/MAE **aumentan** con horizonte más largo
- ✅ ETTm2 típicamente tiene mejor performance (datos más fáciles)
- ✅ Curva puede ser no-lineal

### Curvas de Entrenamiento

```python
axes[idx].plot(epochs, history['train_losses'], label='Train Loss')
axes[idx].plot(epochs, history['val_losses'], label='Val Loss')
```

**Diagnóstico:**
- **Underfitting**: Train y val altos, no convergen
- **Overfitting**: Train bajo, val alto y divergente
- **Bueno**: Ambos bajos, val se estabiliza

### Predicciones vs Real

```python
axes[idx].plot(targets, label='Real')
axes[idx].plot(predictions, label='Predicción')
```

**¿Qué buscar?**
- ✅ Predicciones siguen tendencia general
- ✅ Captura picos y valles importantes
- ❌ Lag (predicción retrasada respecto a real)
- ❌ Sobre-suavizado (predicción demasiado plana)

### Heatmap de Resultados

```python
pivot_mse = results_df.pivot(index='Dataset', columns='Horizon', values='MSE')
sns.heatmap(pivot_mse, annot=True, fmt='.4f', cmap='YlOrRd')
```

**Ventajas:**
- Vista global de performance
- Identificar patrones: ¿qué datasets/horizontes son difíciles?
- Comparar rápidamente

### Comparación con Paper

```python
comparison_data.append({
    'Our_MSE': row['MSE'],
    'Paper_MSE': paper_mse,
    'MSE_Diff_%': ((row['MSE'] - paper_mse) / paper_mse * 100)
})
```

**Expectativas realistas:**
- Diferencias de ±10-20% son normales (diferentes seeds, implementaciones)
- Si diferencia > 50%: revisar bugs
- Paper usa más épocas, múltiples runs, etc.

**Referencias:**
- [Visualizing Time Series](https://otexts.com/fpp3/graphics.html)
- [Model Diagnostics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## Resumen de Conceptos Clave

### Deep Learning
1. **Residual Connections**: Facilitan entrenamiento profundo
2. **Encoder-Decoder**: Compresión → Expansión de información
3. **Normalization**: RevIN, LayerNorm para estabilidad
4. **Regularization**: Dropout para prevenir overfitting
5. **Optimization**: Adam + Cosine Annealing + Early Stopping

### Time Series
1. **Temporal Features**: Capturar estacionalidad
2. **Sliding Windows**: Crear samples de train/test
3. **Temporal Splitting**: No shuffle en series de tiempo
4. **Horizonte Variable**: Diferentes tareas para diferentes horizontes
5. **Covariates**: Features conocidas en el futuro

### TiDE Específico
1. **MLP-based**: No attention, solo dense layers
2. **Channel-Independent**: Procesa cada serie separadamente
3. **Global + Temporal**: Multi-scale processing
4. **Residual Forecast**: Aprende diferencias, no valores absolutos
5. **Complejidad O(L)**: Mucho más rápido que Transformers

---

## Recursos Adicionales

### Papers
- [TiDE Original Paper](https://arxiv.org/abs/2304.08424)
- [Attention Is All You Need (Transformers)](https://arxiv.org/abs/1706.03762)
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)

### Tutoriales
- [PyTorch Time Series Tutorial](https://pytorch.org/tutorials/beginner/timeseries_tutorial.html)
- [Time Series Forecasting with Deep Learning](https://www.tensorflow.org/tutorials/structured_data/time_series)

### Librerías
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)

### Cursos
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [Practical Deep Learning - fast.ai](https://course.fast.ai/)

---

**Autor:** Implementación del paper TiDE para Laboratorio 2
**Fecha:** 2024
**Versión:** 1.0
