# Resumen del Paper: Long-term Forecasting with TiDE

**Título:** Long-term Forecasting with TiDE: Time-series Dense Encoder
**Autores:** Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan Mathur, Rajat Sen, Rose Yu
**Institución:** Google Cloud AI Research
**Año:** 2023
**ArXiv:** [2304.08424](https://arxiv.org/abs/2304.08424)

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Motivación y Problema](#motivación-y-problema)
3. [Arquitectura TiDE](#arquitectura-tide)
4. [Componentes Técnicos](#componentes-técnicos)
5. [Experimentos y Resultados](#experimentos-y-resultados)
6. [Ventajas y Limitaciones](#ventajas-y-limitaciones)
7. [Contribuciones Principales](#contribuciones-principales)
8. [Impacto y Aplicaciones](#impacto-y-aplicaciones)

---

## Resumen Ejecutivo

### ¿Qué es TiDE?

**TiDE (Time-series Dense Encoder)** es un modelo basado completamente en **Multi-Layer Perceptrons (MLPs)** para pronóstico de series de tiempo a largo plazo (Long-Term Time Series Forecasting, LTSF).

### Hallazgos Principales

1. **Simplicidad vs Complejidad**: TiDE, usando solo MLPs, iguala o supera el rendimiento de arquitecturas complejas basadas en Transformers

2. **Eficiencia Computacional**: 5-10x más rápido que modelos basados en atención (Transformers)

3. **Competitividad en Benchmarks**: Logra performance state-of-the-art en múltiples datasets populares

4. **Complejidad Lineal**: O(L) en longitud de secuencia vs O(L²) de Transformers

### Mensaje Central

> "Para pronóstico de series de tiempo a largo plazo, arquitecturas simples basadas en MLPs pueden ser tan efectivas como modelos complejos con atención, mientras son significativamente más rápidas y fáciles de entrenar."

---

## Motivación y Problema

### El Contexto

**Long-Term Time Series Forecasting (LTSF):**
- **Objetivo**: Predecir múltiples pasos hacia el futuro (ej: 96, 192, 720 time-steps)
- **Aplicaciones**: Predicción de demanda energética, pronóstico financiero, predicción del clima, etc.
- **Desafío**: Capturar dependencias a largo plazo mientras se mantiene eficiencia computacional

### El Estado del Arte (Pre-TiDE)

**Dominancia de Transformers (2020-2023):**
- Modelos: Informer, Autoformer, FEDformer, etc.
- **Fortalezas**: Capturan dependencias a largo plazo vía self-attention
- **Debilidades**:
  - Complejidad cuadrática O(L²)
  - Costosos de entrenar
  - Difíciles de interpretar

**Sorpresa Empírica:**
- DLinear (2022): Un modelo lineal simple superó a muchos Transformers
- **Pregunta**: ¿Realmente necesitamos atención para LTSF?

### La Propuesta TiDE

**Hipótesis:**
> "La clave no es la atención, sino la capacidad de codificar eficientemente el pasado y decodificar representaciones ricas hacia el futuro."

**Estrategia:**
- Encoder-Decoder basado en MLPs densos
- Residual connections para facilitar aprendizaje
- Uso de covariables temporales (features conocidas de antemano)
- Diseño modular y flexible

---

## Arquitectura TiDE

### Visión General

TiDE es una arquitectura **encoder-decoder** con 5 componentes principales:

```
Input: [y₁, ..., yₗ, x₁, ..., xₗ₊ₕ, a]
       ↓
[1] Feature Projection
       ↓
[2] Dense Encoder
       ↓
[3] Dense Decoder
       ↓
[4] Temporal Decoder
       ↓
[5] Global Residual Connection
       ↓
Output: [ŷₗ₊₁, ..., ŷₗ₊ₕ]
```

**Notación:**
- **L**: Look-back window (contexto pasado)
- **H**: Horizon (pasos a predecir)
- **y**: Variable objetivo (univariate)
- **x**: Covariables dinámicas (multivariate, conocidas en pasado y futuro)
- **a**: Atributos estáticos (opcional, por ejemplo: ID de serie, categoría)

### Diagrama Detallado

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUTS                               │
├───────────────┬─────────────────┬───────────────────────────┤
│  y_{1:L}      │  x_{1:L}        │  x_{L+1:L+H}      │   a   │
│  (past values)│  (past features)│  (future features)│ (attr)│
└───────┬───────┴────────┬────────┴──────────┬────────┴───┬───┘
        │                │                   │            │
        │         ┌──────▼──────┐     ┌──────▼──────┐    │
        │         │  Feature    │     │  Feature    │    │
        │         │ Projection  │     │ Projection  │    │
        │         │  (MLP)      │     │  (MLP)      │    │
        │         └──────┬──────┘     └──────┬──────┘    │
        │                │                   │            │
        │                x̃_{1:L}            x̃_{L+1:L+H}  │
        │                │                   │            │
        └────────────────┴───────────────────┴────────────┘
                                │
                         ┌──────▼──────┐
                         │    Dense    │
                         │   Encoder   │
                         │ (ResBlocks) │
                         └──────┬──────┘
                                │
                              e (encoding)
                                │
                         ┌──────▼──────┐
                         │    Dense    │
                         │   Decoder   │
                         │ (ResBlocks) │
                         └──────┬──────┘
                                │
                         D ∈ ℝ^{H×d}
                                │
        ┌───────────────────────┴────────────────────┐
        │         (for each time-step t)             │
        │                                            │
    ┌───▼────┐                              ┌───────▼────┐
    │  d_t   │                              │ x̃_{L+t}   │
    └───┬────┘                              └───────┬────┘
        └─────────────┬─────────────────────────────┘
                      │
               ┌──────▼──────┐
               │  Temporal   │
               │   Decoder   │
               │  (ResBlock) │
               └──────┬──────┘
                      │
                   ŷ_{L+t}

        PLUS Global Residual: Linear(y_{1:L}) → H predictions

                    FINAL OUTPUT: ŷ_{L+1:L+H}
```

---

## Componentes Técnicos

### 1. Feature Projection

**Propósito:** Reducir dimensionalidad de covariables

**Arquitectura:**
```
x_t ∈ ℝ^r → ResidualBlock → x̃_t ∈ ℝ^p
```

**Detalles:**
- Se aplica **independientemente** a cada time-step
- Proyecta de `r` (dimensión original) a `p` (temporal width, típicamente 4)
- Shared weights para todos los time-steps (similar a 1D convolution con kernel=1)

**Ventajas:**
- Reduce complejidad computacional
- Aprende representación compacta
- Filtra ruido en covariables

### 2. Dense Encoder

**Input:**
```
Flatten([y_{1:L}; x̃_{1:L}; x̃_{L+1:L+H}; a])
```

**Arquitectura:**
```
Stack of k Residual Blocks
Input dim: L + (L+H)p + d_a
Hidden dim: h
Output: e ∈ ℝ^h
```

**Residual Block:**
```
x → Linear(h) → ReLU → Dropout → Linear(h) → LayerNorm → (+skip) → out
```

**Conceptos Clave:**
- **Global context**: Procesa toda la información de una vez
- **Past + Future**: Usa covariables conocidas del futuro
- **Compression**: De (L + (L+H)p) dimensional → h dimensional

### 3. Dense Decoder

**Input:** Encoding `e ∈ ℝ^h`

**Output:** `D ∈ ℝ^{H×d}` donde `d` = decoder_output_dim

**Arquitectura:**
```
e → Stack of m Residual Blocks → Flatten(Hd) → Reshape → D
```

**Propósito:**
- Expande encoding global → representaciones por time-step
- Cada fila `d_t` contiene información para predecir `ŷ_{L+t}`

### 4. Temporal Decoder

**Por cada time-step futuro t:**

```
Input: [d_t; x̃_{L+t}] ∈ ℝ^{d+p}
      ↓
  ResidualBlock
      ↓
Output: ŷ_{L+t} ∈ ℝ
```

**Ventajas:**
- **Highway connection**: Covariables futuras influencian directamente predicciones
- **Time-specific**: Cada predicción usa información de su momento específico
- **Local refinement**: Ajusta predicción global del decoder

### 5. Global Residual Connection

**Arquitectura:**
```
y_{1:L} → Linear(L → H) → residual ∈ ℝ^H
```

**Predicción Final:**
```
ŷ_final = ŷ_temporal_decoder + residual
```

**Inspiración:** ResNet - aprende residuos, no mapping completo

**Ventajas:**
- Facilita aprendizaje de tendencias simples
- Mitiga vanishing gradients
- Permite al modelo enfocarse en patrones complejos

---

## Experimentos y Resultados

### Datasets Evaluados

**1. ETT (Electricity Transformer Temperature)**
- ETTh1, ETTh2: Datos horarios
- ETTm1, ETTm2: Datos cada 15 minutos
- 7 variables (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)

**2. Electricity**
- Consumo eléctrico de 321 clientes
- Datos horarios

**3. Weather**
- 21 indicadores meteorológicos
- Datos cada 10 minutos

**4. Traffic**
- Ocupación de carreteras
- 862 sensores

### Configuración Experimental

**Horizontes:** 96, 192, 336, 720 time-steps

**Baselines Comparados:**
- **Transformers**: Informer, Autoformer, FEDformer, Pyraformer
- **MLPs**: DLinear, N-BEATS
- **RNNs**: DeepAR
- **CNNs**: TCN

**Métricas:**
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)

### Resultados Principales

#### Tabla 2 del Paper - ETT Datasets (Ejemplo: ETTh1)

| Horizonte | TiDE MSE | TiDE MAE | Mejor Baseline | Mejora |
|-----------|----------|----------|----------------|--------|
| 96        | **0.375** | **0.398** | Autoformer: 0.449 | 16.5% |
| 192       | **0.412** | **0.422** | FEDformer: 0.420 | 1.9% |
| 336       | **0.435** | **0.433** | Autoformer: 0.441 | 1.4% |
| 720       | **0.454** | **0.465** | FEDformer: 0.464 | 2.2% |

**Bold** = Mejor resultado

#### Comparación con DLinear

DLinear era el modelo más simple y eficiente antes de TiDE:

```
DLinear: y_{L+h} = W * y_{1:L} + b
```

**TiDE vs DLinear (promedio en ETT):**
- TiDE MSE: **0.419**
- DLinear MSE: 0.445
- **Mejora**: 5.8%

**Interpretación:**
- TiDE agrega capacidad no-lineal sin sacrificar eficiencia
- Residual connection hace que TiDE ≈ DLinear + non-linearity

### Eficiencia Computacional

**Tiempo de Entrenamiento (ETTh1, horizonte 96):**

| Modelo      | Tiempo/Época | Speedup vs TiDE |
|-------------|--------------|-----------------|
| TiDE        | 3.2s         | 1x              |
| DLinear     | 2.1s         | 0.66x           |
| Autoformer  | 18.7s        | 5.8x            |
| FEDformer   | 31.2s        | 9.8x            |

**Complejidad:**
- TiDE: O(L) - Lineal
- Transformers: O(L²) - Cuadrática

### Ablation Studies

**¿Qué componentes son más importantes?**

| Variante                      | MSE   | Cambio  |
|-------------------------------|-------|---------|
| TiDE completo                 | 0.375 | -       |
| Sin Feature Projection        | 0.391 | +4.3%   |
| Sin Temporal Decoder          | 0.402 | +7.2%   |
| Sin Global Residual           | 0.412 | +9.9%   |
| Sin LayerNorm                 | 0.398 | +6.1%   |
| Encoder shallow (1 layer)     | 0.388 | +3.5%   |
| Decoder shallow (1 layer)     | 0.383 | +2.1%   |

**Conclusiones:**
1. **Global Residual** es el componente más crítico
2. **Temporal Decoder** captura patrones time-specific importantes
3. **LayerNorm** estabiliza entrenamiento
4. Encoder/Decoder profundos ayudan pero con rendimientos decrecientes

### Análisis de Covariables

**Impacto de Covariables Temporales:**

| Setup                         | MSE   |
|-------------------------------|-------|
| TiDE con covariables          | 0.375 |
| TiDE sin covariables futuras  | 0.401 |
| TiDE sin covariables (pasado + futuro) | 0.428 |

**Aprendizaje:**
- Covariables futuras (hora, día, mes, etc.) **son muy valiosas**
- Mejora de 14% al incluirlas

### RevIN (Reversible Instance Normalization)

**Impacto por Dataset:**

| Dataset | Sin RevIN | Con RevIN | Mejora |
|---------|-----------|-----------|--------|
| ETTh1   | 0.401     | **0.375** | 6.5%   |
| ETTh2   | 0.283     | **0.270** | 4.6%   |
| ETTm1   | 0.306     | 0.313     | -2.3%  |
| ETTm2   | 0.169     | **0.161** | 4.7%   |

**Conclusión:**
- RevIN ayuda en la mayoría de casos
- No es universal (ETTm1 empeora)
- Depende de características de la distribución del dataset

---

## Ventajas y Limitaciones

### Ventajas

#### 1. Simplicidad
- Solo MLPs (Fully Connected layers)
- Fácil de implementar en cualquier framework
- Pocas dependencias

#### 2. Eficiencia
- **5-10x más rápido** que Transformers
- Complejidad O(L) vs O(L²)
- Menos memoria requerida

#### 3. Performance
- Competitivo o superior a Transformers
- State-of-the-art en múltiples benchmarks

#### 4. Interpretabilidad
- Componentes tienen roles claros:
  - Encoder: comprime pasado
  - Decoder: genera representaciones futuras
  - Temporal: refina con info local
  - Residual: baseline simple

#### 5. Flexibilidad
- Fácil incorporar:
  - Covariables (dinámicas y estáticas)
  - Múltiples series (con atributos estáticos)
  - RevIN para normalización

### Limitaciones

#### 1. Channel-Independent
- Procesa cada serie **independientemente**
- No captura dependencias **entre** series
- Para N series, entrena N modelos

**Ejemplo:**
```
Series 1: Temperatura
Series 2: Humedad
```
TiDE NO aprende que alta temperatura → baja humedad

**Solución propuesta en paper:**
- Usar atributos estáticos para identificar series
- Compartir pesos entre series similares

#### 2. Covariables Requeridas
- Performance óptimo requiere covariables temporales
- Si no hay features conocidas del futuro, performance baja

#### 3. Horizonte Fijo
- Modelo entrenado para horizonte específico
- Predecir H=96 y H=720 requiere 2 modelos diferentes

**Alternativa:**
- Entrenar con horizonte máximo
- Truncar predicciones para horizontes menores
- (Tradeoff: menos especialización)

#### 4. No Probabilístico
- Solo predicciones puntuales
- No genera intervalos de confianza/distribuciones

**Extensión posible:**
- Output multiple quantiles
- Usar loss de cuantiles en lugar de MSE

#### 5. Hiperparámetros Sensibles
- Requiere tuning por dataset:
  - hidden_size, num_layers, dropout, etc.
- Paper usa extensiva búsqueda de hiperparámetros

---

## Contribuciones Principales

### 1. Demostración Empírica

**Claim:**
> "Para LTSF, MLPs simples son tan buenos como Transformers complejos"

**Evidencia:**
- Resultados en 8 datasets
- Múltiples horizontes
- Comparación con 10+ baselines

### 2. Arquitectura TiDE

**Innovaciones:**
- Combinación efectiva de componentes conocidos:
  - Encoder-Decoder
  - Residual Blocks
  - Global Residual Connection
  - Temporal Decoder (highway)

**No es:** Invención de nuevos bloques
**Sí es:** Diseño arquitectónico efectivo

### 3. Análisis de Covariables

**Insights:**
- Covariables futuras son **cruciales** para LTSF
- Feature projection es importante para alta dimensionalidad

### 4. Eficiencia como Prioridad

**Mensaje:**
> "Performance y velocidad no son mutuamente exclusivos"

**Impacto:**
- Deployment más fácil
- Útil para aplicaciones en tiempo real
- Menor huella de carbono

### 5. Ablation Studies Exhaustivos

- Identifican componentes críticos
- Guía para simplificaciones si es necesario

---

## Impacto y Aplicaciones

### Impacto en la Comunidad

**1. Repensar Transformers para Time Series**
- No todas las tareas requieren atención
- Inductive bias de Transformers puede no ser apropiado para LTSF

**2. Renacimiento de MLPs**
- Motivó trabajos posteriores en MLPs para TS
- Ej: TSMixer, TiDE variants

**3. Benchmark Reference**
- TiDE es baseline estándar en papers nuevos (2023-2024)

### Aplicaciones Prácticas

#### 1. Predicción de Demanda Energética
- Horizonte: 24-720 horas
- Covariables: Hora, día, estacionalidad
- Beneficio: Optimización de generación

#### 2. Pronóstico Financiero
- Precio de acciones, criptomonedas
- Covariables: Día de la semana, eventos calendarios
- Beneficio: Trading algorítmico

#### 3. Planificación de Recursos
- Demanda de productos, tráfico web
- Covariables: Promociones, eventos
- Beneficio: Inventario óptimo

#### 4. Predicción Meteorológica
- Temperatura, precipitación
- Covariables: Estación, ubicación
- Beneficio: Alertas tempranas

#### 5. IoT y Monitoreo
- Sensores industriales, salud
- Covariables: Ciclos de operación
- Beneficio: Mantenimiento predictivo

### Consideraciones de Deployment

**Ventajas para Producción:**
- ✅ Rápido entrenamiento → Updates frecuentes
- ✅ Baja latencia → Tiempo real
- ✅ Pocos recursos → Edge devices

**Desafíos:**
- ⚠️ Modelos separados por horizonte → Múltiples deployments
- ⚠️ Channel-independent → Escalabilidad para muchas series

---

## Comparación con Alternativas

### TiDE vs Transformers

| Aspecto              | TiDE                | Transformers        |
|----------------------|---------------------|---------------------|
| Complejidad          | O(L)                | O(L²)               |
| Velocidad            | 5-10x más rápido    | Lento               |
| Memoria              | Baja                | Alta                |
| Performance LTSF     | Competitivo/Superior| Variable            |
| Interpretabilidad    | Alta                | Baja (atención)     |
| Multivariate         | Channel-independent | Puede capturar deps |
| Implementación       | Simple              | Compleja            |

**¿Cuándo usar Transformers?**
- Series con dependencias complejas entre canales
- Tareas que requieren atención (ej: anomaly detection)
- Abundancia de datos y recursos computacionales

**¿Cuándo usar TiDE?**
- LTSF con covariables temporales
- Recursos limitados
- Necesidad de rapidez
- Series univariadas o débilmente correlacionadas

### TiDE vs DLinear

| Aspecto              | TiDE                | DLinear             |
|----------------------|---------------------|---------------------|
| Capacidad            | No-lineal (MLPs)    | Lineal              |
| Performance          | Superior            | Bueno               |
| Velocidad            | 1.5x más lento      | Muy rápido          |
| Covariables          | Explícitamente      | Implícitamente      |
| Complejidad          | Media               | Muy simple          |

**Tradeoff:**
- DLinear: Baseline excelente, súper simple
- TiDE: Más poder expresivo sin mucho costo

### TiDE vs RNNs/LSTMs

| Aspecto              | TiDE                | RNNs/LSTMs          |
|----------------------|---------------------|---------------------|
| Paralelización       | Total               | Secuencial          |
| Entrenamiento        | Rápido              | Lento               |
| Long-term deps       | Global (encoder)    | Difícil (vanishing) |
| Covariables futuras  | Nativo              | Difícil             |

**Ventaja de TiDE:**
- No sufre de vanishing/exploding gradients
- Procesamiento paralelo

---

## Lecciones Aprendidas

### 1. Simplicidad Funciona
> "No siempre necesitas el modelo más nuevo y complejo"

**Implicación:**
- Prueba baselines simples primero
- MLPs bien diseñados pueden sorprender

### 2. Inductive Bias Importa
> "El inductive bias de Transformers (atención) no es óptimo para todas las tareas"

**Para LTSF:**
- Encoder-decoder + residuals > self-attention
- Global context > pairwise interactions

### 3. Covariables son Críticas
> "Features conocidas del futuro son oro para forecasting"

**Estrategia:**
- Invertir tiempo en feature engineering temporal
- Hora, día, estacionalidad, eventos

### 4. Eficiencia es Feature, no Bug
> "Modelos rápidos permiten más iteraciones, mejores productos"

**Beneficios:**
- Experimentación rápida
- Deployment viable
- Sostenibilidad

### 5. Ablations > Claims
> "Muestra qué componentes importan, no solo el resultado final"

**Valor:**
- Entendimiento profundo
- Adaptaciones informadas
- Confianza en el diseño

---

## Trabajo Futuro y Extensiones

### Propuestas en el Paper

**1. Multivariate TiDE:**
- Capturar dependencias entre series
- Posible vía atributos estáticos o cross-attention selectiva

**2. Probabilistic TiDE:**
- Predicciones con incertidumbre
- Quantile regression o output distributions

**3. Horizonte Variable:**
- Modelo único para múltiples horizontes
- Conditional encoding basado en H

**4. AutoML para TiDE:**
- Búsqueda automática de hiperparámetros
- Neural Architecture Search

### Extensiones de la Comunidad

**TSMixer (Google, 2023):**
- Basado en ideas de TiDE
- Mixing temporal + feature dimensions

**TiDE+ (varios autores):**
- Agregan attention selectiva
- Híbridos TiDE + Transformers

**Aplicaciones Específicas:**
- TiDE para clima (adaptaciones RevIN)
- TiDE para finanzas (volatility modeling)

---

## Recursos y Referencias

### Paper Original
- **ArXiv:** [2304.08424](https://arxiv.org/abs/2304.08424)
- **PDF:** [Direct Link](https://arxiv.org/pdf/2304.08424.pdf)

### Implementaciones

**Oficial (Google Research):**
- [GitHub - google-research/google-research/tree/master/tide](https://github.com/google-research/google-research/tree/master/tide)

**Community:**
- [PyTorch Lightning Implementation](https://github.com/Nixtla/neuralforecast)
- [TensorFlow Implementation](https://github.com/thuml/Time-Series-Library)

### Datasets

**ETT:**
- [GitHub - zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)

**Otros Benchmarks:**
- [Monash Time Series Forecasting Archive](https://forecastingdata.org/)

### Papers Relacionados

**Baselines Comparados:**
1. **Informer** (AAAI 2021): [2012.07436](https://arxiv.org/abs/2012.07436)
2. **Autoformer** (NeurIPS 2021): [2106.13008](https://arxiv.org/abs/2106.13008)
3. **FEDformer** (ICML 2022): [2201.12740](https://arxiv.org/abs/2201.12740)
4. **DLinear** (AAAI 2023): [2205.13504](https://arxiv.org/abs/2205.13504)

**Conceptos:**
5. **RevIN** (ICLR 2022): [Reversible Instance Normalization](https://openreview.net/forum?id=cGDAkQo1C0p)
6. **ResNet** (CVPR 2016): [1512.03385](https://arxiv.org/abs/1512.03385)

### Tutoriales y Blogs

- [TiDE Explained - Towards Data Science](https://towardsdatascience.com/tagged/time-series)
- [Time Series Forecasting Best Practices](https://otexts.com/fpp3/)
- [Google Research Blog](https://ai.googleblog.com/)

---

## Conclusión

### Resumen Final

**TiDE demuestra que:**

1. **Simplicidad arquitectónica** no implica menor performance
2. **Eficiencia computacional** es alcanzable sin sacrificar precisión
3. **Diseño cuidadoso** de componentes simples supera complejidad arbitraria
4. **Covariables temporales** son fundamentales para LTSF
5. **Residual connections** siguen siendo la herramienta más poderosa en DL

### Impacto en el Campo

TiDE ha redefinido la conversación en time series forecasting, demostrando que **la complejidad debe ser justificada, no asumida**. Ha inspirado una nueva generación de modelos que priorizan eficiencia y diseño arquitectónico sobre acumulación de componentes complejos.

### Pregunta Filosófica

> "¿Cuánta complejidad necesitamos realmente?"

TiDE sugiere: **Menos de lo que pensamos, si diseñamos con cuidado.**

---

**Autor del Resumen:** Basado en el paper original de Das et al. (2023)
**Fecha:** 2024
**Versión:** 1.0
**Propósito:** Material educativo para Laboratorio 2 - Deep Learning
