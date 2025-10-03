# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Learning laboratory (Lab 2) implementing **Long-Term Time Series Forecasting (LTSF)** using the **TiDE (Time-series Dense Encoder)** model on ETT datasets.

**Reference Paper**: "Long-term Forecasting with TiDE: Time-series Dense Encoder" (2304.08424v5.pdf)

## Dataset

**Source**: https://github.com/zhouhaoyi/ETDataset

**Datasets to use**:
- ETTh1 (hourly data)
- ETTh2 (hourly data)
- ETTm1 (15-minute data)
- ETTm2 (15-minute data)

**Data splits**:
- Training: 70%
- Validation: 10%
- Test: 20% (temporal split)

## Task Configuration

**Window Configuration**:
- Input length (look-back): 96
- Prediction horizons: [24, 48, 96, 192, 336, 720]
- Target variable: OT (Oil Temperature)

**Evaluation Metrics**: MSE and MAE per horizon

## TiDE Model Architecture

The TiDE model is an MLP-based encoder-decoder architecture with the following key components:

### Encoding
1. **Feature Projection**: Maps dynamic covariates to lower dimensional space (temporalWidth)
2. **Dense Encoder**: Combines look-back, projected covariates, and static attributes

### Decoding
1. **Dense Decoder**: Maps encoding to horizon-length predictions
2. **Temporal Decoder**: Adds highway from future covariates to predictions
3. **Global Residual Connection**: Linear mapping from look-back to horizon

### Key Hyperparameters (from paper Table 7-8)

For ETT datasets, typical ranges:
- `hiddenSize`: [256, 512, 1024]
- `numEncoderLayers`: [1, 2, 3]
- `numDecoderLayers`: [1, 2, 3]
- `decoderOutputDim`: [4, 8, 16, 32]
- `temporalDecoderHidden`: [32, 64, 128]
- `temporalWidth`: typically 4
- `dropoutLevel`: [0.0, 0.1, 0.2, 0.3, 0.5]
- `layerNorm`: [True, False]
- `batchSize`: 512 (fixed in paper)
- `learningRate`: Log-scale in [1e-5, 1e-2]
- `revIn` (Reversible Instance Normalization): [True, False]

**ETTh1 specific** (from paper Table 8):
- hiddenSize: 256, numEncoderLayers: 2, numDecoderLayers: 2
- decoderOutputDim: 8, temporalDecoderHidden: 128
- dropoutLevel: 0.3, layerNorm: True, learningRate: 3.82e-5, revIn: True

**ETTh2 specific**:
- hiddenSize: 512, numEncoderLayers: 2, numDecoderLayers: 2
- decoderOutputDim: 32, temporalDecoderHidden: 16
- dropoutLevel: 0.2, layerNorm: True, learningRate: 2.24e-4, revIn: True

**ETTm1 specific**:
- hiddenSize: 1024, numEncoderLayers: 1, numDecoderLayers: 1
- decoderOutputDim: 8, temporalDecoderHidden: 128
- dropoutLevel: 0.5, layerNorm: True, learningRate: 8.39e-5, revIn: False

**ETTm2 specific**:
- hiddenSize: 512, numEncoderLayers: 2, numDecoderLayers: 2
- decoderOutputDim: 16, temporalDecoderHidden: 128
- dropoutLevel: 0.0, layerNorm: True, learningRate: 2.52e-4, revIn: True

## Implementation Requirements

### Data Preprocessing
1. Load ETT datasets from CSV files
2. Create sliding windows with look-back=96
3. Normalize using training set statistics (mean and std)
4. Generate time-derived features (8 features total):
   - Minute of hour, hour of day, day of week, etc.
   - Normalized to [-0.5, 0.5] scale

### Model Components

**Residual Block** (core building block):
```
Input → Dense(ReLU) → Dropout → Dense(Linear) → LayerNorm → + Skip → Output
```

**Feature Projection** (per time-step):
```
x_t (r dimensions) → ResidualBlock → x̃_t (temporalWidth dimensions)
```

**Dense Encoder**:
```
[y_{1:L}; x̃_{1:L+H}; a] → Stack of ResidualBlocks → e (encoding)
```

**Dense Decoder**:
```
e → Stack of ResidualBlocks → Reshape → D^{p×H} (decoded vectors per time-step)
```

**Temporal Decoder** (per time-step):
```
[d_t; x̃_{L+t}] → ResidualBlock → ŷ_{L+t}
```

### Training

- **Loss**: MSE
- **Optimizer**: Adam with cosine decay learning rate schedule
- **Early stopping**: Monitor validation loss
- Use rolling validation for hyperparameter tuning
- Train separate models for each horizon length

### Expected Performance (from paper Table 2)

ETTh1 MSE/MAE:
- Horizon 96: 0.375/0.398
- Horizon 192: 0.412/0.422
- Horizon 336: 0.435/0.433
- Horizon 720: 0.454/0.465

ETTh2 MSE/MAE:
- Horizon 96: 0.270/0.336
- Horizon 192: 0.332/0.380
- Horizon 336: 0.360/0.407
- Horizon 720: 0.419/0.451

ETTm1 MSE/MAE:
- Horizon 96: 0.306/0.349
- Horizon 192: 0.335/0.366
- Horizon 336: 0.364/0.384
- Horizon 720: 0.413/0.413

ETTm2 MSE/MAE:
- Horizon 96: 0.161/0.251
- Horizon 192: 0.215/0.289
- Horizon 336: 0.267/0.326
- Horizon 720: 0.352/0.383

## Deliverables

### Code (Jupyter Notebook or Python scripts)
1. Data preprocessing and loading
2. Model implementation
3. Training and evaluation
4. Training plots (MSE, MAE)
5. Test results and analysis

### Video (max 15 minutes)
1. Paper methodology (5 min)
2. Implementation details (5 min)
3. Results, comparisons, ablations, discussion, and limitations (5 min)

## Key Implementation Notes

- TiDE is channel-independent: processes one time-series at a time during forward pass
- The model uses both past values AND future covariates (known in advance)
- Reversible Instance Normalization (RevIN) is optional but can improve performance
- Context size of 720 can be used for all horizons (though 96 is specified in assignment)
- Model is 5-10x faster than Transformers while achieving competitive accuracy
