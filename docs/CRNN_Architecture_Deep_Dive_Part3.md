# Part 8: Model Compilation and Optimization

## Model Creation and Compilation

```python
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

def create_complete_crnn(sequence_length=600, n_features=40):
    """
    Complete CRNN architecture for your emotion recognition data

    Input: (batch, 600, 40) - Your preprocessed data shape
    Output: (batch, 2) - Valence and arousal predictions
    """

    # Input layer
    inputs = Input(shape=(sequence_length, n_features), name='audio_features')

    # CNN Block 1: Local pattern extraction
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(bn1)
    pool1 = MaxPooling1D(pool_size=2)(conv2)  # 600 → 300 timesteps
    drop1 = Dropout(0.2)(pool1)

    # CNN Block 2: Higher-level patterns
    conv3 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(drop1)
    bn2 = BatchNormalization()(conv3)
    conv4 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(bn2)
    pool2 = MaxPooling1D(pool_size=2)(conv4)  # 300 → 150 timesteps
    drop2 = Dropout(0.3)(pool2)

    # RNN Block: Temporal dependencies
    lstm1 = LSTM(units=128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(drop2)
    lstm2 = LSTM(units=64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)(lstm1)

    # Dense Block: Final prediction
    dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(lstm2)
    drop3 = Dropout(0.4)(dense1)
    dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(drop3)

    # Output layer
    outputs = Dense(2, activation='linear', name='emotion_output')(dense2)

    model = Model(inputs=inputs, outputs=outputs, name='EmotionCRNN')
    return model

# Create model
model = create_complete_crnn(sequence_length=600, n_features=40)
model.summary()
```

## Loss Functions for Emotion Regression

### 1. Mean Squared Error (MSE) - Default Choice

```python
from tensorflow.keras.losses import MeanSquaredError

# Standard MSE - treats valence and arousal equally
loss = MeanSquaredError()

# Custom weighted MSE if one dimension is more important
def weighted_mse(y_true, y_pred):
    valence_weight = 1.0  # Weight for valence prediction
    arousal_weight = 1.2  # Weight for arousal prediction (slightly higher)

    valence_loss = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0])) * valence_weight
    arousal_loss = tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1])) * arousal_weight

    return valence_loss + arousal_loss
```

### 2. Huber Loss - Robust to Outliers

```python
from tensorflow.keras.losses import Huber

# Huber loss is less sensitive to outliers than MSE
# Good if your emotion annotations have some noisy labels
loss = Huber(delta=1.0)  # delta controls transition point between linear and quadratic

# Custom Huber for both dimensions
def emotion_huber_loss(y_true, y_pred, delta=1.0):
    valence_huber = tf.keras.losses.huber(y_true[:, 0], y_pred[:, 0], delta=delta)
    arousal_huber = tf.keras.losses.huber(y_true[:, 1], y_pred[:, 1], delta=delta)
    return valence_huber + arousal_huber
```

### 3. Custom Emotion-Aware Loss

```python
def emotion_aware_loss(y_true, y_pred):
    """
    Custom loss that considers emotion quadrants
    Penalizes predictions that cross quadrant boundaries more heavily
    """
    valence_true, arousal_true = y_true[:, 0], y_true[:, 1]
    valence_pred, arousal_pred = y_pred[:, 0], y_pred[:, 1]

    # Standard MSE component
    mse_loss = tf.reduce_mean(tf.square(valence_true - valence_pred) +
                             tf.square(arousal_true - arousal_pred))

    # Quadrant penalty - extra penalty if prediction is in wrong quadrant
    true_quadrant_v = tf.sign(valence_true)  # -1 or 1
    true_quadrant_a = tf.sign(arousal_true)  # -1 or 1
    pred_quadrant_v = tf.sign(valence_pred)  # -1 or 1
    pred_quadrant_a = tf.sign(arousal_pred)  # -1 or 1

    quadrant_penalty = tf.reduce_mean(
        tf.square(true_quadrant_v - pred_quadrant_v) +
        tf.square(true_quadrant_a - pred_quadrant_a)
    )

    return mse_loss + 0.1 * quadrant_penalty  # 10% penalty weight
```

## Optimizers - Which One to Choose?

### 1. Adam (Recommended Default)

```python
from tensorflow.keras.optimizers import Adam

# Standard Adam - good default choice
optimizer = Adam(
    learning_rate=0.001,    # Default learning rate
    beta_1=0.9,            # Exponential decay rate for 1st moment estimates
    beta_2=0.999,          # Exponential decay rate for 2nd moment estimates
    epsilon=1e-7,          # Small constant for numerical stability
    amsgrad=False          # Whether to apply AMSGrad variant
)

# Conservative Adam for stable training
optimizer = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)

# Aggressive Adam for faster convergence (if overfitting is not an issue)
optimizer = Adam(learning_rate=0.002, beta_1=0.85, beta_2=0.99)
```

**When to use Adam:**

- ✅ **Default choice** - works well for most problems
- ✅ Adaptive learning rates per parameter
- ✅ Good for sparse gradients
- ❌ Can generalize worse than SGD on some problems

### 2. RMSprop - Good Alternative

```python
from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(
    learning_rate=0.001,
    rho=0.9,              # Discounting factor for the history/coming gradient
    momentum=0.0,         # Momentum factor
    epsilon=1e-7,         # Small constant for numerical stability
    centered=False        # Whether to normalize by estimated variance
)

# For your emotion task
optimizer = RMSprop(learning_rate=0.0008, rho=0.9, momentum=0.1)
```

**When to use RMSprop:**

- ✅ Good for RNNs/LSTMs (originally designed for them)
- ✅ Handles non-stationary objectives well
- ✅ Memory efficient

### 3. SGD with Momentum - Sometimes Best Generalization

```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(
    learning_rate=0.01,    # Usually higher learning rate needed
    momentum=0.9,          # Momentum factor (almost always use this)
    nesterov=True         # Whether to apply Nesterov momentum
)

# Typical schedule: start high, decay over time
from tensorflow.keras.optimizers.schedules import ExponentialDecay

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
```

**When to use SGD:**

- ✅ Often achieves better final generalization
- ✅ More stable training (less oscillation)
- ❌ Slower convergence
- ❌ Requires more tuning

## Learning Rate Schedules

### 1. Exponential Decay

```python
from tensorflow.keras.optimizers.schedules import ExponentialDecay

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,        # Decay every 1000 steps
    decay_rate=0.96,         # Multiply by 0.96
    staircase=True          # Apply decay at discrete intervals
)
optimizer = Adam(learning_rate=lr_schedule)
```

### 2. Cosine Decay (Often Better)

```python
from tensorflow.keras.optimizers.schedules import CosineDecay

# Total training steps for your data
steps_per_epoch = len(X_normalized) // batch_size  # e.g., 1802 // 32 = 56
total_steps = steps_per_epoch * epochs              # e.g., 56 * 100 = 5600

lr_schedule = CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=total_steps,
    alpha=0.1                # Minimum learning rate = 0.1 * initial
)
optimizer = Adam(learning_rate=lr_schedule)
```

### 3. Reduce on Plateau (Most Practical)

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

# This is a callback, not a schedule
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',      # Metric to monitor
    factor=0.5,             # Multiply LR by 0.5
    patience=5,             # Wait 5 epochs without improvement
    min_lr=1e-6,           # Don't go below this
    verbose=1              # Print when LR changes
)

# Use with model.fit()
history = model.fit(X, y, callbacks=[reduce_lr], ...)
```

## Metrics for Emotion Recognition

```python
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

# Standard regression metrics
metrics = [
    RootMeanSquaredError(name='rmse'),
    MeanAbsoluteError(name='mae'),
]

# Custom emotion metrics
def valence_mae(y_true, y_pred):
    """Mean Absolute Error for valence only"""
    return tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))

def arousal_mae(y_true, y_pred):
    """Mean Absolute Error for arousal only"""
    return tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))

def emotion_accuracy(y_true, y_pred, threshold=0.5):
    """
    Emotion prediction accuracy within threshold
    Considers prediction correct if both valence and arousal are within threshold
    """
    valence_correct = tf.abs(y_true[:, 0] - y_pred[:, 0]) <= threshold
    arousal_correct = tf.abs(y_true[:, 1] - y_pred[:, 1]) <= threshold
    both_correct = tf.logical_and(valence_correct, arousal_correct)
    return tf.reduce_mean(tf.cast(both_correct, tf.float32))

# Complete metrics list
metrics = [
    RootMeanSquaredError(name='rmse'),
    MeanAbsoluteError(name='mae'),
    valence_mae,
    arousal_mae,
    emotion_accuracy
]
```

## Complete Model Compilation

```python
# Compile the model with optimal settings for your data
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=MeanSquaredError(),
    metrics=[
        RootMeanSquaredError(name='rmse'),
        MeanAbsoluteError(name='mae'),
        valence_mae,
        arousal_mae,
        emotion_accuracy
    ]
)

print("Model compiled successfully!")
print(f"Total parameters: {model.count_params():,}")
```

---

# Part 9: Hyperparameter Tuning Guide

## Key Hyperparameters to Tune (In Order of Importance)

### 1. Learning Rate (Most Critical)

```python
# Start with these values and adjust based on training curves
learning_rates = [0.0005, 0.001, 0.002, 0.005]

# Learning rate finder (run small experiment)
for lr in learning_rates:
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)
    final_val_loss = min(history.history['val_loss'])
    print(f"LR: {lr}, Best val_loss: {final_val_loss:.4f}")
```

### 2. Architecture Size

```python
# LSTM units - try different sizes
lstm_units = [64, 128, 256]

# Conv1D filters - try different numbers
conv_filters = [(32, 64), (64, 128), (128, 256)]

# Dense layer sizes
dense_sizes = [(32, 16), (64, 32), (128, 64)]
```

### 3. Regularization Parameters

```python
# Dropout rates
dropout_rates = [0.2, 0.3, 0.4, 0.5]

# L2 regularization
l2_values = [0.001, 0.01, 0.1]

# Batch normalization momentum
bn_momentum = [0.9, 0.99, 0.999]
```

### 4. Training Parameters

```python
# Batch sizes (powers of 2)
batch_sizes = [16, 32, 64]  # 32 is usually good for your dataset size

# Number of epochs
epochs = 100  # Use early stopping, so this is maximum

# Sequence length (if you want to experiment)
sequence_lengths = [400, 500, 600, 800]
```

## Systematic Hyperparameter Search

### Method 1: Grid Search (Comprehensive but Slow)

```python
import itertools

# Define parameter grid
param_grid = {
    'learning_rate': [0.0005, 0.001, 0.002],
    'lstm_units': [64, 128],
    'dropout_rate': [0.3, 0.4],
    'l2_reg': [0.001, 0.01]
}

best_val_loss = float('inf')
best_params = None

for params in itertools.product(*param_grid.values()):
    lr, lstm_units, dropout, l2_reg = params

    # Create model with these parameters
    model = create_custom_crnn(lstm_units=lstm_units, dropout=dropout, l2_reg=l2_reg)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    # Train
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=50, verbose=0,
                       callbacks=[EarlyStopping(patience=10)])

    val_loss = min(history.history['val_loss'])
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {'lr': lr, 'lstm_units': lstm_units, 'dropout': dropout, 'l2_reg': l2_reg}

print(f"Best parameters: {best_params}")
print(f"Best validation loss: {best_val_loss:.4f}")
```

### Method 2: Random Search (More Efficient)

```python
import random

def random_hyperparameters():
    """Generate random hyperparameter combination"""
    return {
        'learning_rate': random.choice([0.0003, 0.0005, 0.001, 0.002, 0.003]),
        'lstm_units': random.choice([64, 96, 128, 160]),
        'conv_filters': random.choice([32, 64, 96, 128]),
        'dropout_rate': random.uniform(0.2, 0.5),
        'l2_reg': random.choice([0.0001, 0.001, 0.01, 0.1]),
        'batch_size': random.choice([16, 32, 64])
    }

# Run random search
n_trials = 20
results = []

for trial in range(n_trials):
    params = random_hyperparameters()

    # Create and train model
    model = create_custom_crnn(**params)
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                  loss='mse', metrics=['mae'])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       batch_size=params['batch_size'], epochs=50, verbose=0,
                       callbacks=[EarlyStopping(patience=10)])

    best_val_loss = min(history.history['val_loss'])
    results.append((params, best_val_loss))
    print(f"Trial {trial+1}: Val loss = {best_val_loss:.4f}")

# Find best result
best_params, best_loss = min(results, key=lambda x: x[1])
print(f"\nBest parameters: {best_params}")
print(f"Best validation loss: {best_loss:.4f}")
```

## Model Architecture Variations to Try

### Variation 1: Deeper CNN

```python
def create_deep_cnn_crnn(sequence_length=600, n_features=40):
    inputs = Input(shape=(sequence_length, n_features))

    # Deeper CNN with residual connections
    x = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)

    # Residual block 1
    residual = x
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = Conv1D(32, 3, activation='linear', padding='same')(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    # Residual block 2
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    residual = Conv1D(64, 1, activation='linear', padding='same')(x)  # Match dimensions
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = Conv1D(64, 3, activation='linear', padding='same')(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    # LSTM layers
    x = LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = LSTM(64, return_sequences=False, dropout=0.3)(x)

    # Output
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(2, activation='linear')(x)

    return Model(inputs, outputs)
```

### Variation 2: Attention Mechanism

```python
from tensorflow.keras.layers import MultiHeadAttention

def create_attention_crnn(sequence_length=600, n_features=40):
    inputs = Input(shape=(sequence_length, n_features))

    # CNN feature extraction
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)  # (batch, 300, 64)

    # Self-attention mechanism
    attention_output = MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        dropout=0.1
    )(x, x)  # Self-attention

    # Add residual connection
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)

    # LSTM processing
    x = LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = LSTM(64, return_sequences=False, dropout=0.3)(x)

    # Output
    x = Dense(64, activation='relu')(x)
    outputs = Dense(2, activation='linear')(x)

    return Model(inputs, outputs)
```

## Training Tips for Your Dataset

### 1. Data Considerations

```python
# Your dataset: 1802 songs
# Recommended splits:
train_size = int(0.7 * 1802)  # 1261 songs for training
val_size = int(0.15 * 1802)   # 270 songs for validation
test_size = 1802 - train_size - val_size  # 271 songs for testing

# Use stratified split to ensure balanced emotion distribution
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X_normalized, y_processed, test_size=test_size,
    random_state=42, stratify=emotion_quadrants  # Create quadrants first
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_size/(train_size+val_size),
    random_state=42, stratify=temp_quadrants
)
```

### 2. Training Strategies

```python
# Early stopping - prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,           # Wait 15 epochs without improvement
    restore_best_weights=True,
    verbose=1
)

# Model checkpointing - save best model
checkpoint = ModelCheckpoint(
    'best_emotion_crnn.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Learning rate reduction
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=100,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1
)
```

This completes the comprehensive CRNN architecture guide! You now have detailed explanations of every component, parameter tuning strategies, and specific recommendations for your emotion recognition dataset with shape (1802, 600, 40).
