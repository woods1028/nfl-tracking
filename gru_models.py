from tensorflow.keras.models import Model
from tensorflow.keras.layers import Masking, Dense, Dropout, Input, Concatenate, GRU

def gru_model(transition_shape, defender_shape, offense_shape):
    """
    Build a model using GRU units with CPU-compatible settings.
    
    Parameters:
    transition_shape: Shape of transition features input
    defender_shape: Shape of defender tracking features input
    
    Returns:
    Compiled model
    """
    
    # Zone transitions input
    transition_input = Input(shape=transition_shape, name='transition_input')
    transition_gru = GRU(64, return_sequences=True, recurrent_activation='sigmoid', reset_after=False)(transition_input)
    transition_gru = Dropout(0.3)(transition_gru)
    transition_gru = GRU(32, recurrent_activation='sigmoid', reset_after=False)(transition_gru)
    transition_features = Dense(32, activation='relu')(transition_gru)
    
    # Defender tracking input
    defender_input = Input(shape=defender_shape, name='defender_input')
    defender_gru = GRU(128, return_sequences=True, recurrent_activation='sigmoid', reset_after=False)(defender_input)
    defender_gru = Dropout(0.3)(defender_gru)
    defender_gru = GRU(64, recurrent_activation='sigmoid', reset_after=False)(defender_gru)
    defender_features = Dense(64, activation='relu')(defender_gru)

    # Defender tracking input
    offense_input = Input(shape=offense_shape, name='offense_input')
    offense_gru = GRU(128, return_sequences=True, recurrent_activation='sigmoid', reset_after=False)(offense_input)
    offense_gru = Dropout(0.3)(offense_gru)
    offense_gru = GRU(64, recurrent_activation='sigmoid', reset_after=False)(offense_gru)
    offense_features = Dense(64, activation='relu')(offense_gru)
    
    # Combine both feature streams
    combined = Concatenate()([transition_features, defender_features, offense_features])
    
    # Classification layers
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.1)(x)
    #x = Dense(4, activation='relu')(x)
    #x = Dropout(0.2) (x)
    
    # Final output - binary classification for single-high vs not
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[transition_input, defender_input, offense_input], outputs=output)
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def gru_model_w_mask(transition_feature_dim, defender_feature_dim, offense_feature_dim, mask_value):
    """
    Build a model using GRU units that can handle variable-length sequences using RaggedTensors.
    """
    
    # Zone transitions input
    transition_input = Input(shape = (None, transition_feature_dim), name='transition_input')
    transition_masked = Masking(mask_value = mask_value)(transition_input)
    transition_gru = GRU(64, return_sequences=True, recurrent_activation='sigmoid', reset_after=False)(transition_masked)
    transition_gru = Dropout(0.3)(transition_gru)
    transition_gru = GRU(32, recurrent_activation='sigmoid', reset_after=False)(transition_gru)
    transition_features = Dense(32, activation='relu')(transition_gru)
    
    # Defender tracking input
    defender_input = Input(shape = (None, defender_feature_dim), name='defender_input')
    defender_masked = Masking(mask_value = mask_value)(defender_input)
    defender_gru = GRU(128, return_sequences=True, recurrent_activation='sigmoid', reset_after=False)(defender_masked)
    defender_gru = Dropout(0.3)(defender_gru)
    defender_gru = GRU(64, recurrent_activation='sigmoid', reset_after=False)(defender_gru)
    defender_features = Dense(64, activation='relu')(defender_gru)

    # Defender tracking input
    offense_input = Input(shape = (None, offense_feature_dim), name='offense_input')
    offense_masked = Masking(mask_value = mask_value)(offense_input)
    offense_gru = GRU(128, return_sequences=True, recurrent_activation='sigmoid', reset_after=False)(offense_masked)
    offense_gru = Dropout(0.3)(offense_gru)
    offense_gru = GRU(64, recurrent_activation='sigmoid', reset_after=False)(offense_gru)
    offense_features = Dense(64, activation='relu')(offense_gru)

    # Combine both feature streams
    combined = Concatenate()([transition_features, defender_features, offense_features])
    
    # Classification layers
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.1)(x)
    #x = Dense(4, activation='relu')(x)
    #x = Dropout(0.2) (x)
    
    # Final output - binary classification for single-high vs not
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[transition_input, defender_input, offense_input], outputs=output)
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model