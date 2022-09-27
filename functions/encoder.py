from keras.layers import GRU, LSTM, Dense, Input, RepeatVector, TimeDistributed
from keras.models import Model


def encoder(n_in, latent_dim, input_dim):
    """
    creates a model of encoder & decoder
    :param n_in: dimension inside the encoder
    :param latent_dim: dimension of the latent space
    :param input_dim: input dimension
    :return: encoder & decoder models
    """
    inputs = Input(shape=(n_in, input_dim))
    encoder = LSTM(latent_dim, activation="relu")(inputs)
    decoder = RepeatVector(n_in)(encoder)
    decoder = LSTM(latent_dim, activation="relu", return_sequences=True)(decoder)
    decoder = TimeDistributed(Dense(input_dim))(decoder)
    encoder_model = Model(inputs=inputs, outputs=[encoder])
    model = Model(inputs=inputs, outputs=[decoder])
    model.compile(optimizer="adam", loss="mse")

    return encoder_model, decoder, model