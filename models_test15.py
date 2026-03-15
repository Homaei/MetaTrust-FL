import src.models as models
enc_p, head_p = models.MetaTrustModel().get_parameter_counts()
print(f"LSTM: {enc_p}")
print(f"MLP: {head_p}")
print(f"Total parameters: {enc_p + head_p}")
