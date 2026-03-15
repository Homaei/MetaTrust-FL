import src.models as models
enc_p, head_p = models.MetaTrustModel().get_parameter_counts()
print(f"Total parameters: {enc_p + head_p}")
