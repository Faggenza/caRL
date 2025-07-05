from torchrl.envs.transforms import Transform
import torch
from torch import nn

class CarRacingTransform(Transform):
    """Transform personalizzato per CarRacing che gestisce correttamente i pixel"""

    def __init__(self):
        # CORRETTO: in_keys=["observation"] perché legge dall'ambiente originale
        # out_keys=["pixels"] perché produce la nuova chiave processata
        super().__init__(in_keys=["observation"], out_keys=["pixels"])

    def _reset(self, tensordict, tensordict_reset):
        """Applica la trasformazione durante il reset dell'ambiente"""
        if "observation" in tensordict_reset.keys():
            observation = tensordict_reset["observation"]

            # Converti da uint8 a float32 e normalizza a [0,1]
            if observation.dtype == torch.uint8:
                pixels = observation.float() / 255.0
            else:
                pixels = observation.float()

            # Aggiungi la nuova chiave "pixels"
            tensordict_reset.set("pixels", pixels)

        return tensordict_reset

    def _step(self, tensordict, next_tensordict):
        """Applica la trasformazione durante ogni step dell'ambiente"""
        # CONTROLLO: Verifica se "observation" esiste nel next_tensordict
        if "observation" in next_tensordict.keys():
            observation = next_tensordict["observation"]

            # Converti da uint8 a float32 e normalizza a [0,1]
            if observation.dtype == torch.uint8:
                pixels = observation.float() / 255.0
            else:
                pixels = observation.float()

            # Aggiungi la nuova chiave "pixels"
            next_tensordict.set("pixels", pixels)

        return next_tensordict


class PixelPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pixels):
        # Assicura che i pixel siano float32
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0

        # Flatten dei pixel: [batch, height, width, channels] -> [batch, height*width*channels]
        batch_size = pixels.shape[0] if pixels.dim() > 3 else 1
        if pixels.dim() == 3:  # Se non c'è batch dimension
            pixels = pixels.unsqueeze(0)

        # Flatten preservando la batch dimension
        flattened = pixels.reshape(batch_size, -1)
        return flattened
