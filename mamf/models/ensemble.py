import torch
from core import MAMFBase

class MAMFEnsemble(MAMFBase):
    def __init__(self, text_model, audio_model, lower_bound=0.3, upper_bound=0.7):
        """
        Args:
            text_model: text model to use
            audio_model: audio model to use
            lower_bound: lower bound for the weight
            upper_bound: upper bound for the weight
        """
        super().__init__()
        self.text_model = text_model
        self.audio_model = audio_model
        # weight to balance the two models, 0 because (tanh(0)+1)/2 = 0.5 => equal weight to both models
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def forward(self, text_features, text_attentions, audio_features, audio_attentions, **kwargs):
        """
        Forward pass of the model
        Args:
            texts: texts to use
            audio_features: audio features to use
            audio_attentions: audio attentions to use
        """
        text_logits = self.text_model(text_features, text_attentions)
        audio_logits = self.audio_model(audio_features, audio_attentions)
        
        text_probabilities = torch.nn.functional.softmax(text_logits)
        audio_probabilities = torch.nn.functional.softmax(audio_logits)
        
        # coefficient to balance the two models based on weight learned
        # (tanh + 1) / 2 to have values in [0,1]
        coefficient = (torch.tanh(self.weight) + 1) / 2
        # next step is to have values in [lower_bound, upper_bound] to avoid too much imbalance
        coefficient = coefficient * (self.upper_bound - self.lower_bound) + self.lower_bound
        
        return coefficient * text_probabilities + (1 - coefficient) * audio_probabilities