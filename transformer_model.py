import torch
import torch.nn as nn

# transformer-based classifier
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim = 5, model_dim = 64, num_heads = 2, num_layers = 2, num_classes= 3, dropout = 0.1):
        super(TransformerClassifier, self).__init__()

        # linear transformation to scale input features to model size 
        self.input_proj = nn.Linear(input_dim, model_dim)

        # positional embedding 
        self.pos_embedding = nn.Parameter(torch.randn(1, 20, model_dim))

        # transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model= model_dim,
            nhead= num_heads,
            dim_feedforward= 128,
            dropout= dropout,
            batch_first= True # data shape (batch, ...)
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers= num_layers)

        # mean pooling to summarize the output sequence representations
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # output layer: classifier
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, num_classes) # output
        )

    def forward(self, X):
        X = self.input_proj(X) #(batch, seq, model_dim)

        X = X + self.pos_embedding # position info (batch, seq, model_dim)

        X = self.transformer_encoder(X) #(batch, seq, model_dim)

        X = X.transpose(1, 2)  # (batch, model_dim, seq)
        X = self.pooling(X).squeeze(-1)  # (batch, model_dim)

        # classifier layer
        out = self.classifier(X) # (batch, num_classes)

        return out

if __name__ == "__main__":
    dummy_input = torch.randn(32, 20, 5)
    model = TransformerClassifier()
    output = model(dummy_input)
    print(output.shape)

