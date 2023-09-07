import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize the hidden state with correct dimensions
        h0 = torch.zeros(self.num_layers, self.hidden_size).requires_grad_().to(device)

        # Pass input through the GRU
        out, _ = self.gru(x, h0)

        # Extract the output at the last time step
        out = self.fc(out)

        return out
