import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, hidden_units)
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        outputs = outputs.transpose(-1, -2)  # back to (batch_size, seq_len, hidden_units)
        return outputs


class FCLayer(nn.Module):
    def __init__(self, input_size,
                  hidden_size,
                  bias=True,
                  batch_norm=False,
                  dropout_rate=0.,
                  activation='relu',
                  use_sigmoid=False):
        super(FCLayer, self).__init__()

        self.use_sigmoid = use_sigmoid

        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm1d(input_size))

        # FC -> activation -> Dropout
        layers.append(nn.Linear(input_size, hidden_size, bias=bias))
        if activation.lower() == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation.lower() == 'prelu':
            layers.append(nn.PReLU())
        else:
            pass
        layers.append(nn.Dropout(p=dropout_rate))

        self.fc = nn.Sequential(*layers)
        if self.use_sigmoid:
            self.output_layer = nn.Sigmoid()

        # weight initialization xavier_normal
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.use_sigmoid else self.fc(x)


if __name__ == "__main__":
    # Test PointWiseFeedForward
    batch_size, seq_len, hidden_units = 32, 100, 50
    pff = PointWiseFeedForward(hidden_units, 0.2)
    inputs = torch.randn(batch_size, seq_len, hidden_units)
    outputs = pff(inputs)
    print(f"PointWiseFeedForward input shape: {inputs.shape}")
    print(f"PointWiseFeedForward output shape: {outputs.shape}")
    
    # Test FCLayer
    fc = FCLayer(100, 50, activation='relu')
    x = torch.randn(32, 100)
    y = fc(x)
    print(f"FCLayer input shape: {x.shape}")
    print(f"FCLayer output shape: {y.shape}")