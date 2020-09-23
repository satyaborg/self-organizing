import torch.nn as nn

class Autoencoder(nn.Module):
    """Autoencoder with linear layers"""
    def __init__(self, input_size, hidden_units, dropout=0):
        super(Autoencoder, self).__init__()

        self.input_size = input_size
        self.hidden_units = hidden_units
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=self.input_size, out_features=self.hidden_units, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=self.hidden_units, out_features=self.input_size, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class LogisticRegressionModel(nn.Module):
    """Logistic Regressor"""
    def __init__(self, in_dim, classes, hidden_layers, activation_fn):
        """
        in_dim:         This is the input dimension of the Logistic Classifier
        classes:        The number of classes,i.e. the output dimensions of the classifier
        hidden_layers:  A python list specifying the no. of neurons in each hidden layer. 
                        Length of this list will give the no. of hidden layers.
                        Empty list [] indicates 'no hidden layers'.
        """
        
        super(LogisticRegressionModel, self).__init__()
        
        # activation_fn = nn.ReLU(inplace = True)
        n = len(hidden_layers)
        if n > 0:   
            layers = [[nn.Linear(in_dim, hidden_layers[0]), activation_fn]]
            layers.extend([[nn.Linear(hidden_layers[i], hidden_layers[i+1]), activation_fn] for i in range(n-1)])

            # Put everything into a single proper list
            self.linear = [elem for lists in layers for elem in lists]
            self.linear.append(nn.Linear(hidden_layers[n-1], classes))
        
            # Unpack the list arguments into the nn.Sequential container
            self.linear = nn.Sequential(*self.linear)
        else:
            # vanilla logistic NN
            self.linear = nn.Linear(in_dim, classes)   # i.e. no hidden layers
            
    def forward(self, x):
        y = self.linear(x) # F.sigmoid(self.linear(x))
        return y