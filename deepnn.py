import neural_net as nn
import optimisers as opt


class DNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, act_type='sigmoid', init_method="random"):
        super().__init__()
        self.n = len(hidden_dims)
        self.layers = []
        prev_dim = in_dim
        # set activation
        if act_type == 'sigmoid':
            self.act = nn.sigmoid
            self.act_backprop = nn.sigmoid_backprop
        elif act_type == 'ReLU':
            self.act = nn.relu
            self.act_backprop = nn.relu_backprop
        elif act_type == 'tanh':
            self.act = nn.tanh
            self.act_backprop = nn.tanh_backprop
        elif act_type == 'identity':
            self.act = nn.identity
            self.act_backprop = nn.identity_backprop
        else:
            raise NotImplementedError
        # create the hidden layers
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim, init_method))
            prev_dim = h_dim
        self.layers.append(nn.Linear(prev_dim, out_dim, init_method))

    def forward(self, x):
        self.h = []
        self.a = []
        if isinstance(self.optimiser, opt.NAG):
            # lookahead step for NAG optimiser
            self.optimiser.lookahead(self)
        for i in range(self.n):
            # compute the Wx + b
            x = self.layers[i].forward(x)
            self.h.append(x)
            # compute activation
            x = self.act(x)
            self.a.append(x)
        x = self.layers[self.n].forward(x)
        self.h.append(x)
        # final output
        return nn.softmax(x)

    def backprop(self, grad_prev):
        for i in range(self.n, 0, -1):
            # wrt prev layer h
            gx = self.layers[i].backprop(grad_prev)
            # wrt current layer activation a
            grad_prev = self.act_backprop(gx, self.a[i-1])
        self.layers[0].backprop(grad_prev)
