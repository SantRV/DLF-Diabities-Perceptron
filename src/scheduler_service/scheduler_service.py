class SchedulerService():
    def __init__(self, optimizer, factor=2, patience=20, min_lr=1e-6, verbose=True):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.lr = None

    def step(self, loss):
        if self.lr is None:
            self.lr = self.optimizer.param_groups[0]['lr']
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.lr = max(self.min_lr, self.lr * self.factor)
                if self.verbose:
                    print(f"Learning rate increased to {self.lr}")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                self.patience_counter = 0
