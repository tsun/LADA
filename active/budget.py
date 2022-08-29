import numpy as np

class BudgetAllocator():
    def __init__(self, budget, cfg):
        self.budget = budget
        self.max_epochs = cfg.TRAINER.MAX_EPOCHS
        self.cfg = cfg
        self.build_budgets()

    def build_budgets(self):
        self.budgets = np.zeros(self.cfg.TRAINER.MAX_EPOCHS, dtype=np.int32)
        rounds = self.cfg.ADA.ROUNDS or np.arange(0, self.cfg.TRAINER.MAX_EPOCHS, self.cfg.TRAINER.MAX_EPOCHS // self.cfg.ADA.ROUND)

        for r in rounds:
            self.budgets[r] = self.budget // len(rounds)

        self.budgets[rounds[-1]] += self.budget - self.budgets.sum()

    def get_budget(self, epoch):
        curr_budget = self.budgets[epoch]
        used_budget = self.budgets[:epoch].sum()
        return curr_budget, used_budget
