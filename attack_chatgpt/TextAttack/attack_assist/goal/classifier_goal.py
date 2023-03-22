from .base import AttackGoal

class ClassifierGoal(AttackGoal):
    def __init__(self, target, targeted):
        self.target = target
        self.targeted = targeted
    
    @property
    def is_targeted(self):
        return self.targeted

    def check(self, adversarial_sample, prediction):
        if self.targeted:
            return prediction == self.target
        else:
            if 'negative' in prediction:
                prediction = 0
            elif 'negati' in prediction:
                prediction = 0
            elif 'nega' in prediction:
                prediction = 0
            elif 'neg' in prediction:
                prediction = 0
            elif 'negat' in prediction:
                prediction = 0
            elif 'ne' in prediction:
                prediction = 0
            else:
                prediction = 1
            return prediction != self.target