# Create a minimally working mock-up for BO.
# feel free to move or remove.

from TextAttack.data_manager import DataManager
from TextAttack.attackers.bo import BayesOptAttacker
from TextAttack.attack_assist.goal.classifier_goal import ClassifierGoal

def main():
    loadVictim = DataManager.loadVictim


    # An example from SST2
    sample_text = "hide new secretions from the parental units "
    sample_label = 0

    # Start with a small victim model
    victim_model = loadVictim("BERT.SST")
    goal = ClassifierGoal(sample_label, False)

    # Instantiate a BO attacker
    attacker = BayesOptAttacker(
        max_iters=100,
        display_interval=1
    )


    print( victim_model.get_prob([sample_text]) )

    attacker.attack(victim_model, sample_text, goal)

if __name__ == "__main__":
    main()