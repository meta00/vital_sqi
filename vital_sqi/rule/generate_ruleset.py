"""
Class RuleSet contains rules of SQIs and their splitting orders, used to
build a decision flow/tree.
"""
from vital_sqi.rule.rule_class import Rule


class RuleSet:
    """ """

    def __init__(self, rules):
        self.rules = rules

    def __setattr__(self, name, value):
        if name == 'rules':
            if not isinstance(value, dict):
                raise AttributeError('Rule set must be of dict type.')
            order = list(value.keys())
            if not all(isinstance(order, int)):
                raise ValueError('Order must be of type int')
            if sorted(order) != list(range(min(order), max(order) + 1)):
                raise ValueError('Order must contain consecutive numbers')
            if order[0] != 1:
                raise ValueError('Order must start with 1.')
            if not all(isinstance(order, Rule)):
                raise ValueError('Rules must be of class Rule')
        super().__setattr__(name, value)

    def plot_tree(self):
        return

    def export_rules(self):
        return

    def execute(self, x):
        rules = self.rules
        rules = rules.items()
        rules = sorted(rules)
        decision = 'accept'
        for i in rules:
            rule = rules[i][1]
            decision = rule.apply_rule(x)
            if decision == 'reject':
                break
        return decision

