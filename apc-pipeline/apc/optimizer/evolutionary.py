from __future__ import annotations
import random
from copy import deepcopy
from apc.core.genome import PromptGenome

class GenomeMutator:
    def mutate(self, genome: PromptGenome) -> PromptGenome:
        g = deepcopy(genome)
        t = random.choice(["toggle_examples","change_example_count","change_reasoning_strategy","toggle_verification","change_output_strictness","change_constraint_placement","change_verbosity"])
        if t=="toggle_examples": g.examples.enabled = not g.examples.enabled
        elif t=="change_example_count": g.examples.count = random.choice([0,1,2,3])
        elif t=="change_reasoning_strategy": g.reasoning.strategy = random.choice(["none","brief_plan","hidden_analysis","structured_checklist"])
        elif t=="toggle_verification": g.verification.enabled = not g.verification.enabled
        elif t=="change_output_strictness": g.output.strictness = random.choice(["low","medium","high"])
        elif t=="change_constraint_placement": g.constraints.placement = random.choice(["top","after_goal","bottom"])
        elif t=="change_verbosity": g.style.verbosity = random.choice(["low","medium","high"])
        return g

class GenomeSelector:
    def select_elite(self, genomes: list[PromptGenome], scores: list[float], top_k=5):
        paired = sorted(zip(genomes, scores), key=lambda x: x[1], reverse=True)
        return [g for g,_ in paired[:top_k]]

class EvolutionaryOptimizer:
    def __init__(self, compiler, evaluator, mutator=None, selector=None):
        self.compiler=compiler; self.evaluator=evaluator
        self.mutator=mutator or GenomeMutator()
        self.selector=selector or GenomeSelector()
    def optimize(self, task_spec, model_profile, seed_genomes, generations=3, population_size=20):
        pop = self._init_pop(seed_genomes, population_size)
        best=None; best_score=-1
        for _ in range(generations):
            scores=[]
            for genome in pop:
                prompt=self.compiler.compile(genome, task_spec, model_profile)
                score=self.evaluator.evaluate_prompt(prompt) if hasattr(self.evaluator,'evaluate_prompt') else random.random()
                scores.append(score)
                if score>best_score: best_score=score; best=genome
            elite=self.selector.select_elite(pop, scores, top_k=5)
            pop=self._next_pop(elite, population_size)
        return best, best_score
    def _init_pop(self, seeds, n):
        pop=list(seeds)
        while len(pop)<n: pop.append(self.mutator.mutate(random.choice(seeds)))
        return pop[:n]
    def _next_pop(self, elite, n):
        pop=list(elite)
        while len(pop)<n: pop.append(self.mutator.mutate(random.choice(elite)))
        return pop[:n]
