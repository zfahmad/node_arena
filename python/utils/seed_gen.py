from typing import List

import numpy.random as rand


def generate_random_seeds(base_seed: int, num_seeds: int) -> List[int]:
    seed_sequence = rand.SeedSequence(base_seed)
    child_seeds = seed_sequence.spawn(num_seeds)
    return [int(seed.generate_state(1)[0]) for seed in child_seeds]
