"Definition of custom functions for the genetic algorithm"
import numpy as np
from rouge_score import rouge_scorer


class Genetic_Algorithm_custom:
    def __init__(self, vocabulary_weights, sentences) -> None:
        self.vocabulary_weights = vocabulary_weights
        self.sentences = sentences
        self.initial_population = None

    def fitness_function(self, solution):
        scorer = rouge_scorer.RougeScorer([f"rouge1"], use_stemmer=True)
        total_score = []
        for sample in self.sentences:
            solution_words = [self.vocabulary_weights.get(index, "") for index in solution]
            solution_words = [word for word in solution_words if word]
            generated_summary = " ".join(solution_words)

            scores = scorer.score(sample, generated_summary)

            rouge_n_score = scores[f"rouge1"]
            avg_rouge_n_score = (rouge_n_score.fmeasure + rouge_n_score.precision + rouge_n_score.recall) / 3
            total_score.append(avg_rouge_n_score)
        max_value = max(total_score)
        return max_value

    def delete_mutation(self, solution, mutation_probability):
        if np.random.rand() < mutation_probability and len(solution) > 1:
            delete_index = np.random.randint(len(solution))
            solution[delete_index] = 0
        return solution

    def inicialize_random_population(self, population_size, sequence_length):
        initial_population = [np.random.rand(population_size, sequence_length)]
        self.initial_population = np.round(initial_population, 2)

    def get_final_sentence(self, model_output):
        solution_words = [self.vocabulary_weights.get(round(index, 2), "") for index in model_output]
        solution_words = [word for word in solution_words if word]
        sentence = " ".join(solution_words)
        return sentence
