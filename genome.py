from gene import Gene
import random

class Genome:
    def __init__(self, genes: list[Gene]):
        self.genes = genes
    
    @classmethod
    def random(cls, num_genes):
        genes = []
        for i in range(num_genes):
            genes.append(Gene.random())
        return cls(genes)
    
    def mutate(self, mutation_rate):
        """Point mutations, insertions, deletions"""
        for gene in self.genes:
            # Point mutation - flip a random bit
            if random.random() < mutation_rate:
                gene.mutate()
        
        # Deletion - remove a random gene
        if random.random() < mutation_rate * 0.2 and len(self.genes) > 1:
            self.genes.pop(random.randint(0, len(self.genes) - 1))
        
        # Insertion - add a new random gene
        if random.random() < mutation_rate * 0.2 and len(self.genes) < 50:
            self.genes.append(Gene.random())

    @staticmethod
    def crossover(parent_a, parent_b):
        """Sexual reproduction - combine two genomes"""
        parent_a_genes = random.randint(0, len(parent_a.genes))
        parent_b_genes = random.randint(0, len(parent_b.genes))
        child_genes = [g.copy() for g in parent_a.genes[:parent_a_genes]] + [g.copy() for g in parent_b.genes[parent_b_genes:]]

        if len(child_genes) > 50:
            child_genes = child_genes[:50]
    
        # Need at least 1 gene
        if len(child_genes) == 0:
            child_genes = [Gene.random()]
            
        child = Genome(child_genes)
        return child

    def __str__(self):
        """Pretty print the genome"""
        return ", ".join(gene.to_hex() for gene in self.genes)