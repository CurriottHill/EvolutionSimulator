import math

class Brain:
    def __init__(self, genome, num_sensors, num_actions, num_internal):
        """
        Takes a genome (list of genes) and builds a working brain.
        
        genome = Genome object from genome.py
        num_sensors = how many sensors exist (e.g. 8)
        num_actions = how many actions exist (e.g. 6)
        num_internal = how many internal neurons (e.g. 4)
        """
        
        self.num_sensors = num_sensors
        self.num_actions = num_actions
        self.num_internal = num_internal
        
        # Map gene IDs to valid indices using modulo
        self.connections = []
        
        for gene in genome.genes:
            if gene.source_type == 1:  # sensor
                source_id = gene.source_id % num_sensors
            else:  # neuron
                source_id = gene.source_id % num_internal
            
            if gene.sink_type == 1:  # action
                sink_id = gene.sink_id % num_actions
            else:  # neuron
                sink_id = gene.sink_id % num_internal
            
            self.connections.append((
                gene.source_type,
                source_id,
                gene.sink_type,
                sink_id,
                gene.weight
            ))
        
        # Neuron memory - stores previous outputs
        self.neuron_outputs = {}
        for i in range(num_internal):
            self.neuron_outputs[i] = 0.0
    

    def feed_forward(self, sensor_values):
        """
        Run one step of thinking.
        
        sensor_values = dict like {0: 0.8, 1: 0.3, 2: 0.5}
                        where key = sensor ID, value = what that sensor reads
        
        returns = dict like {0: -0.84, 1: 0.2}
                  where key = action ID, value = how strongly to do that action
        """
        
        # Create accumulators
        neuron_accumulators = {}
        for i in range(self.num_internal):
            neuron_accumulators[i] = 0.0
        
        action_accumulators = {}
        for i in range(self.num_actions):
            action_accumulators[i] = 0.0
        
        # Process each connection
        for (source_type, source_id, sink_type, sink_id, weight) in self.connections:
            if source_type == 1:  # sensor
                source_value = sensor_values.get(source_id, 0.0)
            else:  # neuron - use previous step's output
                source_value = self.neuron_outputs.get(source_id, 0.0)
            
            signal = source_value * weight
            
            if sink_type == 1:  # action
                action_accumulators[sink_id] += signal
            else:  # neuron
                neuron_accumulators[sink_id] += signal
        
        # Squash with tanh [-1, 1] and save for next step
        for i in range(self.num_internal):
            self.neuron_outputs[i] = math.tanh(neuron_accumulators[i])
        
        # Squash actions and return
        action_outputs = {}
        for i in range(self.num_actions):
            action_outputs[i] = math.tanh(action_accumulators[i])
        
        return action_outputs