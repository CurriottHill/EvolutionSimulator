
import random

class Gene:
    def __init__(self, source_type, source_id, sink_type, sink_id, weight):
        # source_type: 0=neuron, 1=sensor
        # sink_type: 0=neuron, 1=action
        # weight: float in [-4.0, 4.0]
        self.source_type = source_type
        self.source_id = source_id
        self.sink_type = sink_type
        self.sink_id = sink_id
        self.weight = weight
    
    @classmethod
    def random(cls):
        """Create a random gene"""
        return cls(random.randint(0, 1),
            random.randint(0, 127),
            random.randint(0, 1),
            random.randint(0, 127),
            random.uniform(-4.0, 4.0))
    
    def to_int(self):
        """Pack into 32-bit integer"""
        w_int = max(-32768, min(32767, int(self.weight / 4.0 * 32767)))
        w_uint = w_int & 0xFFFF
    
        packed = (self.source_type << 31) | \
            ((self.source_id & 0x7F) << 24) | \
            (self.sink_type << 23) | \
            ((self.sink_id & 0x7F) << 16) | \
            w_uint
        return packed

    def to_hex(self):
        """Pack into 32-bit integer"""
        return f"{self.to_int():08x}"

    @classmethod
    def from_int(cls, packed):
        """Decode from 32-bit integer"""
        source_type = (packed >> 31) & 0x1
        source_id = (packed >> 24) & 0x7F
        sink_type = (packed >> 23) & 0x1
        sink_id = (packed >> 16) & 0x7F
        w_uint = packed & 0xFFFF
        w_int = w_uint if w_uint < 32768 else w_uint - 65536
        weight = (w_int / 32767.0) * 4.0
    
        return cls(source_type, source_id, sink_type, sink_id, weight)

    def copy(self):
        """Return a deep copy of this gene"""
        return Gene(self.source_type, self.source_id, self.sink_type, self.sink_id, self.weight)

    def mutate(self):
        """Flip a single random bit"""
        packed = self.to_int()
        bit = random.randint(0, 31)
        packed ^= (1 << bit)
        mutated = Gene.from_int(packed)
        self.source_type = mutated.source_type
        self.source_id = mutated.source_id
        self.sink_type = mutated.sink_type
        self.sink_id = mutated.sink_id
        self.weight = mutated.weight

