from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister


class ToricCodeMixed:
    def __init__(self, x, y, classical_bit_count=4, ancillas_count=0):
        """

        :param x: Column count. In case of matching boundary condition even rows has one less qubit
        :param y: Row count
        :param classical_bit_count: Number of classical bits
        """
        self.x, self.y = x, y
        self.plaquette_x, self.plaquette_y = self.x // 2 - 1, self.y + 2
        self.star_x, self.star_y = self.x - self.plaquette_x, self.y

        # first coordinate is row index, second is column index
        self.regs = [QuantumRegister(self.x, f'l{lev}') for lev in range(self.y)]
        self.c_reg = ClassicalRegister(classical_bit_count)
        self.circ = QuantumCircuit(*self.regs, self.c_reg)
        # TODO
