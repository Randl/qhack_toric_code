from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister


def first_step(repr_x, repr_y):
    return (repr_x, repr_y), (repr_x + 1, repr_y)


def second_step(repr_x, repr_y):
    return (repr_x, repr_y), (repr_x + 1, repr_y + 1)


def third_step_left(repr_x, repr_y):
    return (repr_x + 1, repr_y), (repr_x + 2, repr_y)


def third_step_right(repr_x, repr_y):
    return (repr_x + 1, repr_y + 1), (repr_x + 2, repr_y)


def get_plaquette(x, y):
    repr_x, repr_y = x * 2, y
    return (repr_x, repr_y), (repr_x + 1, repr_y), (repr_x + 1, repr_y + 1), (repr_x + 2, repr_y)


def get_star(x, y, size_x, size_y):
    bott_x, bott_y = x * 2 + 1, y
    right_x, right_y = bott_x - 1, bott_y
    left_x, left_y = bott_x - 1, bott_y - 1
    top_x, top_y = bott_x - 2, y

    res = []
    if bott_x < size_x:
        res.append((bott_x, bott_y))
    if top_x > 0:
        res.append((top_x, top_y))
    if right_y < size_y:
        res.append((bott_x, bott_y))
    if left_y > 0:
        res.append((top_x, top_y))
    return res


from scipy.stats import rv_continuous


class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)


class ToricCode:
    def __init__(self, x, y, class_size=4):
        self.x, self.y = x, y
        self.plaquette_x, self.plaquette_y = self.x - 1, self.y // 2
        self.star_x, self.star_y = self.x, self.y // 2 + 1
        # print(self.plaquette_x, self.plaquette_y)
        self.regs = [QuantumRegister(self.x - 1, f'l{lev}') if lev % 2 == 0 else QuantumRegister(self.x, f'l{lev}') for
                     lev in range(self.y)]
        self.c_reg = ClassicalRegister(class_size)
        self.circ = QuantumCircuit(*self.regs, self.c_reg)

        plaquette_reprs_all = []
        for i, r in enumerate(self.regs):
            if i % 2 == 0 and i != self.y - 1:
                self.circ.h(r)
                for j in range(r.size):
                    plaquette_reprs_all.append((i, j))

        self.plaquette_reprs_cols = [[rep for rep in plaquette_reprs_all if rep[1] == i] for i in range(self.x - 1)]
        # print(self.plaquette_reprs_cols)
        order = []
        for i in range((self.x - 1) // 2):
            order.append((i, self.x - 1 - (i + 1)))
        order = order[::-1]

        for col in order[0]:
            for rep in self.plaquette_reprs_cols[col]:
                from_q, to_q = first_step(*rep)

                # print(from_q, to_q)
                self.circ.cnot(self.regs[from_q[0]][from_q[1]], self.regs[to_q[0]][to_q[1]])

        for col in order[0]:
            for rep in self.plaquette_reprs_cols[col]:
                from_q, to_q = second_step(*rep)
                self.circ.cnot(self.regs[from_q[0]][from_q[1]], self.regs[to_q[0]][to_q[1]])

        for i, col_pair in enumerate(order[1:]):

            for rep in self.plaquette_reprs_cols[order[i][0]]:
                from_q, to_q = third_step_left(*rep)
                self.circ.cnot(self.regs[from_q[0]][from_q[1]], self.regs[to_q[0]][to_q[1]])
            for rep in self.plaquette_reprs_cols[order[i][1]]:
                from_q, to_q = third_step_right(*rep)
                self.circ.cnot(self.regs[from_q[0]][from_q[1]], self.regs[to_q[0]][to_q[1]])

            for col in col_pair:
                for rep in self.plaquette_reprs_cols[col]:
                    from_q, to_q = first_step(*rep)

                    # print(from_q, to_q)
                    self.circ.cnot(self.regs[from_q[0]][from_q[1]], self.regs[to_q[0]][to_q[1]])

            for col in col_pair:
                for rep in self.plaquette_reprs_cols[col]:
                    from_q, to_q = second_step(*rep)
                    self.circ.cnot(self.regs[from_q[0]][from_q[1]], self.regs[to_q[0]][to_q[1]])

        for rep in self.plaquette_reprs_cols[order[-1][0]]:
            from_q, to_q = third_step_left(*rep)
            self.circ.cnot(self.regs[from_q[0]][from_q[1]], self.regs[to_q[0]][to_q[1]])
        for rep in self.plaquette_reprs_cols[order[-1][1]]:
            from_q, to_q = third_step_right(*rep)
            self.circ.cnot(self.regs[from_q[0]][from_q[1]], self.regs[to_q[0]][to_q[1]])

    def measure_plaquette(self, x, y):
        qubits = get_plaquette(x, y)
        self.circ.barrier()
        # measure in x basis
        self.circ.h([self.regs[q[0]][q[1]] for q in qubits])
        self.circ.measure([self.regs[q[0]][q[1]] for q in qubits], range(4))
        # print(self.circ)

    def measure_star(self, x, y):
        qubits = get_star(x, y, self.y, self.x)
        if len(qubits) < 4:
            return
        self.circ.barrier()
        self.circ.measure([self.regs[q[0]][q[1]] for q in qubits], range(4))
        # print(self.circ)

    def measure_haar(self, qubits):
        self.circ.barrier()
        for x, y in qubits:
            # https://pennylane.ai/qml/demos/tutorial_haar_measure.html
            # Samples of theta should be drawn from between 0 and pi
            sin_sampler = sin_prob_dist(a=0, b=np.pi)

            phi, lam = 2 * np.pi * np.random.uniform(size=2)  # Sample phi and omega as normal
            theta = sin_sampler.rvs(size=1)  # Sample theta from our new distribution
            self.circ.u(theta, phi, lam, self.regs[x][y])
        self.circ.measure([self.regs[q[0]][q[1]] for q in qubits], range(len(qubits)))

    def measure_pauli(self, qubits, gates):
        self.circ.barrier()
        for (x, y), gate in zip(qubits, gates):
            if gate == 'x':
                self.circ.x(self.regs[x][y])
            if gate == 'y':
                self.circ.y(self.regs[x][y])
            if gate == 'z':
                self.circ.z(self.regs[x][y])
        self.circ.measure([self.regs[q[0]][q[1]] for q in qubits], range(len(qubits)))
