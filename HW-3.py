import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

I = np.array([
    [1, 0],
    [0, 1]
], dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

X1 = np.kron(np.kron(X, I), I)
X2 = np.kron(np.kron(I, X), I)
X3 = np.kron(np.kron(I, I), X)

Z1Z2 = np.kron(np.kron(Z, Z), I)
Z2Z3 = np.kron(np.kron(I, Z), Z)
Z1Z3 = np.kron(np.kron(Z, I), Z)

# Hamiltonians
H_B = X1 + X2 + X3
H_J = Z1Z2 + Z2Z3 + Z1Z3
H_I = 0.1 * H_B + 0.1 * H_J

# |psi(0)> = |000>
psi0 = np.zeros(8, dtype=complex)
psi0[0] = 1

# Exact and approximate evolution and expectation functions
def U_evolution(psi0, t):
    U = expm(-1j * H_I * t)
    return U @ psi0

def V_evolution(psi0, t):
    return (
        expm(-1j * 0.1 * H_J * t)
        @ expm(-1j * 0.1 * H_B * t)
        @ psi0
    )

def W_evolution(psi0, t):
    return (
        expm(-1j * 0.1 * H_J * t / 2)
        @ expm(-1j * 0.1 * H_B * t / 2)
        @ expm(-1j * 0.1 * H_J * t / 2)
        @ expm(-1j * 0.1 * H_B * t / 2)
        @ psi0
    )

def A_evolution(psi0, t):
    return (
        expm(-1j * 0.1 * H_B * t / 2)
        @ expm(-1j * 0.1 * H_J * t)
        @ expm(-1j * 0.1 * H_B * t / 2)
        @ psi0
    )

def expectation(psi, operator):
    return np.real(np.conjugate(np.transpose(psi)) @ operator @ psi)

# Exact and approximate calculations
times = np.arange(0, 11, 1)
H_B_expectations_U, H_J_expectations_U = [], []
H_B_expectations_V, H_J_expectations_V = [], []
H_B_expectations_W, H_J_expectations_W = [], []
H_B_expectations_A, H_J_expectations_A = [], []

for t in times:
    psi = U_evolution(psi0, t)
    H_B_expectations_U.append(expectation(psi, H_B))
    H_J_expectations_U.append(expectation(psi, H_J))

    psi = V_evolution(psi0, t)
    H_B_expectations_V.append(expectation(psi, H_B))
    H_J_expectations_V.append(expectation(psi, H_J))

    psi = W_evolution(psi0, t)
    H_B_expectations_W.append(expectation(psi, H_B))
    H_J_expectations_W.append(expectation(psi, H_J))

    psi = A_evolution(psi0, t)
    H_B_expectations_A.append(expectation(psi, H_B))
    H_J_expectations_A.append(expectation(psi, H_J))


# Part A
plt.plot(times, H_B_expectations_U, marker='o', label=r'$\langle H_B \rangle$')
plt.plot(times, H_J_expectations_U, marker='o', label=r'$\langle H_J \rangle$')
plt.xlabel('Time $t$')
plt.ylabel('Expectation Value')
plt.title('Exact Time Evolution')
plt.legend()
plt.grid(True)
plt.show()

for t, hb, hj in zip(times, H_B_expectations_U, H_J_expectations_U):
    print(f"t = {t}: <H_B> = {hb:.4f}, <H_J> = {hj:.4f}")

# Part B
plt.plot(times, H_B_expectations_U, marker='o', label='U')
plt.plot(times, H_B_expectations_V, marker='o', label='V')
plt.plot(times, H_B_expectations_W, marker='o', label='W')
plt.plot(times, H_B_expectations_A, marker='o', label='A')
plt.xlabel('Time $t$')
plt.ylabel(r'$\langle H_B \rangle$')
plt.title(r'$\langle H_B \rangle$ vs Time')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(times, H_J_expectations_U, marker='o', label='U')
plt.plot(times, H_J_expectations_V, marker='o', label='V')
plt.plot(times, H_J_expectations_W, marker='o', label='W')
plt.plot(times, H_J_expectations_A, marker='o', label='A')
plt.xlabel('Time $t$')
plt.ylabel(r'$\langle H_J \rangle$')
plt.title(r'$\langle H_J \rangle$ vs Time')
plt.legend()
plt.grid(True)
plt.show()
