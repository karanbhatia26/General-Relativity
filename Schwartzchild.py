from sympy import symbols, diff, Matrix, simplify, sin

M, t, r, theta, phi = symbols("M t r theta phi")
coor = [t, r, theta, phi]


def Tensor(i, j):
    g00 = 1 - 2 * M / r
    g01 = 0
    g02 = 0
    g03 = 0
    g10 = 0
    g11 = -1 / (1 - 2 * M / r)
    g12 = 0
    g13 = 0
    g20 = 0
    g21 = 0
    g22 = -r**2
    g23 = 0
    g30 = 0
    g31 = 0
    g32 = 0
    g33 = -(r**2 * sin(theta)**2)
    return [[g00, g01, g02, g03],
            [g10, g11, g12, g13],
            [g20, g21, g22, g23],
            [g30, g31, g32, g33]][i][j]


def dm(i, j, k):
    return diff(Tensor(i, j), coor[k])


def im(i, j):
    K = Matrix([[Tensor(0, 0), Tensor(0, 1), Tensor(0, 2), Tensor(0, 3)],
                 [Tensor(1, 0), Tensor(1, 1), Tensor(1, 2), Tensor(1, 3)],
                 [Tensor(2, 0), Tensor(2, 1), Tensor(2, 2), Tensor(2, 3)],
                 [Tensor(3, 0), Tensor(3, 1), Tensor(3, 2), Tensor(3, 3)]]
                )
    return K.inv()[i, j]


def gamma(i, j, k):
    s = 0
    for l in range(4):
        s += 0.5 * im(i, l) * (dm(k, l, j) + dm(l, j, k) - dm(j, k, l))
    return simplify(s)


for a in range(4):
    for b in range(4):
        for c in range(4):
            if gamma(a, b, c) == 0:
                pass
            else:
                print("[", a, b, c, "]", gamma(a, b, c))
