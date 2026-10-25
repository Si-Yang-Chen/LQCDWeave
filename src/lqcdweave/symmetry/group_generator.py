from sympy import Matrix, sqrt, I, S
import numpy as np
import sympy as sp

Fermion_generator = {
    "c4y": Matrix(
        [
            [sqrt(2) / 2, -sqrt(2) / 2, 0, 0],
            [sqrt(2) / 2, sqrt(2) / 2, 0, 0],
            [0, 0, sqrt(2) / 2, -sqrt(2) / 2],
            [0, 0, sqrt(2) / 2, sqrt(2) / 2],
        ]
    ),
    "c4z": Matrix(
        [
            [sqrt(2) * (1 - I) / 2, 0, 0, 0],
            [0, sqrt(2) * (1 + I) / 2, 0, 0],
            [0, 0, sqrt(2) * (1 - I) / 2, 0],
            [0, 0, 0, sqrt(2) * (1 + I) / 2],
        ]
    ),
    "inviden": Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
}

OhD_generator = {
    "A_1": {
        "c4y": Matrix([[1]]),
        "c4z": Matrix([[1]]),
    },
    "A_2": {
        "c4y": Matrix([[-1]]),
        "c4z": Matrix([[-1]]),
    },
    "E": {
        "c4y": Matrix([[S(1) / 2, sqrt(3) / 2], [sqrt(3) / 2, -S(1) / 2]]),
        "c4z": Matrix([[-1, 0], [0, 1]]),
    },
    "T_1": {
        "c4y": Matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        "c4z": Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    },
    "T_2": {
        "c4y": Matrix([[0, 0, -1], [0, -1, 0], [1, 0, 0]]),
        "c4z": Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]),
    },
    "G_1": {
        "c4y": Matrix([[sqrt(2) / 2, -sqrt(2) / 2], [sqrt(2) / 2, sqrt(2) / 2]]),
        "c4z": Matrix([[sqrt(2) * (1 - I) / 2, 0], [0, sqrt(2) * (1 + I) / 2]]),
    },
    "G_2": {
        "c4y": Matrix([[-sqrt(2) / 2, sqrt(2) / 2], [-sqrt(2) / 2, -sqrt(2) / 2]]),
        "c4z": Matrix([[sqrt(2) * (-1 + I) / 2, 0], [0, sqrt(2) * (-1 - I) / 2]]),
    },
    "H": {
        "c4y": Matrix(
            [
                [sqrt(2) / 4, -sqrt(6) / 4, sqrt(6) / 4, -sqrt(2) / 4],
                [sqrt(6) / 4, -sqrt(2) / 4, -sqrt(2) / 4, sqrt(6) / 4],
                [sqrt(6) / 4, sqrt(2) / 4, -sqrt(2) / 4, -sqrt(6) / 4],
                [sqrt(2) / 4, sqrt(6) / 4, sqrt(6) / 4, sqrt(2) / 4],
            ]
        ),
        "c4z": Matrix(
            [
                [sqrt(2) * (-1 - I) / 2, 0, 0, 0],
                [0, sqrt(2) * (1 - I) / 2, 0, 0],
                [0, 0, sqrt(2) * (1 + I) / 2, 0],
                [0, 0, 0, sqrt(2) * (-1 + I) / 2],
            ]
        ),
    },
}

Dic4_generator = {
    "A_1": {
        "c4z": Matrix([[1]]),
        "invc2y": Matrix([[1]]),
    },
    "A_2": {
        "c4z": Matrix([[1]]),
        "invc2y": Matrix([[-1]]),
    },
    "B_1": {
        "c4z": Matrix([[-1]]),
        "invc2y": Matrix([[1]]),
    },
    "B_2": {
        "c4z": Matrix([[-1]]),
        "invc2y": Matrix([[-1]]),
    },
    "E": {
        "c4z": Matrix([[0, -1], [1, 0]]),
        "invc2y": Matrix([[1, 0], [0, -1]]),
    },
    "G_1": {
        "c4z": Matrix([[sqrt(2) * (1 - I) / 2, 0], [0, sqrt(2) * (1 + I) / 2]]),
        "invc2y": Matrix([[0, -S((1))], [S(1), 0]]),
    },
    "G_2": {
        "c4z": Matrix([[-sqrt(2) * (1 - I) / 2, 0], [0, -sqrt(2) * (1 + I) / 2]]),
        "invc2y": Matrix([[0, -S(1)], [S(1), 0]]),
    },
}

Dic2_generator = {
    "A_1": {
        "c2e": Matrix([[1]]),
        "invc2f": Matrix([[1]]),
    },
    "A_2": {
        "c2e": Matrix([[1]]),
        "invc2f": Matrix([[-1]]),
    },
    "B_1": {
        "c2e": Matrix([[-1]]),
        "invc2f": Matrix([[1]]),
    },
    "B_2": {
        "c2e": Matrix([[-1]]),
        "invc2f": Matrix([[-1]]),
    },
    "G": {
        "c2e": Matrix(
            [[-sqrt(2) * I / 2, -sqrt(2) / 2], [sqrt(2) / 2, sqrt(2) * I / 2]]
        ),
        "invc2f": Matrix(
            [[sqrt(2) * I / 2, -sqrt(2) / 2], [sqrt(2) / 2, -sqrt(2) * I / 2]]
        ),
    },
}

Dic3_generator = {
    "A_1": {
        "c3delta": Matrix([[1]]),
        "invc2b": Matrix([[1]]),
    },
    "A_2": {
        "c3delta": Matrix([[1]]),
        "invc2b": Matrix([[-1]]),
    },
    "F_1": {
        "c3delta": Matrix([[-1]]),
        "invc2b": Matrix([[I]]),
    },
    "F_2": {
        "c3delta": Matrix([[-1]]),
        "invc2b": Matrix([[-I]]),
    },
    "E": {
        "c3delta": Matrix([[-S(1) / 2, sqrt(3) / 2], [-sqrt(3) / 2, -S(1) / 2]]),
        "invc2b": Matrix([[-1, 0], [0, 1]]),
    },
    "G": {
        "c3delta": Matrix(
            [
                [S(1) / 2 - I / 2, -S(1) / 2 - I / 2],
                [S(1) / 2 - I / 2, S(1) / 2 + I / 2],
            ]
        ),
        "invc2b": Matrix([[0, sqrt(2) * (1 - I) / 2], [sqrt(2) * (-1 - I) / 2, 0]]),
    },
}

C4_generator1 = {
    "A_1": {
        "invc2f": Matrix([[1]]),
    },
    "A_2": {
        "invc2f": Matrix([[-1]]),
    },
    "F_1": {
        "invc2f": Matrix([[I]]),
    },
    "F_2": {
        "invc2f": Matrix([[-I]]),
    },
}

C4_generator2 = {
    "A_1": {
        "invc2x": Matrix([[1]]),
    },
    "A_2": {
        "invc2x": Matrix([[-1]]),
    },
    "F_1": {
        "invc2x": Matrix([[I]]),
    },
    "F_2": {
        "invc2x": Matrix([[-I]]),
    },
}

irrep_generators={
    "0,0,0":OhD_generator,
    "0,0,1":Dic4_generator,
    "0,1,1":Dic2_generator,
    "1,1,1":Dic3_generator,
    "2,1,1":C4_generator1,
    "0,1,2":C4_generator2,
}

from .hardcoded_rep import refRotateDict


falvor2irrep = {
    1: [np.array([[1, 0], [0, 1]]) / sp.sqrt(2)],
    3: [
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 0], [1, 0]]),
        np.array([[1, 0], [0, -1]]) / sp.sqrt(2),
    ],
}

falvor3irrep = {
    1: [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / sp.sqrt(3)],
    8: [
        np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) / sp.sqrt(2),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sp.sqrt(6),
    ],
}
