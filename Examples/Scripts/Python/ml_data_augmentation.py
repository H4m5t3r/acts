import numpy as np
import matplotlib.pyplot as plt


def newBeamlines():
    steps = 40
    space = np.linspace(-1, 1, steps)
    a = np.array(
        [[[space[j], space[i]] for i in range(len(space))] for j in range(len(space))]
    )
    print(a)
    print(a.shape)
    # for row in a[mask]:
    #     for c in row:
    #         plt.scatter(c[0], c[1])
    # plt.savefig("./beamspots.png")
    origo = np.array([0, 0])
    r = 1
    mask = np.array(
        [[np.linalg.norm(beamspot - origo) <= r for beamspot in row] for row in a]
    )
    print(mask)
    print(mask.shape)
    print(a[mask])
    for p in a[mask]:
        plt.scatter(p[0], p[1])
    plt.savefig("./beamspots.png")


newBeamlines()
