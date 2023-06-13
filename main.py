import plotly.graph_objects as go
import numpy as np


class Parallelepiped:
    def __init__(self, k, l, m):
        self.k = k
        self.l = l
        self.m = m


class Subdivisions:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


def printMatrix(matrix):
    for row in matrix:
        print('[', end='')
        for elem in row:
            print(f' {elem}', end='')
        print(' ]')


def getNodeCoordinates(parallelepiped, subdivisions):
    coordinates = []

    dx = parallelepiped.k / subdivisions.a
    dy = parallelepiped.l / subdivisions.b
    dz = parallelepiped.m / subdivisions.c

    # generate corner nodes
    for i in range(subdivisions.a + 1):
        for j in range(subdivisions.b + 1):
            for k in range(subdivisions.c + 1):
                x = i * dx
                y = j * dy
                z = k * dz
                coordinates.append([x, y, z])

    # generate edge nodes
    for i in range(subdivisions.a + 1):
        for j in range(subdivisions.b + 1):
            for k in range(subdivisions.c):
                x = i * dx
                y = j * dy
                z = k * dz + dz / 2
                coordinates.append([x, y, z])
    for i in range(subdivisions.a + 1):
        for j in range(subdivisions.b):
            for k in range(subdivisions.c + 1):
                x = i * dx
                y = j * dy + dy / 2
                z = k * dz
                coordinates.append([x, y, z])
    for i in range(subdivisions.a):
        for j in range(subdivisions.b + 1):
            for k in range(subdivisions.c + 1):
                x = i * dx + dx / 2
                y = j * dy
                z = k * dz
                coordinates.append([x, y, z])
    coordinates = sorted(coordinates, key=lambda c: (c[2], c[1], c[0]))
    new_coord = []
    for i in range(len(coordinates)):
        new_coord.append((i + 1, coordinates[i]))
    return new_coord


def parallelepipedCorners(parallelepiped, subdivisions):
    dx = parallelepiped.k / subdivisions.a
    dy = parallelepiped.l / subdivisions.b
    dz = parallelepiped.m / subdivisions.c
    parallelepipedCorner = []
    for k in range(subdivisions.c):
        for j in range(subdivisions.b):
            for i in range(subdivisions.a):
                index = k * subdivisions.b * subdivisions.a + j * subdivisions.a + i + 1
                corner = []
                corner.append([i * dx, j * dy, k * dz])
                corner.append([(i + 1) * dx, j * dy, k * dz])
                corner.append([(i + 1) * dx, (j + 1) * dy, k * dz])
                corner.append([i * dx, (j + 1) * dy, k * dz])

                corner.append([i * dx, j * dy, (k + 1) * dz])
                corner.append([(i + 1) * dx, j * dy, (k + 1) * dz])
                corner.append([(i + 1) * dx, (j + 1) * dy, (k + 1) * dz])
                corner.append([i * dx, (j + 1) * dy, (k + 1) * dz])

                corner.append([(i + 0.5) * dx, j * dy, k * dz])
                corner.append([(i + 1) * dx, (j + 0.5) * dy, k * dz])
                corner.append([(i + 0.5) * dx, (j + 1) * dy, k * dz])
                corner.append([i * dx, (j + 0.5) * dy, k * dz])

                corner.append([i * dx, j * dy, (k + 0.5) * dz])
                corner.append([(i + 1) * dx, j * dy, (k + 0.5) * dz])
                corner.append([(i + 1) * dx, (j + 1) * dy, (k + 0.5) * dz])
                corner.append([i * dx, (j + 1) * dy, (k + 0.5) * dz])

                corner.append([(i + 0.5) * dx, j * dy, (k + 1) * dz])
                corner.append([(i + 1) * dx, (j + 0.5) * dy, (k + 1) * dz])
                corner.append([(i + 0.5) * dx, (j + 1) * dy, (k + 1) * dz])
                corner.append([i * dx, (j + 0.5) * dy, (k + 1) * dz])


                slides = []

                slides.append(
                    [corner[0], corner[1], corner[5], corner[4], corner[8], corner[13], corner[16], corner[12]])
                slides.append(
                    [corner[1], corner[2], corner[6], corner[5], corner[9], corner[14], corner[17], corner[13]])
                slides.append(
                    [corner[2], corner[3], corner[7], corner[6], corner[10], corner[15], corner[18], corner[14]])
                slides.append(
                    [corner[0], corner[3], corner[7], corner[4], corner[11], corner[15], corner[19], corner[12]])
                slides.append(
                    [corner[0], corner[1], corner[2], corner[3], corner[8], corner[9], corner[10], corner[11]])
                slides.append(
                    [corner[4], corner[5], corner[6], corner[7], corner[16], corner[17], corner[18], corner[19] ])

                parallelepipedCorner.append((index, corner, slides))

    return parallelepipedCorner


def getIndices(figuresCoordinates, coordinates):
    indices = []
    for fig in figuresCoordinates:
        indexList = []
        for corner in fig[1]:
            for coord in coordinates:
                if corner == coord[1]:
                    indexList.append(coord[0])
        indices.append((fig[0], indexList))
    return indices


def findMaxDifference(indices):
    max_difference = 0
    for idx in indices:
        index_list = idx[1]
        if len(index_list) > 1:
            min_val = min(index_list)
            max_val = max(index_list)
            difference = abs(max_val - min_val)
            if difference > max_difference:
                max_difference = difference
    return max_difference


def edgeCoordinates(findFiguresWithZeroZ, slides, side):
    val = []
    for i in findFiguresWithZeroZ:
        val.append(slides[i - 1][side - 1])
    return val













def createNG(NT):
    NG_list = []
    for item in NT:
        sublist = item[1]
        NG_list.append(np.max(sublist) - np.min(sublist))
    return int(3 * (max(NG_list) + 1))



def calculateZp(nx,ny,nz):
    zp = [[0] * 2 for _ in range(nx * ny)]

    FENumber = (nx * ny * nz) - (nx * ny)
    for i in range(nx * ny):
        zp[i][0] = FENumber + 1
        zp[i][1] = 6
        FENumber += 1

    return zp


def calculateZu(nx,ny):
    zu = [[0] * 2 for _ in range(nx * ny)]

    FENumber = 0
    for i in range(nx * ny):
        zu[i][0] = FENumber + 1
        zu[i][1] = 5
        FENumber += 1

    return zu




def calculate_dfiabg():
    DFIABG = np.zeros((27, 3, 20))
    AiBiGi = np.array([[-1, 1, 1, -1, -1, 1, 1, -1, 0, 1, 0, -1, -1, 1, 1, -1, 0, 1, 0, -1],
                       [1, 1, -1, -1, 1, 1, -1, -1, 1, 0, -1, 0, 1, 1, -1, -1, 1, 0, -1, 0],
                       [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]])
    AiBiGi = np.array([list(round(val) for val in row) for row in zip(*AiBiGi)])
    ABG = {-np.sqrt(0.6), 0, np.sqrt(0.6)}
    counter = 0
    gaussPoints = np.zeros((27, 3))
    for g in ABG:
        for b in ABG:
            for a in ABG:
                for i in range(0, 8):
                    a_i = AiBiGi[i, 0]
                    b_i = AiBiGi[i, 1]
                    g_i = AiBiGi[i, 2]
                    dAlpha = 0.125 * (1 + b * b_i) * (1 + g * g_i) * (a_i * (2 * a * a_i + b * b_i + g * g_i - 1))
                    dBeta = 0.125 * (1 + a * a_i) * (1 + g * g_i) * (b_i * (a * a_i + 2 * b * b_i + g * g_i - 1))
                    dGamma = 0.125 * (1 + b * b_i) * (1 + a * a_i) * (g_i * (a * a_i + b * b_i + 2 * g * g_i - 1))
                    DFIABG[counter, 0, i] = dAlpha
                    DFIABG[counter, 1, i] = dBeta
                    DFIABG[counter, 2, i] = dGamma
                    gaussPoints[counter, 0] = a
                    gaussPoints[counter, 1] = b
                    gaussPoints[counter, 2] = g
                for i in range(8, 20):
                    a_i = AiBiGi[i, 0]
                    b_i = AiBiGi[i, 1]
                    g_i = AiBiGi[i, 2]
                    dAlpha = 0.25 * (1 + b * b_i) * (1 + g * g_i) * (
                            -a_i ** 3 * b_i ** 2 * g ** 2 - a_i ** 3 * b ** 2 * g_i ** 2 - 3 * a ** 2 * a_i * b_i ** 2 * g_i ** 2 + a_i - 2 * a * b_i ** 2 * g_i ** 2)
                    dBeta = 0.25 * (1 + a * a_i) * (1 + g * g_i) * (
                            -a_i ** 2 * b_i ** 3 * g ** 2 - a ** 2 * b_i ** 3 * g_i ** 2 - 3 * a_i ** 2 * b ** 2 * b_i * g_i ** 2 + b_i - 2 * a_i ** 2 * b * g_i ** 2)
                    dGamma = 0.25 * (1 + b * b_i) * (1 + a * a_i) * (
                            -a_i ** 2 * b ** 2 * g_i ** 3 - a ** 2 * b_i ** 2 * g_i ** 3 - 3 * a_i ** 2 * b_i ** 2 * g ** 2 * g_i + g_i - 2 * a_i ** 2 * b_i ** 2 * g)
                    DFIABG[counter, 0, i] = dAlpha
                    DFIABG[counter, 1, i] = dBeta
                    DFIABG[counter, 2, i] = dGamma
                counter += 1

    return DFIABG, gaussPoints





def calc_delta(cueb_index, determinant_index, divided_cueb_coordinates,DFIABG):
    delta = np.zeros((3, 3))
    count = 0
    for i in range(0, 3):
        for r in range(0, 3):
            sum = 0
            for j in range(0, 20):
                count += 1
                sum += divided_cueb_coordinates[cueb_index][1][j][i] * DFIABG[determinant_index][r][j]
            delta[r][i] = sum
    return delta


def gaussian_elimination(matrix, vector):
    augmented_matrix = np.column_stack((matrix, vector))
    rows, cols = augmented_matrix.shape

    for i in range(rows):
        # Partial pivoting
        max_row = i
        for j in range(i + 1, rows):
            if abs(augmented_matrix[j, i]) > abs(augmented_matrix[max_row, i]):
                max_row = j
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        # Forward elimination
        for j in range(i + 1, rows):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, :] -= factor * augmented_matrix[i, :]

    # Backward substitution
    solution = np.zeros(rows)
    for i in range(rows - 1, -1, -1):
        solution[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, :-1], solution)) / augmented_matrix[i, i]

    return solution


def calc_dfixyz(cube,cube_index, DFIABG):
    DFIXYZ = np.zeros((27, 20, 3))
    for i in range(0, 27):
        for j in range(0, 20):
            A = cube[cube_index][1]
            b = np.array([DFIABG[i][0][j], DFIABG[i][1][j], DFIABG[i][2][j]])
            gaussian_res = gaussian_elimination(A, b)
            DFIXYZ[i][j][0] = gaussian_res[0]
            DFIXYZ[i][j][1] = gaussian_res[1]
            DFIXYZ[i][j][2] = gaussian_res[2]
    return DFIXYZ




def calc_MGE_a11(DFIXYZ, cube, c, L, mu, nu):
    MGE_a11 = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            indx = 0
            sum = 0
            for m in range(len(c)):
                inner_sum_1 = 0
                for n in range(len(c)):
                    inner_sum_2 = 0
                    for k in range(len(c)):
                        dxx = DFIXYZ[indx][i][0] * DFIXYZ[indx][j][0]
                        dyy = DFIXYZ[indx][i][1] * DFIXYZ[indx][j][1]
                        dzz = DFIXYZ[indx][i][2] * DFIXYZ[indx][j][2]

                        inner_sum_2 += c[k] * (
                                ((L * (1 - nu)) * dxx + mu * (dyy + dzz)) * (np.linalg.det(cube[indx])))
                        indx += 1

                    inner_sum_1 += c[n] * inner_sum_2
                sum += c[m] * inner_sum_1
            MGE_a11[i][j] = sum
    return MGE_a11


def calc_MGE_a12(DFIXYZ, cube, c, L, mu, nu):
    MGE_a12 = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            indx = 0
            sum = 0
            for m in range(len(c)):
                inner_sum_1 = 0
                for n in range(len(c)):
                    inner_sum_2 = 0
                    for k in range(len(c)):
                        dxy = DFIXYZ[indx][i][0] * DFIXYZ[indx][j][1]
                        dyx = DFIXYZ[indx][i][1] * DFIXYZ[indx][j][0]
                        inner_sum_2 += c[k] * ((L * nu * dxy + mu * dyx) * (np.linalg.det(cube[indx])))
                        indx += 1
                    inner_sum_1 += c[n] * inner_sum_2
                sum += c[m] * inner_sum_1
            MGE_a12[i][j] = sum
    return MGE_a12


def calc_MGE_a13(DFIXYZ, cube, c, L, mu, nu):
    MGE_a13 = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            indx = 0
            sum = 0
            for m in range(len(c)):
                inner_sum_1 = 0
                for n in range(len(c)):
                    inner_sum_2 = 0
                    for k in range(len(c)):
                        dxz = DFIXYZ[indx][i][0] * DFIXYZ[indx][j][2]
                        dzx = DFIXYZ[indx][i][2] * DFIXYZ[indx][j][0]
                        inner_sum_2 += c[k] * ((L * nu * dxz + mu * dzx) * (np.linalg.det(cube[indx])))
                        indx += 1
                    inner_sum_1 += c[n] * inner_sum_2
                sum += c[m] * inner_sum_1
            MGE_a13[i][j] = sum
    return MGE_a13


def calc_MGE_a22(DFIXYZ, cube, c, L, mu, nu):
    MGE_a22 = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            indx = 0
            sum = 0
            for m in range(len(c)):
                inner_sum_1 = 0
                for n in range(len(c)):
                    inner_sum_2 = 0
                    for k in range(len(c)):
                        dyy = DFIXYZ[indx][i][1] * DFIXYZ[indx][j][1]
                        dxx = DFIXYZ[indx][i][0] * DFIXYZ[indx][j][0]
                        dzz = DFIXYZ[indx][i][2] * DFIXYZ[indx][j][2]
                        inner_sum_2 += c[k] * (((L * (1 - nu)) * dyy + mu * (dxx + dzz)) * (np.linalg.det(cube[indx])))
                        indx += 1
                    inner_sum_1 += c[n] * inner_sum_2
                sum += c[m] * inner_sum_1
            MGE_a22[i][j] = sum
    return MGE_a22


def calc_MGE_a33(DFIXYZ, cube, c, L, mu, nu):
    MGE_a33 = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            indx = 0
            sum = 0
            for m in range(len(c)):
                inner_sum_1 = 0
                for n in range(len(c)):
                    inner_sum_2 = 0
                    for k in range(len(c)):
                        dyy = DFIXYZ[indx][i][1] * DFIXYZ[indx][j][1]
                        dxx = DFIXYZ[indx][i][0] * DFIXYZ[indx][j][0]
                        dzz = DFIXYZ[indx][i][2] * DFIXYZ[indx][j][2]
                        inner_sum_2 += c[k] * (((L * (1 - nu)) * dzz + mu * (dxx + dyy)) * (np.linalg.det(cube[indx])))
                        indx += 1
                    inner_sum_1 += c[n] * inner_sum_2
                sum += c[m] * inner_sum_1
            MGE_a33[i][j] = sum
    return MGE_a33


def calc_MGE_a23(DFIXYZ, cube, c, L, mu, nu):
    MGE_a23 = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            indx = 0
            sum = 0
            for m in range(len(c)):
                inner_sum_1 = 0
                for n in range(len(c)):
                    inner_sum_2 = 0
                    for k in range(len(c)):
                        dyz = DFIXYZ[indx][i][1] * DFIXYZ[indx][j][2]
                        dzy = DFIXYZ[indx][i][2] * DFIXYZ[indx][j][1]
                        inner_sum_2 += c[k] * ((L * nu * dyz + mu * dzy) * (np.linalg.det(cube[indx])))
                        indx += 1
                    inner_sum_1 += c[n] * inner_sum_2
                sum += c[m] * inner_sum_1
            MGE_a23[i][j] = sum
    return MGE_a23



def calculate_dspite():
    DSPITE = np.zeros((9, 2, 8))
    MiTi = np.array([[-1, 1, 1, -1, 0, 1, 0, -1],
                     [-1, -1, 1, 1, -1, 0, 1, 0]])

    psi_i = np.zeros((9,8))

    AiBiGi = np.array([list(round(val) for val in row) for row in zip(*MiTi)])
    ABG = {-np.sqrt(0.6), 0, np.sqrt(0.6)}
    counter = 0
    gaussPoints = np.zeros((9, 2))
    iterNum = 0
    for tau in ABG:
        for eta in ABG:
            for i in range(0, 8):
                eta_i = AiBiGi[i, 0]
                tau_i = AiBiGi[i, 1]



                if (i < 4):


                    dEta = 0.25 * (1 + tau * tau_i) * eta_i * (2 * eta * eta_i + tau * tau_i)
                    dTau = 0.25 * (1 + eta * eta_i) * tau_i * (2 * tau * tau_i + eta * eta_i)

                    DSPITE[counter, 0, i] = dEta
                    DSPITE[counter, 1, i] = dTau

                    gaussPoints[counter, 0] = eta
                    gaussPoints[counter, 1] = tau

                    psi_i[iterNum][i] = 0.25 * (1 + eta * eta_i) * (1 + tau * tau_i) * (eta * eta_i + tau * tau_i - 1)


                if (i == 4 or i == 6):

                    dEta = -eta * (1 + tau * tau_i)
                    dTau = 0.5 * (1 - eta ** 2) * tau_i

                    DSPITE[counter, 0, i] = dEta
                    DSPITE[counter, 1, i] = dTau

                    psi_i[iterNum][i] = 0.5 * (1 - eta ** 2) * (1 + tau * tau_i)

                if (i == 5 or i == 7):

                    dEta = 0.5 * eta_i * (1 - tau ** 2)
                    dTau = -tau * (1 + eta * eta_i)

                    DSPITE[counter, 0, i] = dEta
                    DSPITE[counter, 1, i] = dTau

                    psi_i[iterNum][i] = 0.5 * (1 - tau ** 2) * (1 + eta * eta_i)
            iterNum += 1
            counter += 1
    print("PSI_I: ", psi_i)
    return DSPITE, gaussPoints, psi_i



def getSlides(divided_cube_coordinates):
    slides = []
    for fig in divided_cube_coordinates:
        slides.append(fig[2])
    return slides


def calc_DFITE(DSPITE,z_i):
    dfite = []
    indx = 0
    for i in range(3):
        for j in range(3):
            sum_eta_x = 0
            sum_eta_y = 0
            sum_eta_z = 0
            sum_tau_x = 0
            sum_tau_y = 0
            sum_tau_z = 0
            for k in range(0, 8):
                sum_eta_x += z_i[k][0] * DSPITE[indx][0][k]
                sum_tau_x += z_i[k][0] * DSPITE[indx][1][k]

                sum_eta_y += z_i[k][1] * DSPITE[indx][0][k]
                sum_tau_y += z_i[k][1] * DSPITE[indx][1][k]

                sum_eta_z += z_i[k][2] * DSPITE[indx][0][k]
                sum_tau_z += z_i[k][2] * DSPITE[indx][1][k]
            indx += 1
            dfite.append([[sum_eta_x, sum_tau_x], [sum_eta_y, sum_tau_y], [sum_eta_z, sum_tau_z]])
    return dfite



def calc_FE1(DFITE, psi_i,c,p):
    FE_1 = []

    for k in range(0, 8):
        sum = 0
        iterNum = 0
        for m in range(3):
            sub_sum = 0
            for n in range(3):
                dy_dn = DFITE[k][1][0]
                dy_dt = DFITE[k][1][1]
                dz_dn = DFITE[k][2][0]
                dz_dt = DFITE[k][2][1]
                sub_sum += c[n] * ( dy_dn * dz_dt - dy_dt * dz_dn ) * p * psi_i[iterNum][k]
                iterNum += 1
            sum += c[m] * sub_sum
        FE_1.append(sum)
    print("FE_1: ", FE_1)
    return FE_1



def calc_FE2(DFITE, psi_i,c,p):
    FE_2 = np.zeros((8, 1))

    for k in range(0, 8):
        sum = 0
        iterNum = 0
        for m in range(3):
            sub_sum = 0
            for n in range(3):
                dz_dn = DFITE[k][2][0]
                dz_dt = DFITE[k][2][1]
                dx_dt = DFITE[k][0][1]
                dx_dn = DFITE[k][0][0]
                sub_sum += c[n] * (dz_dn * dx_dt - dx_dn * dz_dt) * p * psi_i[iterNum][k]
                iterNum += 1
            sum += c[m] * sub_sum
        FE_2[k] = sum
    print("FE_2: ", FE_2)
    return FE_2



def calc_FE3(DFITE, psi_i,c,p):
    FE_3 = np.zeros((8, 1))

    for i in range(0, 8):
        sum = 0
        iterNum = 0
        for m in range(3):
            sub_sum = 0
            for n in range(3):
                dx_dn = DFITE[i][0][0]
                dx_dt = DFITE[i][0][1]
                dy_dt = DFITE[i][1][1]
                dy_dn = DFITE[i][1][0]
                sub_sum += c[n] * (dx_dn * dy_dt - dy_dn * dx_dt) * p * psi_i[iterNum][i]
                iterNum += 1
            sum += c[m] * sub_sum

        FE_3[i] = sum
    print('FE_3',FE_3)
    return FE_3






