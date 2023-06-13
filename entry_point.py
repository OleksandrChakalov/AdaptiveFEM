from main import *



# Example usage:
ax = 2
ay = 1
az = 2

nx = 2
ny = 1
nz = 2

#CONSTANTS
nu = 0.3
E = 1.0
p = 1
L = E / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
c = [5 / 9, 8 / 9, 5 / 9]


#Create parallelepiped and subdivisions
parallelepiped = Parallelepiped(ax, ay, az)
subdivisions = Subdivisions(nx, ny, nz)

#Get coordinates of AKT and figures (list of [x,y,z])
AKT = getNodeCoordinates(parallelepiped, subdivisions)


with open("result.txt", "w") as file:
    file.write(f"ax = {ax} \n")
    file.write(f"ay = {ay} \n")
    file.write(f"az = {az} \n")
    file.write(f"nx = {nx} \n")
    file.write(f"ny = {ny} \n")
    file.write(f"nz = {nz} \n")
    file.write(f"nu = {nu} \n")
    file.write(f"E = {E} \n")
    file.write(f"p = {p} \n")
    file.write(f"L = {L} \n")
    file.write(f"mu = {mu} \n")
    file.write(f"c = {c} \n")
    file.write("--------------------------------------------------------------------------------------------------\n")

#Write AKT to file
with open("result.txt", "a") as file:
    file.write(f"AKT {len(AKT)}" + "\n")
    for item in AKT:
        file.write(f"{item}\n")
    file.write("--------------------------------------------------------------------------------------------------\n")

#Get coordinates of each divided cube in the parallelepiped ( list of (index, [x,y,z]) )
divided_cubes_coordinates = parallelepipedCorners(parallelepiped, subdivisions)

#Get NT (list of (index, [global numeration of points]), )
NT = getIndices(divided_cubes_coordinates, AKT)

#Write NT to file
with open("result.txt", "a") as file:
    file.write(f"NT {len(NT)} \n")
    for item in NT:
        file.write(f"{item} \n")
        file.write("--------------------------------------------------------------------------------------------------\n")





# Plot CUBE
trace = go.Scatter3d(
    x=[coord[1][0] for coord in AKT],
    y=[coord[1][1] for coord in AKT],
    z=[coord[1][2] for coord in AKT],
    text=[coord[0] for coord in AKT],
    mode='markers',
    marker=dict(
        size=3,
        color='blue',
        opacity=0.8
    )
)

# Create layout
layout = go.Layout(
    title='Parallelepiped Node Coordinates',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
    ),
)

# Create figure
fig = go.Figure(data=[trace], layout=layout)

# Show figure
fig.show()


NG = createNG(NT)
#Write NG to file
with open("result.txt", "a") as file:
    file.write(f"NG {NG} \n")
    file.write("--------------------------------------------------------------------------------------------------\n")


ZP = calculateZp(nx, ny, nz)

#Write ZP to file
with open("result.txt", "a") as file:
    file.write(f"ZP {len(ZP)} \n")
    for item in ZP:
        file.write(f"{item} \n")
    file.write("--------------------------------------------------------------------------------------------------\n")


ZU = calculateZu(nx,ny)

#Write ZP to file
with open("result.txt", "a") as file:
    file.write(f"ZU {len(ZU)} \n")
    for item in ZU:
        file.write(f"{item} \n")
    file.write("--------------------------------------------------------------------------------------------------\n")


DFIABG, gaussPoints = calculate_dfiabg()

#Write DFIABG to file
with open("result.txt", "a") as file:
    file.write(f"DFIABG {len(DFIABG)} \n")
    for item in DFIABG:
        file.write(f"{item} \n")
    file.write("--------------------------------------------------------------------------------------------------\n")

#Write gaussPoints to file
with open("result.txt", "a") as file:
    file.write(f"gaussPoints {len(gaussPoints)} \n")
    for item in gaussPoints:
        file.write(f"{item} \n")
    file.write("--------------------------------------------------------------------------------------------------\n")



cube = np.zeros((len(NT), 27,3,3))


for i in range(0, len(NT)):
    for j in range(0, 27):
        cube[i][j] = calc_delta(i, j, divided_cubes_coordinates, DFIABG,)




# #Write DJ to file
# with open("result.txt", "a") as file:
#     for i in range(len(NT)):
#         file.write(f"cube {i+1} \n")
#         file.write(f"DJ \n")
#         for j in range(27):
#             file.write(f"Gauss Point{j+1}\t({gaussPoints[j][0]:>8.4f}; {gaussPoints[j][1]:>8.4f}; {gaussPoints[j][2]:>8.4f}): \n")
#             file.write(f"{cube[i][j]} \n")
#         file.write("--------------------------------------------------------------------------------------------------\n")


for i in range(0, len(NT)):
    with open(f"cube{i+1}.txt", "w") as file:
        file.write("DJ \n")
        for j in range(27):
            file.write(f"Gauss Point{j+1}\t({gaussPoints[j][0]:>8.4f}; {gaussPoints[j][1]:>8.4f}; {gaussPoints[j][2]:>8.4f}): \n")
            file.write(f"{cube[i][j]} \n")
            file.write(f"DET  {np.linalg.det(cube[i][j])} \n")
            file.write("--------------------------------------------------------------------------------------------------\n")
        file.write("\n")



for i in range(0, len(NT)):
    with open(f"cube{i+1}.txt", "a") as file:
        file.write("\n")
        file.write("DFIXYZ \n")
        DFIXYZ = calc_dfixyz(cube, i, DFIABG)
        for j in range(27):
            file.write(f"Gauss Point #{j}\t({gaussPoints[j][0]:>8.4f}; {gaussPoints[j][1]:>8.4f}; {gaussPoints[j][2]:>8.4f}): \n")
            file.write(f"{DFIXYZ[j]} \n")

MGE_ALL = []

for i in range(0, len(NT)):
    DFIXYZ = calc_dfixyz(cube, i, DFIABG)
    MGE_a11 = calc_MGE_a11(DFIXYZ, cube[i], c, L, mu, nu)
    MGE_a12 = calc_MGE_a12(DFIXYZ, cube[i], c, L, mu, nu)
    MGE_a13 = calc_MGE_a13(DFIXYZ, cube[i], c, L, mu, nu)
    MGE_a22 = calc_MGE_a22(DFIXYZ, cube[i], c, L, mu, nu)
    MGE_a23 = calc_MGE_a23(DFIXYZ, cube[i], c, L, mu, nu)
    MGE_a33 = calc_MGE_a33(DFIXYZ, cube[i], c, L, mu, nu)

    MGE = np.zeros((60, 60))

    MGE[:20, :20] = MGE_a11
    MGE[:20, 20:40] = MGE_a12
    MGE[:20, 40:60] = MGE_a13
    MGE[20:40, :20] = MGE_a12
    MGE[20:40, 20:40] = MGE_a22
    MGE[20:40, 40:60] = MGE_a23
    MGE[40:60, :20] = MGE_a13
    MGE[40:60, 20:40] = MGE_a23
    MGE[40:60, 40:60] = MGE_a33

    MGE_ALL.append(MGE)

    with open(f"cube{i + 1}.txt", "a") as file:
        file.write("\n")
        file.write("MGE \n")
        for j in range(60):
            file.write(f"{MGE[j]} \n")
        file.write("\n")
        file.write("--------------------------------------------------------------------------------------------------\n")


DSPITE, gauss_points, psi_i = calculate_dspite()

#Write DSPITE to file
with open("result.txt", "a") as file:
    file.write(f"DSPITE {len(DSPITE)} \n")
    for i in range(len(DSPITE)):
        file.write(f"Gauss Point #{i+1}\t({gauss_points[i][0]:>8.4f}; {gauss_points[i][1]:>8.4f};)\n")
        file.write(f"{DSPITE[i]} \n")
    file.write("--------------------------------------------------------------------------------------------------\n")



slides = getSlides(divided_cubes_coordinates)


for i in range(len(NT)):
    for j in range(len(ZP)):
        if(i == (ZP[j][0]-1)):
            with open(f"cube{i + 1}.txt", "a") as file:
                file.write("\n")
                file.write("DFITE \n")
                sl = slides[ZP[j][0] - 1][ZP[j][1]-1]
                DFITE = calc_DFITE(DSPITE, slides[ZP[j][0] - 1][ZP[j][1]-1])
                for i in range(0, 9):
                    file.write(f"Gauss Point #{i}\t({gauss_points[i][0]:>8.4f}; {gauss_points[i][1]:>8.4f};)\n")
                    file.write(f"{DFITE[i]} \n")



for i in range(len(NT)):
    for j in range(len(ZP)):
        if(i == (ZP[j][0]-1)):
            DFITE = calc_DFITE(DSPITE, slides[ZP[j][0] - 1][ZP[j][1]-1])
            FE_1 = calc_FE1(DFITE, psi_i, c, p)
            FE_2 = calc_FE2(DFITE, psi_i, c, p)
            FE_3 = calc_FE3(DFITE, psi_i, c, p)
            with open(f"cube{i + 1}.txt", "a") as file:
                file.write("\n")
                file.write("FE \n")
                file.write(f"{FE_1} \n")
                file.write(f"{FE_2} \n")
                file.write(f"{FE_3} \n")
                file.write("--------------------------------------------------------------------------------------------------\n")


connection = [4,5,6,7,16,17,18,19]

def calc_FE60():
    fe60_all = []
    for i in range(len(ZP)):
        fe60 = np.zeros((3, 20))
        sl = slides[ZP[i][0] - 1][ZP[i][1] - 1]
        for slides_side in range(len(sl)):
            nlocal = connection[slides_side]
            fe60[0][nlocal] = FE_1[slides_side]
            fe60[1][nlocal] = FE_2[slides_side]
            fe60[2][nlocal] = FE_3[slides_side]

        fe60_r = np.concatenate((fe60[0], fe60[1], fe60[2]))
        fe60_all.append(fe60)

    F = np.zeros((len(AKT)*3, 1))
    for l in range (len(fe60_all)):
        for i in range (len(fe60_all[l])):
            for j in range (len(fe60_all[l][i])):
                #F[i] += fe60_all[l][i][j]
                en = ZP[l][0] - 1
                g = NT[en][1][j] - 1
                globalIndex = 3 * (g) + i

                F[globalIndex] += fe60_all[l][i][j]

    return F


def getMG():
    MG = np.zeros((len(AKT)*3, len(AKT)*3))
    for mI in range(len(MGE_ALL)):
        for i in range(60):
            for j in range(60):
                gi = NT[mI][1][i % 20] - 1
                gj = NT[mI][1][j % 20] - 1

                globalI = 3 * gi + int(i / 20)
                globalJ = 3 * gj + int(j / 20)

                MG[globalI][globalJ] += MGE_ALL[mI][i][j]
    return MG


fe60 = calc_FE60()
MG = getMG()

global_conect = [0,1,2,3,4,5,6,7,8,9,10,11,12]

for i in range(len(global_conect)):
    gx = 3 * global_conect[i] + 0
    gy = 3 * global_conect[i] + 1
    gz = 3 * global_conect[i] + 2

    MG[gx][gx] += 10e20
    MG[gy][gy] += 10e20
    MG[gz][gz] += 10e20


U = gaussian_elimination(MG, fe60)

print(U)

modified = []
for i in range(len(AKT)):
    x = AKT[i][1][0] + U[3*i+0]
    y = AKT[i][1][1] + U[3*i+1]
    z = AKT[i][1][2] + U[3*i+2]

    modified.append([x,y,z])


# Plot CUBE
trace = go.Scatter3d(
    x=[coord[0] for coord in modified],
    y=[coord[1] for coord in modified],
    z=[coord[2] for coord in modified],
    text=[coord[0] for coord in modified],
    mode='markers',
    marker=dict(
        size=3,
        color='blue',
        opacity=0.8
    )
)

# Create layout
layout = go.Layout(
    title='Parallelepiped Node Coordinates',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
    ),
)

# Create figure
fig = go.Figure(data=[trace], layout=layout)

# Show figure
fig.show()




