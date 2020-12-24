
def xmaslight():
    # This is the code from my 
    
    #NOTE THE LEDS ARE GRB COLOUR (NOT RGB)
    
    # Here are the libraries I am currently using:
    import time
    import board
    #import neopixel
    import re
    import math
    
    # You are welcome to add any of these:
    # import random
    # import numpy
    # import scipy
    # import sys
    
    # If you want to have user changable values, they need to be entered from the command line
    # so import sys sys and use sys.argv[0] etc
    # some_value = int(sys.argv[0])
    
    # IMPORT THE COORDINATES (please don't break this bit)
    
    coordfilename = "Python/coords.txt"
	
    fin = open(coordfilename,'r')
    coords_raw = fin.readlines()
    
    coords_bits = [i.split(",") for i in coords_raw]
    
    coords = []
    
    for slab in coords_bits:
        new_coord = []
        for i in slab:
            new_coord.append(int(re.sub(r'[^-\d]','', i)))
        coords.append(new_coord)
    
    #set up the pixels (AKA 'LEDs')
    #PIXEL_COUNT = len(coords) # this should be 500

    
    #pixels = neopixel.NeoPixel(board.D18, PIXEL_COUNT, auto_write=False)
    
    
    # YOU CAN EDIT FROM HERE DOWN

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    # The change of the angle per update in DEGREES
    # Rotations apply in the order X -> Y -> Z because I'm bad at quaternions
    xAngleChange = 5
    yAngleChange = 10
    zAngleChange = 15

    fig = plt.figure()
    for i in range(500):
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_xlim([-400, 400])
        ax.set_ylim([-400, 400])
        ax.set_zlim([-400, 400])


        scale = 400
        xOffset = 0
        yOffset = 0
        zOffset = .4
        
        tetrahedron = [
            [(-1 + xOffset) * scale, (0 + yOffset) * scale, (-1/1.414 + zOffset) * scale],
            [(1 + xOffset) * scale, (0 + yOffset) * scale, (-1/1.414 + zOffset) * scale],
            [(0 + xOffset) * scale, (-1 + yOffset) * scale, (1/1.414 + zOffset) * scale],
            [(0 + xOffset) * scale, (1 + yOffset) * scale, (1/1.414 + zOffset) * scale]
        ]
        # vertexes of a tetrahedron
        tetrahedron = [
            matrixRotate(tetrahedron[0], xAngleChange * i, yAngleChange * i , zAngleChange * i),
            matrixRotate(tetrahedron[1], xAngleChange * i, yAngleChange * i , zAngleChange * i),
            matrixRotate(tetrahedron[2],  xAngleChange * i, yAngleChange * i , zAngleChange * i),
            matrixRotate(tetrahedron[3],  xAngleChange * i , yAngleChange * i , zAngleChange * i)
        ]

        inside = []
        outside = []
        
        for coord in coords:
            if (pointInsideTetrahedron(tetrahedron[0], tetrahedron[1], tetrahedron[2], tetrahedron[3], coord)):
                inside.append(coord)
            else:
                outside.append(coord)

        if (len(inside) > 0):
            xs, ys, zs = zip(*inside)
            ax.scatter(xs, ys, zs, color='orange')

        if (len(outside) > 0):
            xs, ys, zs = zip(*outside)
            ax.scatter(xs, ys, zs, color='purple')

        xs = []
        ys = []
        zs = []

        mesh = [
            tetrahedron[0],
            tetrahedron[1],
            tetrahedron[2],
            tetrahedron[0],
            tetrahedron[3],
            tetrahedron[1],
            tetrahedron[3],
            tetrahedron[2]
        ]

        for coord in mesh:
            xs.append(coord[0])
            ys.append(coord[1])
            zs.append(coord[2])
        ax.scatter(xs, ys, zs)
        ax.plot(xs, ys, zs)

        plt.savefig(str(i) + '.png')
    
    #plt.show()


def sameSide(v1, v2, v3, v4, point):
    import numpy as np

    normal = np.cross(np.subtract(v2, v1), np.subtract(v3, v1))
    dotv4 = np.dot(normal, np.subtract(v4, v1))
    dotpoint = np.dot(normal, np.subtract(point, v1))

    return (np.sign(dotv4) == np.sign(dotpoint))

def pointInsideTetrahedron(v1, v2, v3, v4, point):
    return (sameSide(v1, v2, v3, v4, point) and
            sameSide(v2, v3, v4, v1, point) and
            sameSide(v3, v4, v1, v2, point) and
            sameSide(v4, v1, v2, v3, point))

def matrixRotate(point, xAngle, yAngle, zAngle):
    import numpy as np
    import math

    v1 = np.array([[point[0]], [point[1]], [point[2]]])

    xAngle = math.radians(xAngle)
    yAngle = math.radians(yAngle)
    zAngle = math.radians(zAngle)

    xRotation = np.array([
        [1, 0, 0],
        [0, np.cos(xAngle), -np.sin(xAngle)],
        [0, np.sin(xAngle), np.cos(xAngle)]
    ])

    yRotation = np.array([
        [np.cos(yAngle), 0, np.sin(yAngle)],
        [0, 1, 0],
        [-np.sin(yAngle), 0, np.cos(yAngle)]
    ])

    zRotation = np.array([
        [np.cos(zAngle), -np.sin(zAngle), 0],
        [np.sin(zAngle), np.cos(zAngle), 0],
        [0, 0, 1]
    ])

    v2 = zRotation.dot(yRotation.dot(xRotation.dot(v1)))
    return v2[0][0], v2[1][0], v2[2][0]


    


# yes, I just put this at the bottom so it auto runs
xmaslight()