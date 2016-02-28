
from PIL import Image, ImageFilter
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line, resize, hough_circle)
from skimage.feature import peak_local_max, canny
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx


class Node:
    def __init__(self):
        self.x = -1
        self.y = -1
        self.intensity = -1
        
    def __init__(self, x, y, intensity):
        self.x = x
        self.y = y
        self.intensity = intensity
        
    def __repr__(self):
        return repr((self.x, self.y, self.intensity))


#We assume that out image has a margin of this many pixels
def getMargin():
    return 10
    
def getMinNodeDist():
    return 10

def getCurvatureCheckDist():
    return 4

#Returns a list of coordinates of potential nodes, ordered from highest to
#lowest intensity.
def findNodesIntensity(imageData):
    nodes = []
    margin = getMargin()
    minIntensity = 20
    for i in xrange(margin, len(imageData) - margin):
        for j in xrange(margin, len(imageData[0]) - margin):
            if(imageData[i][j] > minIntensity and isLocalMax(imageData, i, j, margin)):
                nodes.append(Node(j, i, imageData[i][j]))
                
    nodes.sort(key = lambda node: -node.intensity)
    return nodes


#Returns true if the current point is the maximum over all points within
#[x - dist, x + dist] x [y - dist, y + dist]
def isLocalMax(imageData, x, y, dist):
    for i in range(-dist, dist + 1):
        for j in range(-dist, dist + 1):
            #print x, y, x + i, y + j
            if(imageData[x + i][y + j] > imageData[x][y]):
                return False
    return True


#Returns the minimum and maximum curvature over all possible directions at the
#given location
def getCurvatureRange(imageData, x, y):
    assert(isLocalMax(imageData, x, y, getMargin()))
    curvatureCheckDist = getCurvatureCheckDist()
    minCurvature = -1
    maxCurvature = -1
    for angle in [angleDiff*math.pi/180.0 for angleDiff in xrange(0, 180, 10)]:
        #Take a slice with the given distance at the given angle in both directions,
        #and get the list of heights
        xDiff = math.cos(angle)
        yDiff = math.sin(angle)
        xLocs = [x + i*xDiff for i in xrange(-curvatureCheckDist, curvatureCheckDist + 1)]
        yLocs = [y + i*yDiff for i in xrange(-curvatureCheckDist, curvatureCheckDist + 1)]
        heights = [interpolateVal(imageData, xLocs[i], yLocs[i]) for i in xrange(len(xLocs))]
        
        newCurvature = getCurvature(heights)
        #print newCurvature, heights
        if(angle == 0 or newCurvature < minCurvature):
            minCurvature = newCurvature
        if(angle == 0 or newCurvature > maxCurvature):
            maxCurvature = newCurvature
    
    return minCurvature, maxCurvature
        

#Computes the curvature of the best-fit circle for the arc with the given heights.
#We average the values we get for the curvature when we compare against each point.
def getCurvature(heights):
    avgCurvature = 0
    maxHeight = heights[len(heights)/2]
    for i in xrange(len(heights)):
        if(not i == len(heights)/2): #Check middle point against all others
            avgCurvature += getPointCurvature(i - len(heights)/2, maxHeight - heights[i])
            #print "Average curvature at ", i, ": ", avgCurvature
    return avgCurvature/(len(heights) - 1)            


#Get the radius of a circle whose height drops by yDiff when we move xDiff units
#to the right of the maximum
def getPointCurvature(xDiff, yDiff):
    #print "Point curvature for: ", xDiff, yDiff
    if(yDiff < 0):
        print "Error: negative value for yDiff"
    return (2.0*yDiff)/(xDiff*xDiff + yDiff*yDiff)

    
def findNodesHoughCircles(cannyEdges, hough_radii):
    hough_res = hough_circle(cannyEdges, hough_radii)
    
    centers = []
    accums = []
    radii = []

    #print hough_res
    for radius, h in zip(hough_radii, hough_res):
        num_peaks = 5
        peaks = peak_local_max(h, num_peaks=num_peaks)
        #print peaks
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    #Sort each list by accums
    tupleCenters = [tuple(center) for center in centers]
    accums, radii, tupleCenters = zip(*sorted(zip(accums, radii, tupleCenters))[::-1])
    
    disjointCenters = []
    disjointRadii = []
    disjointAccums = []
    
    removed = [False for i in xrange(len(centers))]
    #Loop through the circles in order of their weight
    minRadiusDist = getMinNodeDist()
    for i in xrange(len(centers)):
        #print "Next index: ", i
        if(not removed[i]):
            #If our current circle does not overlap with previous circles, add it.
            #Then remove all later overlapping circles
            center_x, center_y = centers[i]
            disjointCenters.append(centers[i])
            disjointRadii.append(radii[i])
            disjointAccums.append(accums[i])
            
            for j in xrange(i + 1, len(centers)):
                newCenter_x, newCenter_y = centers[j]
                newRadius = radii[i]
                distSq = findDistSqPoints(center_x, center_y, newCenter_x, newCenter_y)
                neededDist = radius + newRadius + minRadiusDist
                if(distSq < neededDist*neededDist):
                    removed[j] = True
    
    print "Disjoint centers: ", disjointCenters
    print "Disjoint accums: ", disjointAccums
    return disjointAccums, disjointCenters, disjointRadii


def findEdgePoints(imageData):
    edgePoints = []
    maxSmallestCurvature = 0.01
    minLargestCurvature = 0.05
    for y in xrange(margin, len(imageData) - margin):
        for x in xrange(margin, len(imageData) - margin):
            minCurvature, maxCurvature = getCurvatureRange(imageData, x, y)
            if(isLocalMax(imageData, x, y) and minCurvature < maxSmallestCurvature and maxCurvature > minLargestCurvature):
                edgePoints.append(Node(x, y, imageData[i][j]))
                
    print edgePoints
    return edgePoints

    
#For real-valued x and y, we interpolate the value of imageData[x][y]
#We average the four surrounding values to get the point in the middle
#of them, and then use the middle vertex to divide our square into four
#triangles. Then we draw a plane through the three points in each triangle.
def interpolateVal(imageData, x, y):
    print x, y
    xLower = int(x)
    yLower = int(y)
    xDiff = x - xLower
    yDiff = y - yLower
    if(xDiff == 0 and yDiff == 0):
        return imageData[xLower, yLower]
    elif(xLower == x):
        #Draw line between imageData[x][yLower] and imageData[x][yLower + 1]
        #Weight by 1 - distance
        return yDiff*imageData[xLower][yLower + 1] + (1 - yDiff)*imageData[xLower][yLower]
    elif(yLower == y):
        #Draw line between imageData[xLower][y] and imageData[xLower + 1][y]
        #Weight by 1 - distance
        xDiff = x - xLower
        return xDiff*imageData[xLower + 1][yLower] + (1 - xDiff)*imageData[xLower][yLower]
    else:
        #TODO: Draw plane between the three points in our triangle
        pass
        

#Just round to the closest pixel value
def interpolateValLazy(imageData, x, y):
    x = int(x + 0.5)
    y = int(y + 0.5)
    return imageData[x][y]
        
    
     
#Return squared distance between two points
def invDistanceSq(x1, y1, x2, y2):
    return 1/((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))


    

#Takes the line from (0, y0) to (width, y1), and scales it to fit
#inside our range
def scaleLines(y0, y1, width, height):
    x0 = 0
    x1 = width
    slope = (y1 - y0)/(x1 - x0)
    intercept = y0
    
    #x = (y - intercept)/slope
    if(slope > 0):
        if(y0 < 0):
            #Find x such that (x, 0) is on this line        
            x0 = (0 - intercept)/slope
            y0 = 0
        
        if(y1 > height):
            #Find x xuch that (x, height) is on this line
            x1 = (height - intercept)/slope
            y1 = height
    else:
        if(y0 > height):
            #Find x such that (x, height) is on this line
            x0 = (height - intercept)/slope
            y0 = height
            
        if(y1 < 0):
            #Find x such that (x, 0) is on this line)
            x1 = (0 - intercept)/slope
            y1 = 0
    return x0, y0, x1, y1
    

def readImage(image):
    blurredImage = image.filter(ImageFilter.GaussianBlur(radius = 2))
    imageData = toMatrix(blurredImage, blurredImage.size[1], blurredImage.size[0])
    cannyEdges = canny(imageData, sigma=3)
    #print "Canny edges: ", cannyEdges
    
    #Look for the most likely Canny edges to be edges between nodes in our diagram
    edges = probabilistic_hough_line(cannyEdges, threshold = 10, line_length = 100, line_gap = 3)
    
    #Look for local maxima in intensity
    nodes1 = findNodesIntensity(imageData)
    #Look for circles on the boundary of each node
    hough_radii = np.arange(3, 15, 2)
    weights, centers, radii = findNodesHoughCircles(cannyEdges, hough_radii)
    
    #Choose the closest point in nodes1 for each circle in nodes2
    finalNodes = []
    for center in centers:
        minDistSq = 1000000
        minNode = nodes1[0]
        for node in nodes1:
            newDistSq = findDistSqPoints(center[0], center[1], node.x, node.y)
            if(newDistSq < minDistSq):
                minDistSq = newDistSq
                minNode = node
        print center, node, minDistSq
        finalNodes.append(minNode)
    
    #Find all of the nodes within maxDist of each edge
    print "Finding nodes for each edge"
    graph = nx.Graph()
    maxDist = 20
    for edge in edges:
        print "Edge: ", edge
        edgeNodes = []
        for node in finalNodes:
            dist = findDistPointToLine(node, edge)
            print "Node: ", node
            print "Distance: ", dist
            if(findDistPointToLine(node, edge) < maxDist):
                edgeNodes.append(node)
        
        if(len(edgeNodes) == 2):
            if(edgeNodes[0] not in graph.nodes()):
                graph.add_node(edgeNodes[0])
            if(edgeNodes[1] not in graph.nodes()):
                graph.add_node(edgeNodes[1])
            if((edgeNodes[0], edgeNodes[1]) not in graph.edges()):
                graph.add_edge(edgeNodes[0], edgeNodes[1])
        else:
            edgeNodes.sort(key = lambda node: (node.x, node.y))
    
    printTest(imageData)
    return graph
    
def findDistSqPoints(x0, y0, x1, y1):
    return (x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0)


#Finds the distance from a node to an edge    
#The edge is given as a pair of points, and each point as a pair of coordinates
def findDistPointToLine(node, edge):
    x0, y0 = edge[0]
    x1, y1 = edge[1]
    if(x0 == x1):
        return abs(x0 - node.x)
    elif(y0 == y1):
        return abs(y0 - node.y)
    else:
        slope = float(y1 - y0)/(x1 - x0)
        intercept = y0 - slope*x0
        #print "Slope/intercept: ", slope, intercept
        
        #First we find the perpendicular line from our node
        #to our given line
        newSlope = -1/slope
        newIntercept = node.y + node.x/slope
        #print "New slope/intercept: ", newSlope, newIntercept
        
        #Now we find the intersection point of these two lines,
        #and compute the distance
        xIntersect, yIntersect = findLineIntersect(slope, intercept, newSlope, newIntercept)
        #print xIntersect, yIntersect
        distSq = findDistSqPoints(node.x, node.y, xIntersect, yIntersect)
        return math.sqrt(distSq)
        

#Returns the intersection point of a pair of lines, given as a slope/intercept
#The lines must not have the same slope
def findLineIntersect(slope1, intercept1, slope2, intercept2):
    assert(not slope1 == slope2)
    xIntersect = (intercept2 - intercept1)/(slope1 - slope2)
    yIntersect = slope1*xIntersect + intercept1
    return xIntersect, yIntersect
    
    
def printTest(imageData):
    print "Entering printTest..."
    h, theta, d = hough_line(imageData)
    fig, ax = plt.subplots(1, 3, figsize = (10, 4))
    
    ax[0].imshow(imageData, cmap=plt.cm.gray)
    ax[0].set_title("Input image")
    ax[0].axis("image")
    
    ax[1].imshow(np.log(1 + h), extent = [np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
                    d[-1], d[0]],
                    cmap = plt.cm.gray, aspect = 1/1.5)
    ax[1].set_title("Hough transform")
    ax[1].set_xlabel("Angles (degrees)")
    ax[1].set_ylabel("Distance (pixels)")
    ax[1].axis("image")
    
    ax[2].imshow(imageData, cmap = plt.cm.gray)
    height, width = imageData.shape
        
    ax[2].axis((0, width, height, 0))
    ax[2].set_title("Detected lines")
    ax[2].axis("image")
    
    edges = canny(imageData, sigma=3)
    lines = probabilistic_hough_line(edges, threshold = 5, line_length = 5, line_gap = 3)
    
    fig2, ax = plt.subplots(1, 3, figsize = (8, 3))
    
    ax[0].imshow(image, cmap = plt.cm.gray)
    ax[0].set_title("Input image")
    ax[0].axis("image")
    
    ax[1].imshow(edges, cmap = plt.cm.gray)
    ax[1].set_title("Canny edges")
    ax[1].axis("image")
    
    ax[2].imshow(edges * 0)
    
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
        
    ax[2].set_title("Probabilistic Hough")
    ax[2].axis("image")
    plt.show()
    

#Converts image to a grayscale numpy array, where 0 = white and 255 = black
#Also pads the sides with black
def toMatrix(image, width, height):
    image = image.convert("L")
    imageData = list(image.getdata())
    for i in xrange(len(imageData)):
        imageData[i] = float(255 - imageData[i]) #Flip scale to 0 = white
        
    minData = min(imageData)
    for i in xrange(len(imageData)):
        imageData[i] -= minData
        
    margin = getMargin()
    dataMatrix = np.zeros((width + 2*margin, height + 2*margin))
    for i in xrange(width + 2*margin):
        for j in xrange(height + 2*margin):
            if(i < margin or i >= width + margin or j < margin or j >= height + margin):
                dataMatrix[i][j] = 0
            else:
                dataMatrix[i][j] = imageData[(i - margin) + (j - margin)*width]
            
    return dataMatrix


def runTests():
    image = Image.open("../Images/image1.png")
    blurredImage = image.filter(ImageFilter.GaussianBlur(radius = 2))
    imageData = toMatrix(blurredImage, blurredImage.size[1], blurredImage.size[0])
    #printTest(imageData)
    #nodes = findNodes(imageData)


def testInterpolateVal():
    size = 2
    testData = [[1,2,3],[5,7,9],[8,6,3]]
    for i in xrange(size*5):
        print [interpolateVal(testData, i/5.0, j/5.0) for j in xrange(size)]


def compareImages():
    image1 = Image.open("../Images/image5.png")
    image2 = Image.open("../Images/image5_bold.png")
    
    blurredImage1 = image1.filter(ImageFilter.GaussianBlur(radius = 2))
    imageData1 = toMatrix(blurredImage1, blurredImage1.size[1], blurredImage1.size[0])
    blurredImage2 = image2.filter(ImageFilter.GaussianBlur(radius = 2))
    imageData2 = toMatrix(blurredImage2, blurredImage2.size[1], blurredImage2.size[0])
    
    h1, theta1, d1 = hough_line(imageData1)
    h2, theta2, d2 = hough_line(imageData2)

if __name__ == '__main__':
    #testInterpolateVal()
    #runTests()
    image = Image.open("../Images/image1.png")
    graph = readImage(image)
    print graph.nodes()
    print graph.edges()