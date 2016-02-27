
from PIL import Image, ImageFilter
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage.feature import canny
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
    return 4

def getCurvatureCheckDist():
    return 4

#Returns a list of coordinates of potential nodes, ordered from highest to
#lowest intensity.
def findNodes(imageData):
    nodes = []
    margin = getMargin()
    minIntensity = 40
    for i in xrange(margin, len(imageData) - margin):
        for j in xrange(margin, len(imageData[0]) - margin):
            if(imageData[i][j] > minIntensity and isLocalMax(imageData, i, j, margin)):
                #TODO: Finish curvature computation to remove some (but not all) 
                #of the edge points
                #minCurvature, maxCurvature = getCurvatureRange(imageData, i, j)
                #nodes.append(Node(i, j, minCurvature, maxCurvature))
                #print i, j, minCurvature, maxCurvature
                #I'm using a hack that manually chooses the minIntensity, which may
                #vary for different input graphs
                nodes.append(Node(j, i, imageData[i][j]))
                #print j, i, margin
                #print imageData[i - margin:i + margin + 1,j - margin:j + margin + 1]
    nodes.sort(key = lambda node: -node.intensity)
    #for node in nodes:
    #    print node
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
        #Draw plane between the three points in our triangle
        pass
        
    
     
#Return squared distance between two points
def invDistanceSq(x1, y1, x2, y2):
    return 1/((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))


#Performs a Hough transform to find all edges
def findEdgesHough(imageData):
    lines = []
    height, width = imageData.shape
    h, theta, d = hough_line(imageData)
    
    """
    targetTheta = 0
    minThetaDiff = 100
    minThetaIndex = 0
    print theta
    for i in range(1, len(theta)):
        thetaDiff = abs(targetTheta - theta[i])
        if(thetaDiff < minThetaDiff):
            minThetaDiff = thetaDiff
            minThetaIndex = i
            
    targetDist = 5
    minDistDiff = 100
    minDistIndex = 0
    for i in range(1, len(d)):
        distDiff = abs(targetDist - d[i])
        if(distDiff < minDistDiff):
            minDistDiff = distDiff
            minDistIndex = i
            
    print "Theta range: ", theta[minThetaIndex - 5: minThetaIndex + 6]
    print "Distance range: ", d[minDistIndex - 5 : minDistIndex + 6]        
    print "Target line value: ", h[minDistIndex-5:minDistIndex+6, minThetaIndex-5:minThetaIndex+6]
    """
    
    for hval, angle, dist, in zip(*hough_line_peaks(h, theta, d)):
        print hval, angle*180.0/math.pi, dist
        #In theory, our line should go from (0, y0) to (width, y1)
        #But if y0 and y1 are outside our bounds, we adjust them
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - width*np.cos(angle)) / np.sin(angle)
        x0, y0, x1, y1 = scaleLines(y0, y1, width, height)
        print x0, y0, x1, y1
                
        lines.append(((x0, y0), (x1, y1)))
        
    return lines
    

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
    #for i in xrange(len(imageData)):
    #    for j in xrange(150, len(imageData[0])):
    #        imageData[i][j] = 0
    graph = nx.Graph()
    nodes = findNodes(imageData)
    edges = findEdgesHough(imageData)
    
    #print "Nodes: ", nodes
    #print "Edges: ", edges
    
    #Find all of the nodes within maxDist of each edge
    print "Finding nodes for each edge"
    maxDist = 20
    for edge in edges:
        #print "Edge: ", edge
        edgeNodes = []
        for node in nodes:
            #print "Node: ", node
            #print "Distance: ", findDistPointToLine(node, edge)
            dist = findDistPointToLine(node, edge)
            if(findDistPointToLine(node, edge) < maxDist):
                #print "Node: ", node
                #print "Distance: ", dist
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
                
    
    #printTest(imageData)
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
        slope = (y1 - y0)/(x1 - x0)
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
    for hval, angle, dist, in zip(*hough_line_peaks(h, theta, d, min_angle = 10)):
        print hval, angle*180.0/math.pi, dist
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - width*np.cos(angle)) / np.sin(angle)
        x0, y0, x1, y1 = scaleLines(y0, y1, width, height)
        #print "Coordinates after: ", x0, y0, x1, y1
        ax[2].plot((x0, x1), (y0, y1), '-r')
        
    ax[2].axis((0, width, height, 0))
    ax[2].set_title("Detected lines")
    ax[2].axis("image")
    
    #edges = canny(imageData, 2, 1, 25)
    #lines = probabilistic_hough_line(edges, threshold = 10, line_length = 5, line_gap = 3)
    
    #fig2, ax = plt.subplots(1, 3, figsize = (8, 3))
    
    #ax[0].imshow(image, cmap = plt.cm.gray)
    #ax[0].set_title("Input image")
    #ax[0].axis("image")
    
    #ax[1].imshow(edges, cmap = plt.cm.gray)
    #ax[1].set_title("Canny edges")
    #ax[1].axis("image")
    
    #ax[2].imshow(edges * 0)
    
    #for line in lines:
    #    p0, p1 = line
    #    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
        
    #ax[2].set_title("Probabilistic Hough")
    #ax[2].axis("image")
    plt.show()
    
    
    
#Converts image to a grayscale numpy array, where 0 = white and 255 = black
def toMatrix(image, width, height):
    image = image.convert("L")
    imageData = list(image.getdata())
    for i in xrange(len(imageData)):
        imageData[i] = 255 - imageData[i] #Flip scale to 0 = white
    
    dataMatrix = np.asarray(imageData).reshape(width, height)    
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


if __name__ == '__main__':
    #testInterpolateVal()
    #runTests()
    image = Image.open("../Images/image5.png")
    graph = readImage(image)
    print graph.nodes()
    print graph.edges()