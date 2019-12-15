import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import imageio
import math as mt
import time
import cv2 as cv
import argparse

class relaxationAnalysis:
    def __init__ (self, dens = 300, sys = 1, v0 = 1.0, initMethod = 'half', pos = 0, sigma = 0., acc = 0, contourDiv = 30, fps=5):
        self.density = dens
        self.system = sys
        self.potentialDiff = float(v0)
        self.initMethod = initMethod
        self.sigma = sigma
        self.mu = float(self.density) / 2.0
        self.baseGrid = []
        self.curGrid = []
        self.totalGrids = []
        self.eletricFields = []   #vectors
        self.fingerPos = pos
        self.acc = acc
        self.contourDiv = contourDiv
        self.fps = fps
        
        self.iterTime = 0
        self.timeSeq = []
        self.densityPics = []
        self.contourPics = []
        self.eletricPics = []

        
    def get2DGaussDist(self, x, y):
        rSq = float((x-self.mu)**2 + (y-self.mu)**2) / float((self.density-self.mu)**2 + (self.density-self.mu)**2)
        po = -rSq/(2*(float(self.sigma))**2)
        N = 1.0/(2 * np.pi * float(self.sigma)**2)
        z = self.potentialDiff * N * mt.exp(po)
        
        return z

    def getElecVec(self, grid, i, j):
        ex = -(grid[i + 1][j] - grid[i - 1][j]) / 2
        ey = -(grid[i][j + 1] - grid[i][j - 1]) / 2
        return ex, ey
    
    def keepFinger(self, grid):
        if self.system == 3:
            for i in range(self.density):
                for j in range(self.density):
                    if i < self.density / 2 and (j == self.fingerPos or j == self.fingerPos + 1):
                        grid[i][j] = self.potentialDiff
        return grid
    
    def initGrid(self, grid):
        if self.system == 1:
            for i in range(self.density):
                if i == 0:
                    col0 = [self.potentialDiff for j in range(self.density)]
                    grid.append(col0)
                else:
                    curCol = [0. for j in range(self.density)]
                    grid.append(curCol)
        elif self.system == 2:
            for i in range(self.density):
                if i != self.density - 1:
                    curCol = [0. for j in range(self.density)]
                    if i != 0: curCol[0], curCol[-1] = self.potentialDiff, self.potentialDiff
                else:
                    curCol = [self.potentialDiff for j in range(self.density)]
                grid.append(curCol)
        elif self.system == 3:
            for i in range(self.density):
                if i == 0:
                    col0 = [self.potentialDiff for j in range(self.density)]
                    grid.append(col0)
                elif i < self.density / 2:
                    curCol = [0. for j in range(self.density)]
                    curCol[self.fingerPos], curCol[self.fingerPos+1] = self.potentialDiff, self.potentialDiff
                    grid.append(curCol)
                else:
                    curCol = [0. for j in range(self.density)]
                    grid.append(curCol)
        else:
            raise ValueError
        
    def assignInitValue(self, grid, isBase=False):
        for i in range(1, self.density - 1):
            for j in range(1, self.density - 1):
                if self.initMethod == 'Half':
                    grid[i][j] = float(self.potentialDiff)/2.0
                elif self.initMethod == 'Gauss':
                    grid[i][j] = self.get2DGaussDist(float(i), float(j))
                elif self.initMethod == 'Random':
                    grid[i][j] = self.potentialDiff * np.random.random([1,1])[0][0]
                else:
                    raise ValueError
                    
        grid = self.keepFinger(grid)
        
        if isBase:
            self.totalGrids.append(grid)
            self.curGrid = grid


        
    def getElectricField(self, grid):
        curElecField = []
        for i in range(1, self.density-1):
            colElecField = []
            for j in range(1, self.density-1):
                ex, ey = self.getElecVec(grid, i, j)
                colElecField.append([ex, ey])
            curElecField.append(colElecField)
        self.eletricFields.append(curElecField)


    def iterGrid(self, grid):
        nextGrid = []
        self.initGrid(nextGrid)
        self.assignInitValue(nextGrid)
        for i in range(1, self.density - 1):
            for j in range(1, self.density - 1):
                nextGrid[i][j] = 0.25 * (grid[i+1][j] + 
                                         grid[i-1][j] + 
                                         grid[i][j-1] + 
                                         grid[i][j+1])
        nextGrid = self.keepFinger(nextGrid)
        self.totalGrids.append(nextGrid)
        self.curGrid = nextGrid
        self.iterTime += 1
        
        return nextGrid
    
    def isStop(self):
        if self.iterTime == 0:
            return False
        for i in range(1, self.density - 1):
            for j in range(1, self.density - 1):
                
                gridAcc = abs((self.totalGrids[-1][i][j] - self.totalGrids[-2][i][j]) / self.totalGrids[-2][i][j])
                if gridAcc > self.acc:
                    return False
        
        return True
    
    def plotDensity(self, grid):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim((0, self.density))
        im = ax.imshow(grid, cmap='rainbow')
        plt.colorbar(im)
        imgName = './pics/DensityPlot_System_' + str(self.system) + '_Density_' + str(self.density) + '_InitMethod_' + self.initMethod + '_Acc_' + str(self.acc) + '_Iter_' + str(self.iterTime) + '.jpg'
        plt.savefig(imgName)
        self.densityPics.append(imgName)
        plt.close('all')
        # plt.show()
        
    def plotContour(self, grid):
        x = np.arange(0, self.density, 1)
        y = np.arange(0, self.density, 1)
        X, Y = np.meshgrid(x, y)
        plt.figure()
        im = plt.contourf(X, Y, grid, self.contourDiv, cmap='rainbow')
        plt.contour(X, Y, grid, self.contourDiv)
        plt.colorbar(im)
        imgName = './pics/ContourPlot_System_' + str(self.system) + '_Density_' + str(self.density) + '_InitMethod_' + self.initMethod + '_Acc_' + str(self.acc) + '_Iter_' + str(self.iterTime) + '.jpg'
        plt.savefig(imgName)
        self.contourPics.append(imgName)
        plt.close('all')
        # plt.show()

    def plotVector(self, grid):
        self.getElectricField(grid)
        plt.figure()
        for i in range(0, self.density - 2, 15):
            for j in range(0, self.density - 2, 15):
                vec = self.eletricFields[-1][i][j]
                plt.quiver(j, i, vec[1], vec[0], angles='xy', scale_units='xy', scale=0.0005)
        imgName = './pics/ElectricFieldPlot_System_' + str(self.system) + '_Density_' + str(self.density) + '_InitMethod_' + self.initMethod + '_Acc_' + str(self.acc) + '_Iter_' + str(self.iterTime) + '.jpg'
        plt.savefig(imgName)
        self.eletricPics.append(imgName)
        plt.close('all')
        # plt.show()

    def plotAnima(self):
        outfilename = './pics/GifPlot_System_' + str(self.system) + '_Density_' + str(self.density) + '_InitMethod_' + self.initMethod + '_Acc_' + str(self.acc) + '_TerminalIterateTime_' + str(self.iterTime) + '.gif'
        filenames = self.contourPics
        frames = []
        for image_name in filenames:
            im = cv.imread(image_name)
            im = im[...,::-1]
            frames.append(im)
        imageio.mimsave(outfilename, frames, 'GIF', fps=self.fps)


        
    def run(self):
        self.initGrid(self.baseGrid)
        self.assignInitValue(self.baseGrid, isBase=True)
        self.plotContour(self.baseGrid)
        self.plotDensity(self.baseGrid)
        while not self.isStop():
            self.iterGrid(self.curGrid)
            print('current iteration time: ' + str(self.iterTime), flush=True, end='\r')
            if self.iterTime % 50 == 0:
                self.plotContour(self.curGrid)
                self.plotDensity(self.curGrid)
                self.plotVector(self.curGrid)
        self.plotContour(self.curGrid)
        self.plotDensity(self.curGrid)
        self.plotVector(self.curGrid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dense', type=int)
    parser.add_argument('-a', '--acc', type=float)
    parser.add_argument('-s', '--sys', type=int)
    parser.add_argument('-m', '--method')
    args = parser.parse_args()

    proj = relaxationAnalysis(sys=args.sys, dens=args.dense, initMethod=args.method, contourDiv=30, acc=args.acc, pos=(args.dense//2), sigma = 0.5)
    print('\nsimulate for system %d with density %d with accuracy %f, initial method %s' %(args.sys, args.dense, args.acc, args.method))
    proj.run()
    print('\nTerminal iteration time: %d' % (proj.iterTime))
    proj.plotAnima()
