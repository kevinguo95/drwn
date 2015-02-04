import glob
import os
import csv
from math import log


def getLoss(sheetName, fileName, weight):
	with open(sheetName, 'rb') as file:
		table = csv.reader(file)
		for row in table:
			if row[0] == (fileName + '.bmp'):
				if (weight == 0):
					if (row[1] == ''): return 0
					else: return float(row[1])
				else:
					pos = int(log(weight, 2)) + 2
					if (row[pos] == ''): return 0
					else: return float(row[pos])

def writeRow (fileExt):
	fileName = os.path.splitext(fileExt)[0]
	histMin = readCSV('./tables/histogram_3bit.csv', fileName)
	gmmMin = readCSV('./tables/drwnGrabCutStatsGMM.csv', fileName)
	histZeroLoss = getLoss('./tables/histogram_3bit.csv', fileName, 0)
	gmmZeroLoss = getLoss('./tables/drwnGrabCutStatsGMM.csv', fileName, 0)
	histLoss = getLoss('./tables/histogram_3bit.csv', fileName, histMin)
	gmmLoss = getLoss('./tables/drwnGrabCutStatsGMM.csv', fileName, gmmMin)
	sizeString = "width=\"80%\" height=\"80%\""
	f.write("<tr>\n")
	f.write("<td><img src= \"./input/images/%s\" %s></td>\n" % (fileExt, sizeString))
	f.write("<td><img src= \"./input/masks/%s.bmp\" %s></td>\n" % (fileName, sizeString))
	f.write("<td><img src= \"./output/groundtruth/%s.bmp\" %s></td>\n" % (fileName, sizeString))
	f.write("<td><img src= \"./output/GMM/%s_mask_0.png\" %s></td>\n" % (fileName, sizeString))
	f.write("<td><img src= \"./output/histogram_3bit/%s_mask_0.png\" %s></td>\n" % (fileName, sizeString))	
	f.write("<td><img src= \"./output/GMM/%s_mask_%d.png\" %s></td>\n" % (fileName, gmmMin, sizeString))
	f.write("<td><img src= \"./output/histogram_3bit/%s_mask_%d.png\" %s></td>\n" % (fileName, histMin, sizeString))
	f.write("</tr>\n\n")
	f.write("<tr>\n<td></td><td></td><td></td>\n<td><font size = \"30\">%f</font></td>" % gmmZeroLoss)
	f.write("<td><font size = \"30\">%f</font></td>\n<td><font size = \"30\">%f  %d</font></td>\n" % (histZeroLoss, gmmLoss, gmmMin))
	f.write("<td><font size = \"30\">%f  %d</font></td>\n\n" % (histLoss, histMin))
	

def readCSV(sheetName, fileName):
	with open(sheetName, 'rb') as file:
		table = csv.reader(file)
		for row in table:
			colNum = 0
			
			if (row[0] == (fileName + '.bmp')):
				for col in row:
					if row[colNum] == '': continue
					if colNum > 10:
						continue
					if col == min(row):
						if colNum == 1:
							return 0
						else:
							return 2**(colNum-2)
					colNum = colNum + 1
	return 0

		
f= open ('results.html', 'w')
f.write("<html>\n<head>\n<title>Results</title>\n</head>\n")
f.write("<body>\n\n<table>\n\n")
f.write("<tr>\n\n<td><font size = \"30\">Input Image</font></td>\n<td><font size = \"30\">Mask</font></td>\n")
f.write("<td><font size = \"30\">Ground-Truth</font></td>\n<td><font size = \"30\">GMM pairwise w = 0</font></td>\n")
f.write("<td><font size = \"30\">Histogram pairwise w = 0</font></td>\n<td><font size = \"30\">GMM optimal w</font></td>\n<td><font size = \"30\">Histogram optimal w</font></td>\n</tr>\n")
fileList = glob.glob('./input/images/*.jpg')
fileList.extend(glob.glob('./input/images/*.bmp'))
fileList.extend(glob.glob('./input/images/*.JPG'))
for file in fileList:
	fileExt = os.path.basename(file)
	writeRow(fileExt)
	
f.write("</table>\n\n</body>\n</html>")




	
					
					
		