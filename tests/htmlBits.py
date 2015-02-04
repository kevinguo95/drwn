import html
import glob
import os
import csv
from math import log

def writeRow (fileExt):
	fileName = os.path.splitext(fileExt)[0]
	sizeString = "width=\"80%\" height=\"80%\""
	twoMin = html.readCSV('./tables/histogram_2bit.csv', fileName)
	threeMin = html.readCSV('./tables/histogram_3bit.csv', fileName)
	fourMin = html.readCSV('./tables/histogram_4bit.csv', fileName)
	fiveMin = html.readCSV('./tables/histogram_5bit.csv', fileName)
	sixMin = html.readCSV('./tables/histogram_6bit.csv', fileName)
	twoLoss = html.getLoss('./tables/histogram_2bit.csv', fileName, twoMin)
	threeLoss = html.getLoss('./tables/histogram_3bit.csv', fileName, threeMin)
	fourLoss = html.getLoss('./tables/histogram_4bit.csv', fileName, fourMin)
	fiveLoss = html.getLoss('./tables/histogram_5bit.csv', fileName, fiveMin)
	sixLoss = html.getLoss('./tables/histogram_6bit.csv', fileName, sixMin)
	
	f.write("<tr>\n")
	f.write("<td><img src= \"./input/images/%s\" %s></td>\n" % (fileExt, sizeString))
	f.write("<td><img src= \"./output/groundtruth/%s.bmp\" %s></td>\n" % (fileName, sizeString))
	f.write("<td><img src= \"./output/histogram_2bit/%s_mask_%d.png\" %s></td>\n" % (fileName, twoMin, sizeString))
	f.write("<td><img src= \"./output/histogram_3bit/%s_mask_%d.png\" %s></td>\n" % (fileName, threeMin, sizeString))
	f.write("<td><img src= \"./output/histogram_4bit/%s_mask_%d.png\" %s></td>\n" % (fileName, fourMin, sizeString))
	f.write("<td><img src= \"./output/histogram_5bit/%s_mask_%d.png\" %s></td>\n" % (fileName, fiveMin, sizeString))
	f.write("<td><img src= \"./output/histogram_6bit/%s_mask_%d.png\" %s></td>\n" % (fileName, sixMin, sizeString))
	
	f.write("</tr>\n\n")
	f.write("<tr>\n<td></td><td></td>\n<td><font size = \"30\">%f  %d</font></td>\n" % (twoLoss, twoMin))
	f.write("<td><font size = \"30\">%f  %d</font></td>\n" % (threeLoss, threeMin))
	f.write("<td><font size = \"30\">%f  %d</font></td>\n" % (fourLoss, fourMin))
	f.write("<td><font size = \"30\">%f  %d</font></td>\n" % (fiveLoss, fiveMin))
	f.write("<td><font size = \"30\">%f  %d</font></td></tr>\n\n" % (sixLoss, sixMin))
	
	

f = open ('bits.html', 'w')
f.write("<html>\n<head>\n<title>Results</title>\n</head>\n")
f.write("<body>\n\n<table>\n\n")
f.write("<tr>\n\n<td><font size = \"30\">Input Image</font></td>\n<td><font size = \"30\">Ground Truth</font></td>\n")
f.write("<td><font size = \"30\">2 Bit</font></td>\n<td><font size = \"30\">3 Bit</font></td>\n")
f.write("<td><font size = \"30\">4 Bit</font></td>\n<td><font size = \"30\">5 Bit</font></td>\n<td><font size = \"30\">6 Bit</font></td>\n</tr>\n")
fileList = glob.glob('./input/images/*.jpg')
fileList.extend(glob.glob('./input/images/*.bmp'))
fileList.extend(glob.glob('./input/images/*.JPG'))
for file in fileList:
	fileExt = os.path.basename(file)
	writeRow(fileExt)
	
f.write("</table>\n\n</body>\n</html>")