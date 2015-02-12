/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnColourHistogram.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              Kevin Guo <kevin.guo@nicta.com.au>
**
*****************************************************************************/

#include <cstdlib>
#include <iomanip>
#include <vector>

#include "cv.h"

#include "drwnBase.h"
#include "drwnColourHistogram.h"

// drwnColourHistogram ------------------------------------------------------

drwnColourHistogram::drwnColourHistogram(double pseudoCounts, unsigned channelBits) :
    _channelBits(channelBits), _pseudoCounts(pseudoCounts), _totalCounts(0.0)
{
    DRWN_ASSERT((channelBits > 0) && (channelBits <= 8));
    DRWN_ASSERT(pseudoCounts >= 0.0);

    _mask =  0xff ^ (0xff >> _channelBits);
    const size_t bins = 0x00000001 << (3 * _channelBits);
    _histogram.resize(bins, 0.0);
}

drwnColourHistogram::drwnColourHistogram(const drwnColourHistogram& histogram) :
    _channelBits(histogram._channelBits), _mask(histogram._mask),
    _pseudoCounts(histogram._pseudoCounts), _histogram(histogram._histogram),
    _totalCounts(histogram._totalCounts)
{
    // do nothing
}

void drwnColourHistogram::clear(double pseudoCounts)
{
    DRWN_ASSERT(pseudoCounts >= 0.0);
    _pseudoCounts = pseudoCounts;

    std::fill(_histogram.begin(), _histogram.end(), 0.0);
    _totalCounts = 0.0;
}


bool drwnColourHistogram::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "channelBits", toString(_channelBits).c_str(), false);
    drwnAddXMLAttribute(xml, "pseudoCounts", toString(_pseudoCounts).c_str(), false);
    drwnAddXMLAttribute(xml, "totalCounts", toString(_totalCounts).c_str(), false);

    drwnXMLNode *node = drwnAddXMLChildNode(xml, "histogram", NULL, false);
    drwnXMLUtils::serialize(*node, (const char *)&_histogram[0], _histogram.size() * sizeof(double));

    return true;
}

bool drwnColourHistogram::load(drwnXMLNode& xml)
{
    _channelBits = atoi(drwnGetXMLAttribute(xml, "channelBits"));
    _pseudoCounts = atoi(drwnGetXMLAttribute(xml, "pseudoCounts"));
    _totalCounts = atof(drwnGetXMLAttribute(xml, "totalCounts"));
    _mask =  0xff ^ (0xff >> _channelBits);

    const size_t bins = 0x00000001 << (3 * _channelBits);
    _histogram.resize(bins);
    drwnXMLNode *node = xml.first_node("histogram");
    DRWN_ASSERT(node != NULL);
    drwnXMLUtils::deserialize(*node, (char *)&_histogram[0], bins * sizeof(double));


    return true;
}

void drwnColourHistogram::accumulate(unsigned char red, unsigned char green, unsigned char blue)
{
	//! \todo interpolate between 8 neighbouring bins
	const unsigned indx_r = (red & _mask) >> (8 - _channelBits);
	const unsigned indx_g = (green & _mask) >> (8 - _channelBits);
	const unsigned indx_b = (blue & _mask) >> (8 - _channelBits);
	const unsigned dist_r = red & ~_mask;
	const unsigned dist_g = green & ~_mask;
	const unsigned dist_b = blue & ~_mask;

	int middle = pow(2, 7 - _channelBits);

	int dir_r = dist_r < middle ? -1 : 1;
	int dir_g = dist_g < middle ? -1 : 1;
	int dir_b = dist_b < middle ? -1 : 1;


	const unsigned indx = (indx_r << (2 * _channelBits)) | (indx_g << _channelBits) | indx_b;
	//cout << ~_mask << endl;
	//cout << " red = " << dir_r << " green = " << dir_g << " blue = " << dir_b << endl;
	//cout << " red = " << dist_r << " green = " << dist_g << " blue = " << dist_b << endl;
	//vector to store bin ratios
	std::vector<double> ratio = calcRatios(dist_r, dist_g, dist_b, indx);

	_histogram[indx] += ratio.at(0);
	//cout << indx << " += " << ratio.at(0) << endl;
	if (ratio.at(1) != 0) {
		DRWN_ASSERT(indx + dir_b < _histogram.size());
		_histogram[indx + dir_b] += ratio.at(1);
	}
	if (ratio.at(2) != 0) {
		DRWN_ASSERT(indx + dir_g*(1 << _channelBits) < _histogram.size());
		_histogram[indx + dir_g*(1 << _channelBits)] += ratio.at(2);
	}
	if (ratio.at(3) != 0) {
		DRWN_ASSERT(indx + dir_b + dir_g*(1 << _channelBits) < _histogram.size());
		_histogram[indx + dir_b + dir_g*(1 << _channelBits)] += ratio.at(3);
	}
	if (ratio.at(4) != 0) {
		DRWN_ASSERT(indx + dir_r*(1 << (2 * _channelBits)) < _histogram.size());
		_histogram[indx + dir_r*(1 << (2 * _channelBits))] += ratio.at(4);

	}
	if (ratio.at(5) != 0) {
		DRWN_ASSERT(indx + dir_b + dir_r*(1 << (2 * _channelBits)) < _histogram.size());
		_histogram[indx + dir_b + dir_r*(1 << (2 * _channelBits))] += ratio.at(5);
	}
	if (ratio.at(6) != 0) {
		DRWN_ASSERT(indx + dir_g*(1 << _channelBits) + dir_r*(1 << (2 * _channelBits)) < _histogram.size());
		_histogram[indx + dir_r*(1 << (2 * _channelBits)) + dir_g*(1 << _channelBits)] += ratio.at(6);
	}

	if (ratio.at(7) != 0) {
		DRWN_ASSERT(indx + dir_b + dir_g*(1 << _channelBits) + dir_r*(1 << (2 * _channelBits)) < _histogram.size());
		_histogram[indx + dir_b + dir_g*(1 << _channelBits) + dir_r*(1 << (2 * _channelBits))] += ratio.at(7);
	}

	_totalCounts += 1.0;	
}

double drwnColourHistogram::probability(unsigned char red, unsigned char green, unsigned char blue) const
{
    const unsigned indx_r = (red & _mask) >> (8 - _channelBits);
    const unsigned indx_g = (green & _mask) >> (8 - _channelBits);
    const unsigned indx_b = (blue & _mask) >> (8 - _channelBits);
    const unsigned indx = (indx_r << (2 * _channelBits)) | (indx_g << _channelBits) | indx_b;
	//cout << "red = " << (int)red << " green = " << (int)green << " blye = " << (int)blue << endl;
	//cout << "indx = " << indx <<  " _histogram[indx] = " << _histogram[indx];
    return (_histogram[indx] + _pseudoCounts) / (_totalCounts + _histogram.size() * _pseudoCounts);
}

cv::Mat drwnColourHistogram::visualize() const
{
    const unsigned legendWidth = 16;
    const unsigned spaceWidth = 8;
    const unsigned barWidth = 100;

    const double maxCount = drwn::maxElem(_histogram) + _pseudoCounts;

    cv::Mat canvas(_histogram.size(), legendWidth + 2 * spaceWidth + barWidth, CV_8UC3, cv::Scalar::all(0xff));
    for (unsigned indx = 0; indx < _histogram.size(); indx++) {
        unsigned char red = ((indx >> (2 * _channelBits)) << (8 - _channelBits)) & _mask;
        unsigned char green = ((indx >> _channelBits) << (8 - _channelBits)) & _mask;
        unsigned char blue = (indx << (8 - _channelBits)) & _mask;

        cv::line(canvas, cv::Point(0, indx), cv::Point(legendWidth, indx), CV_RGB(red, green, blue), 1);

        double v = (_histogram[indx] + _pseudoCounts) / maxCount;
        cv::line(canvas, cv::Point(legendWidth + spaceWidth, indx),
            cv::Point(legendWidth + spaceWidth + v * barWidth, indx), CV_RGB(0x7f, 0x7f, 0x7f), 1);
    }

    return canvas;
}

vector<double> drwnColourHistogram::calcRatios(unsigned dist_r, unsigned dist_g, unsigned dist_b, unsigned indx) {
	std::vector<double> distances;
	std::vector<double> ratio;
	int binSize = pow(2, 8 - _channelBits);
	int middle = binSize/2;

	unsigned indx_r = indx >> 2 * _channelBits;
	unsigned indx_g = (indx >> _channelBits) & ((1 << _channelBits) - 1);
	unsigned indx_b = indx & ((1 << _channelBits) - 1);

	//convert values to distance from midpoint
	int red = abs((int)dist_r - middle);
	int green = abs((int)dist_g - middle);
	int blue = abs((int)dist_b - middle);

	//cout << " red = " << red << " green = " << green << " blue = " << blue << endl;
	DRWN_ASSERT(red <= middle && green <= middle && blue <= middle && red >= 0 && green >= 0 && blue >= 0);

	//calculate inverse of distances to bin
	distances.push_back(1/drwnColourHistogram::distance(red, green, blue));
	if (!isEdge(indx_b, dist_b)) distances.push_back(1/drwnColourHistogram::distance(red, green, binSize - blue));
	else distances.push_back(0);

	if (!isEdge(indx_g, dist_g)) distances.push_back(1 / drwnColourHistogram::distance(red, binSize - green, blue));
	else distances.push_back(0);

	if (!isEdge(indx_b, dist_b) && !isEdge(indx_g, dist_g))
		distances.push_back(1/drwnColourHistogram::distance(red, binSize - green, binSize - blue));
	else distances.push_back(0);

	if (!isEdge(indx_r, dist_r)) distances.push_back(1/drwnColourHistogram::distance(binSize - red, green, blue));
	else distances.push_back(0);

	if (!isEdge(indx_b, dist_b) && !isEdge(indx_r, dist_r))
		distances.push_back(1/drwnColourHistogram::distance(binSize - red, green, binSize - blue));
	else distances.push_back(0);

	if (!isEdge(indx_r, dist_r) && !isEdge(indx_g, dist_g))
		distances.push_back(1/drwnColourHistogram::distance(binSize - red, binSize - green, blue));
	else distances.push_back(0);

	if (!isEdge(indx_b, dist_b) && !isEdge(indx_r, dist_r) && !isEdge(indx_g, dist_g))
	distances.push_back(1/drwnColourHistogram::distance(binSize - red, binSize - green, binSize - blue));
	else distances.push_back(0);

	DRWN_ASSERT(distances.size() == 8);
	//find sum of distances
	double sum = 0.0;
	for (int i = 0; i < distances.size(); i++) {
		sum += distances.at(i);
	}

	//calculate bin ratios
	for (int i = 0; i < distances.size(); i++) {
		if (!isfinite(distances.at(i))) {
			ratio.push_back(1);
		} else {
			ratio.push_back(distances.at(i) / sum);
		}
		DRWN_ASSERT(isfinite(ratio.at(i)));
	}
	//DRWN_ASSERT(ratioSum == 1);
	//cout << ratio.at(0) << " " << ratio.at(1) << " " << ratio.at(2) << " " << ratio.at(3) << " " << ratio.at(7) << endl;
	DRWN_ASSERT(ratio.size() == 8);
	return ratio;
}

bool drwnColourHistogram::isEdge(int indx, int dist)
{ 
	int middle;
	if (_channelBits == 8) middle = 0;
	else middle = pow(2, 7 - _channelBits);
	return(((indx == 0) && (dist < middle)) || ((indx == ((1 << _channelBits)-1) && (dist >= middle)))); 
	
}