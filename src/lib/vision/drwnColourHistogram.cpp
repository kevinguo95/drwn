/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnColourHistogram.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
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

void drwnColourHistogram::clear(double pseudoCounts) 
{
    DRWN_ASSERT(pseudoCounts >= 0.0);
    _pseudoCounts = pseudoCounts;

    std::fill(_histogram.begin(), _histogram.end(), 0.0);
    _totalCounts = 0.0;
}

void drwnColourHistogram::accumulate(unsigned char red, unsigned char green, unsigned char blue)
{
    //! \todo interpolate between 8 neighbouring bins
    const unsigned indx_r = (red & _mask) >> (8 - _channelBits);
    const unsigned indx_g = (green & _mask) >> (8 - _channelBits);
    const unsigned indx_b = (blue & _mask) >> (8 - _channelBits);
    //const unsigned dist_r = red & !_mask;
    //const unsigned dist_g = green & !_mask;
    //const unsigned dist_b = blue & !_mask;
    
    const unsigned indx = (indx_r << (2 * _channelBits)) | (indx_g << _channelBits) | indx_b;
    _histogram[indx] += 1.0;
    _totalCounts += 1.0;
}

double drwnColourHistogram::probability(unsigned char red, unsigned char green, unsigned char blue) const
{
    const unsigned indx_r = (red & _mask) >> (8 - _channelBits);
    const unsigned indx_g = (green & _mask) >> (8 - _channelBits);
    const unsigned indx_b = (blue & _mask) >> (8 - _channelBits);
    const unsigned indx = (indx_r << (2 * _channelBits)) | (indx_g << _channelBits) | indx_b;

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

bool drwnColourHistogram::save(drwnXMLNode& xml) const
{
	DRWN_TODO;
}

bool drwnColourHistogram::load(drwnXMLNode& xml) 
{
	DRWN_TODO;
}