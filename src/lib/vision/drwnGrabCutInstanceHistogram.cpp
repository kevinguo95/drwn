/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGrabCutInstanceHistogram.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**				Kevin Guo <Kevin.Guo@nicta.com.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "drwnGrabCutInstanceHistogram.h"


using namespace std;
using namespace Eigen;

drwnGrabCutInstanceHistogram::drwnGrabCutInstanceHistogram() :
	drwnGrabCutInstance(), _fgColourModel(pseudoCounts, channelBits), _bgColourModel(pseudoCounts, channelBits)
{

}


drwnGrabCutInstanceHistogram::~drwnGrabCutInstanceHistogram()
{
	free();
}


void drwnGrabCutInstanceHistogram::learnColourModel(const cv::Mat& mask, bool fg)
{
	DRWN_ASSERT((mask.rows == _img.rows) && (mask.cols == _img.cols));
	DRWN_FCN_TIC;

	drwnColourHistogram * model;
	if (fg == FOREGROUND) {
		model = &_fgColourModel;
	}
	else {
		model = &_bgColourModel;
	}

	model->clear();

	// extract colour samples for pixels in mask
	vector<unsigned char> clrVector;

	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			if (mask.at<unsigned char>(y, x) != 0x00) {
				clrVector = drwnGrabCutInstanceHistogram::pixelColour(y, x);
				if (drwnGrabCutInstance::bInterpolate) {
					model->interpolatedAccumulate(clrVector.at(0), clrVector.at(1), clrVector.at(2));
				} else {
					model->accumulate(clrVector.at(0), clrVector.at(1), clrVector.at(2));
				}
			}

		}
	}
	DRWN_FCN_TOC;
}

// save colour models
void drwnGrabCutInstanceHistogram::saveColourModels(const char *filename) const
{
	DRWN_ASSERT(filename != NULL);
	drwnXMLDoc xml;
	drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnGrabCutInstance", NULL, false);
	drwnAddXMLAttribute(*node, "drwnVersion", DRWN_VERSION, false);

	drwnXMLNode *child = drwnAddXMLChildNode(*node, "foreground", NULL, false);
	_fgColourModel.save(*child);
	child = drwnAddXMLChildNode(*node, "background", NULL, false);
	_bgColourModel.save(*child);

	ofstream ofs(filename);
	ofs << xml << endl;
	DRWN_ASSERT(!ofs.fail());
	ofs.close();
}


// load colour models
void drwnGrabCutInstanceHistogram::loadColourModels(const char *filename)
{
	DRWN_ASSERT(filename != NULL);

	drwnXMLDoc xml;
	drwnXMLNode *node = drwnParseXMLFile(xml, filename, "drwnGrabCutInstance");

	drwnXMLNode *subnode = node->first_node("foreground");
	DRWN_ASSERT(subnode != NULL);
	_fgColourModel.load(*subnode);
	subnode = node->first_node("background");
	DRWN_ASSERT(subnode != NULL);
	_bgColourModel.load(*subnode);

	updateUnaryPotentials();
}

void drwnGrabCutInstanceHistogram::updateUnaryPotentials()
{
	DRWN_ASSERT(_img.data != NULL);

	DRWN_FCN_TIC;
	DRWN_LOG_VERBOSE("updating unary potentials for " << toString(_img) << "...");
	if ((_unary.data == NULL) || (_unary.rows != _img.rows) || (_unary.cols != _img.cols)) {
		_unary = cv::Mat(_img.rows, _img.cols, CV_32FC1);
	}
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			// skip "known" pixels
			if (!isUnknownPixel(x, y)) {
				_unary.at<float>(y, x) = 0.0f;
				continue;
			}

			// evaluate difference in log-likelihood
			vector<unsigned char> colour(pixelColour(y, x));
			//cout << "y = " << y << " x = " << x << endl;
			double p_fg = _fgColourModel.probability(colour.at(0), colour.at(1), colour.at(2));
			double p_bg = _bgColourModel.probability(colour.at(0), colour.at(1), colour.at(2));
			//assert probabilities are between 0 and 1
			//cout << "p_fg = " << p_fg << " p_bg = " << p_bg << endl;
			DRWN_ASSERT((p_fg > 0 && p_fg <= 1 && p_bg > 0 && p_bg <= 1));
			DRWN_ASSERT(isfinite(log(p_fg)) && isfinite(log(p_bg)));
			_unary.at<float>(y, x) = (float)(log(p_fg) - log(p_bg));
			}
	}

	DRWN_FCN_TOC;
}

vector<unsigned char> drwnGrabCutInstanceHistogram::pixelColour(int y, int x) const
{

	const unsigned char *p = _img.ptr<const unsigned char>(y) +3 * x;
	vector<unsigned char> colour(3);
	colour[2] = p[0];
	colour[1] = p[1];
	colour[0] = p[2];

	return colour;
}