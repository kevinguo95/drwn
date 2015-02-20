/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGrabCutInstanceGMM.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**				Kevin Guo <Kevin.Guo@nicta.com.au>
**
*****************************************************************************/

#include "drwnGrabCutInstanceGMM.h"


drwnGrabCutInstanceGMM::drwnGrabCutInstanceGMM() :
	drwnGrabCutInstance(), _fgColourModel(3, numMixtures), _bgColourModel(3, numMixtures)
{

}


drwnGrabCutInstanceGMM::~drwnGrabCutInstanceGMM()
{
	free();
}

void drwnGrabCutInstanceGMM::learnColourModel(const cv::Mat& mask, bool fg)
{
	DRWN_ASSERT((mask.rows == _img.rows) && (mask.cols == _img.cols));
	if (maxSamples == 0) {
		DRWN_LOG_WARNING("skipping colour model learning (maxSamples is zero)");
		return;
	}

	DRWN_FCN_TIC;
	//check foreground or background
	drwnGaussianMixture * model;
	if (fg == FOREGROUND) model = &_fgColourModel;
	else model = &_bgColourModel;

	// extract colour samples for pixels in mask
	vector<vector<double> > data;
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			if (mask.at<unsigned char>(y, x) != 0x00) {
				data.push_back(pixelColour(y, x));
			}
		}
	}

	// subsample if too many
	data = drwn::subSample(data, maxSamples);
	DRWN_LOG_VERBOSE("learning " << numMixtures << "-component model using "
		<< data.size() << " pixels...");

	// check variance of data
	drwnSuffStats stats(3, DRWN_PSS_FULL);
	stats.accumulate(data);
	VectorXd mu = stats.firstMoments() / stats.count();
	MatrixXd sigma = stats.secondMoments() / stats.count() - mu * mu.transpose();
	double det = sigma.determinant();
	if (det <= 0.0) {
		DRWN_LOG_WARNING("no colour variation in data; adding noise (|Sigma| = " << det << ")");
		for (unsigned i = 0; i < data.size(); i++) {
			data[i][0] += 0.01 * (drand48() - 0.5);
			data[i][1] += 0.01 * (drand48() - 0.5);
			data[i][2] += 0.01 * (drand48() - 0.5);
		}
	}

	// learn model
	DRWN_ASSERT(data.size() > 1);
	model->initialize(3, std::min(numMixtures, (int)data.size() - 1));
	model->train(data);

	DRWN_FCN_TOC;
}


// save colour models
void drwnGrabCutInstanceGMM::saveColourModels(const char *filename) const
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
void drwnGrabCutInstanceGMM::loadColourModels(const char *filename)
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

void drwnGrabCutInstanceGMM::updateUnaryPotentials()
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
			vector<double> colour(drwnGrabCutInstanceGMM::pixelColour(y, x));
			double p_fg = _fgColourModel.evaluateSingle(colour);
			double p_bg = _bgColourModel.evaluateSingle(colour);
			DRWN_ASSERT(isfinite(p_fg) && isfinite(p_bg));
			_unary.at<float>(y, x) = (float)(p_fg - p_bg);
		}
	}

	DRWN_FCN_TOC;
}

vector<double> drwnGrabCutInstanceGMM::pixelColour(int y, int x) const
{
	const unsigned char *p = _img.ptr<const unsigned char>(y) +3 * x;
	vector<double> colour(3);
	colour[2] = (double)p[0] / 255.0;
	colour[1] = (double)p[1] / 255.0;
	colour[0] = (double)p[2] / 255.0;
	return colour;
}

