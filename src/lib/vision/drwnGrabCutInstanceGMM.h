/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGrabCutInstanceGMM.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**				Kevin Guo <Kevin.Guo@nicta.com.au>
**
*****************************************************************************/

#pragma once
#include "Eigen/Core"

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnPGM.h"
#include "drwnVision.h"

#include "drwnGrabCutInstance.h"


class drwnGrabCutInstanceGMM :public drwnGrabCutInstance
{
public:
	drwnGrabCutInstanceGMM();
	~drwnGrabCutInstanceGMM();

	//update colour models
	void updateColourModels(drwnGaussianMixture& fgColourModel,
		drwnGaussianMixture& bgColourModel) {
		_fgColourModel = fgColourModel;
		_bgColourModel = bgColourModel;
		updateUnaryPotentials();
	}

	//declaration to prevent unused pure virtual function 
	void updateColourModels(drwnColourHistogram& fgColourModel,
		drwnColourHistogram& bgColourModel) {}

	//! learn a histogram model for pixels in masked region
	void learnColourModel(const cv::Mat& mask, bool fg);

	//! save and load colour models
	void saveColourModels(const char *filename) const;
	void loadColourModels(const char *filename);

protected:
	drwnGaussianMixture _fgColourModel; //!< foreground colour model
	drwnGaussianMixture _bgColourModel; //!< background colour model

	//! update unary potential from colour models
	void updateUnaryPotentials();

	//! extract pixel colour as a 3-vector
	inline vector <double> pixelColour(int y, int x) const;
};

