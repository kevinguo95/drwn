#pragma once
#include "Eigen/Core"

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnPGM.h"
#include "drwnVision.h"

#include "drwnColourHistogram.h"
#include "drwnGrabCutInstance.h"


using namespace std;
using namespace Eigen;

class drwnGrabCutInstanceHistogram : public drwnGrabCutInstance
{
public:
	
	//constructor
	drwnGrabCutInstanceHistogram();
	//destructor
	~drwnGrabCutInstanceHistogram();

	//update colour models
	void updateColourModels(drwnColourHistogram& fgColourModel,
		drwnColourHistogram& bgColourModel) {
		_fgColourModel = fgColourModel;
		_bgColourModel = bgColourModel;
		updateUnaryPotentials();
	}

	//declaration to prevent unused pure virtual function 
	void updateColourModels(drwnGaussianMixture& fgColourModel,
		drwnGaussianMixture& bgColourModel) {}

	//! learn a histogram model for pixels in masked region
	void learnColourModel(const cv::Mat& mask, bool fg);

	//! save and load colour models
	void saveColourModels(const char *filename) const;
	void loadColourModels(const char *filename);

protected:
	drwnColourHistogram _fgColourModel; //!< foreground colour model
	drwnColourHistogram _bgColourModel; //!< background colour model

	//! update unary potential from colour models
	virtual void updateUnaryPotentials();

	//! extract pixel colour as a 3-vector
	inline vector<unsigned char> pixelColour(int y, int x) const;

};

