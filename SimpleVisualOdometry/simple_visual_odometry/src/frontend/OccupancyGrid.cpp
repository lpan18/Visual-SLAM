#include "frontend/OccupancyGrid.h"
using namespace std;
using namespace cv;

OccupancyGrid::OccupancyGrid()
{
	// initializer();
	initializer1();
}

void OccupancyGrid::initializer1()
{
	Ix = 10;
	Iy = 10;
	resetGrid1();
}

void OccupancyGrid::setImageSize1(size_t cols, size_t rows)
{
	Ix = cols / nx; // width of each cell
    Iy = rows / ny; // height of each cell
}

void OccupancyGrid::addPoint1(Point2f& p)
{
	size_t whichX = round(p.x / Ix);
    size_t whichY = round(p.y / Iy);

    if (whichX <= nx && whichY <= ny)
    {
		isFree[whichY][whichX] = false;
	}
}

bool OccupancyGrid::isNewFeature1(Point2f& p)
{
	// return true;
	size_t whichX = round(p.x / Ix);
    size_t whichY = round(p.y / Iy);
    bool isNew = true;

    for (size_t j = whichY - 1; j < whichY + 2; j++) {
		if (j < 0 || j >= ny) 	continue;

		for (size_t i = whichX - 1; i < whichX + 2; i++) {
			if (i < 0 || i >= nx) 	continue;

			bool cellIsFree = isFree[j][i];
			isNew = isNew && cellIsFree;

		}
	}
	return isNew;
}

void OccupancyGrid::resetGrid1()
{
	for (size_t j = 0; j < ny; j++)
	{
        for (size_t i = 0; i < nx; i++)
        {
            isFree[j][i] = true;
		}
	}
}
