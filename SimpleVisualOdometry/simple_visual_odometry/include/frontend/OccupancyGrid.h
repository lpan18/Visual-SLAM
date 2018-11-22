#define EIGEN_DONT_ALIGN_STATICALLY
#ifndef OCCUPANCYGRID_H_
#define OCCUPANCYGRID_H_

#include <vector>
#include <opencv2/opencv.hpp>

class OccupancyGrid
{
public:
	OccupancyGrid();

    void initializer();
	void setImageSize(size_t cols, size_t rows); // compute the size of cell (Ix and Iy) according to given image size and grid resolution
	void addPoint(cv::Point2f& p); // update the OccupancyGrid (isFree) when adding new features
	bool isNewFeature(cv::Point2f& p); // p is a newfeature when cell / neighbour cells are all free
	void resetGrid(); // rest isFree

    // implementing these 5 functions below
    void initializer1();
    void setImageSize1(size_t cols, size_t rows);
    void addPoint1(cv::Point2f& p);
    bool isNewFeature1(cv::Point2f& p);
    void resetGrid1();

private:
	// number of cells
	static const size_t nx = 32;
	static const size_t ny = 20;

	// Data needed by the algorithm
	bool isFree[nx][ny];
	size_t Ix;
	size_t Iy;

};


#endif /* OCCUPANCYGRID_H_ */
