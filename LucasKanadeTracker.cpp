#include <vector>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <opencv2\imgproc\types_c.h>  
using namespace std;
using namespace cv;
Mat start(2, 1, CV_32F, Scalar(0));
Mat finish(2, 1, CV_32F, Scalar(0));
bool isStartSet = 0;
class LucasKanadeTracker
{
public:
	void LowpassFilter(const Mat &image, Mat &result);//低通滤波
	uchar get_value(Mat const &image, Point2f index);//读取像素
	uchar get_subpixel_value(Mat const &image, Point2f index);//线性插值
	void BuildPyramid(Mat const &input, vector <Mat> &outputArray, int maxLevel);//建立图像金字塔
	int LucasKanade(vector <Mat> &prevImage, vector <Mat> &nextImage,Mat &prevPoint, Mat &nextPoint);//LK
};
void LucasKanadeTracker::LowpassFilter(const Mat &image, Mat &result)
{
	Mat kernel(3, 3, CV_32F, Scalar(0));//3*3卷积核

	kernel.at <float>(0, 0) = kernel.at <float>(2, 0) = 1.0f / 16.0f;
	kernel.at <float>(0, 2) = kernel.at <float>(2, 2) = 1.0f / 16.0f;
	kernel.at <float>(1, 0) = kernel.at <float>(1, 2) = 1.0f / 8.0f;
	kernel.at <float>(0, 1) = kernel.at <float>(2, 1) = 1.0f / 8.0f;
	kernel.at <float>(1, 1) = 1.0f / 4.0f;

	filter2D(image, result, image.depth(), kernel);
}

uchar LucasKanadeTracker::get_value(Mat const &image, Point2f index)//不越界读取
{
	if (index.x >= image.rows)   index.x = image.rows - 1.0f;
	else if (index.x < 0)         index.x = .0f;

	if (index.y >= image.cols)    index.y = image.cols - 1.0f;
	else if (index.y < 0)         index.y = .0f;

	return image.at <uchar>(index.x, index.y);
}

uchar LucasKanadeTracker::get_subpixel_value(Mat const &image, Point2f index)
{
	float floorX = (float)floor(index.x);
	float floorY = (float)floor(index.y);

	float fractX = index.x - floorX;
	float fractY = index.y - floorY;

	return ((1.0f - fractX) * (1.0f - fractY) * get_value(image, Point2f(floorX, floorY))
		+ (fractX * (1.0f - fractY) * get_value(image, Point2f(floorX + 1.0f, floorY)))
		+ ((1.0f - fractX) * fractY * get_value(image, Point2f(floorX, floorY + 1.0f)))
		+ (fractX * fractY * get_value(image, Point2f(floorX + 1.0f, floorY + 1.0f))));
}

void LucasKanadeTracker::BuildPyramid(Mat const &input, vector <Mat> &outputArray, int maxLevel)//高层的一个像素表示低层的两个像素
{
	outputArray.push_back(input);
	for (int k = 1; k <= maxLevel; ++k)
	{
		Mat prevImage;
		LowpassFilter(outputArray.at(k - 1), prevImage);

		int limRows = (prevImage.rows + 1) / 2;
		int limCols = (prevImage.cols + 1) / 2;

		Mat currMat(limRows, limCols, CV_8UC1, Scalar(0));

		for (int i = 0; i < limRows; ++i)
		{
			for (int j = 0; j < limCols; ++j)
			{
				
				float indexX = 2 * i;
				float indexY = 2 * j;

				//卷积
				float firstSum = (get_value(prevImage, Point2f(indexX, indexY))) / 4.0f;

				float secondSum = .0f;
				secondSum += get_value(prevImage, Point2f(indexX - 1.0f, indexY));
				secondSum += get_value(prevImage, Point2f(indexX + 1.0f, indexY));
				secondSum += get_value(prevImage, Point2f(indexX, indexY - 1.0f));
				secondSum += get_value(prevImage, Point2f(indexX, indexY + 1.0f));
				secondSum /= 8.0f;

				float thirdSum = .0f;
				thirdSum += get_value(prevImage, Point2f(indexX - 1.0f, indexY - 1.0f));
				thirdSum += get_value(prevImage, Point2f(indexX + 1.0f, indexY - 1.0f));
				thirdSum += get_value(prevImage, Point2f(indexX - 1.0f, indexY + 1.0f));
				thirdSum += get_value(prevImage, Point2f(indexX + 1.0f, indexY + 1.0f));
				thirdSum /= 16.0f;

				currMat.at <uchar>(i, j) = firstSum + secondSum + thirdSum;
			}
		}
		outputArray.push_back(currMat);
	}
}

int LucasKanadeTracker::LucasKanade(vector <Mat> &prevImage, vector <Mat> &nextImage,
	Mat &prevPoint, Mat &nextPoint)
{
	Mat piramidalGuess(2, 1, CV_32F, Scalar(0));
	Mat opticalFlowFinal(2, 1, CV_32F, Scalar(0));

	for (int level = prevImage.size() - 1; level >= 0; --level)
	{
		Mat currPoint(2, 1, CV_32F, Scalar(0));
		currPoint.at <float>(0, 0) = prevPoint.at <float>(0, 0) / pow(2, level);
		currPoint.at <float>(1, 0) = prevPoint.at <float>(1, 0) / pow(2, level);

		int omegaX = 7;
		int omegaY = 7;

		// 定义矩形窗口范围
		float indexXLeft = currPoint.at <float>(0, 0) - omegaX;
		float indexYLeft = currPoint.at <float>(1, 0) - omegaY;

		float indexXRight = currPoint.at <float>(0, 0) + omegaX;
		float indexYRight = currPoint.at <float>(1, 0) + omegaY;

		//定义梯度矩阵
		Mat gradient(2, 2, CV_32F, Scalar(0));
		
		vector <Point2f> derivatives;

		for (float i = indexXLeft; i <= indexXRight; i += 1.0f)
		{
			for (float j = indexYLeft; j <= indexYRight; j += 1.0f)
			{
				//IL对X求偏导数
				float derivativeX = (get_subpixel_value(prevImage.at(level), Point2f(i + 1.0f, j))
					- get_subpixel_value(prevImage.at(level), Point2f(i - 1.0f, j))) / 2.0f;

				//IL对Y求偏导数
				float derivativeY = (get_subpixel_value(prevImage.at(level), Point2f(i, j + 1.0f))
					- get_subpixel_value(prevImage.at(level), Point2f(i, j - 1.0f))) / 2.0f;

				derivatives.push_back(Point2f(derivativeX, derivativeY));
				//计算梯度矩阵
				gradient.at <float>(0, 0) += derivativeX * derivativeX;
				gradient.at <float>(0, 1) += derivativeX * derivativeY;
				gradient.at <float>(1, 0) += derivativeX * derivativeY;
				gradient.at <float>(1, 1) += derivativeY * derivativeY;
			}
		}

		gradient = gradient.inv();

		int maxCount = 3;
		Mat opticalFlow(2, 1, CV_32F, Scalar(0));
		for (int k = 0; k < maxCount; ++k)
		{
			int cnt = 0;
			//图像不匹配向量
			Mat imageMismatch(2, 1, CV_32F, Scalar(0));
			for (float i = indexXLeft; i <= indexXRight; i += 1.0f)
			{
				for (float j = indexYLeft; j <= indexYRight; j += 1.0f)
				{
					float nextIndexX = i + piramidalGuess.at <float>(0, 0) + opticalFlow.at <float>(0, 0);
					float nextIndexY = j + piramidalGuess.at <float>(1, 0) + opticalFlow.at <float>(1, 0);
					//图像像素差
					int pixelDifference = (int)(get_subpixel_value(prevImage.at(level), Point2f(i, j))
						- get_subpixel_value(nextImage.at(level), Point2f(nextIndexX, nextIndexY)));
					//图像不匹配向量
					imageMismatch.at <float>(0, 0) += pixelDifference * derivatives.at(cnt).x;
					imageMismatch.at <float>(1, 0) += pixelDifference * derivatives.at(cnt).y;

					cnt++;
				}
			}
			//光流矢量
			opticalFlow += gradient * imageMismatch;
		}

		if (level == 0)     opticalFlowFinal = opticalFlow;
		else
			piramidalGuess = 2 * (piramidalGuess + opticalFlow);

	}
	//最后的光流矢量
	opticalFlowFinal += piramidalGuess;
	//在下一帧图像上被跟踪的特征点
	nextPoint = prevPoint + opticalFlowFinal;

	if ((nextPoint.at <float>(0, 0) < 0) || (nextPoint.at <float>(1, 0) < 0) ||
		(nextPoint.at <float>(0, 0) >= prevImage.at(0).rows) ||
		(nextPoint.at <float>(1, 0) >= prevImage.at(0).cols))
	{
		cout << "Object is lost" << endl;
		return 0;
	}
	return 1;
}


//鼠标响应事件
static void onMouse(int event, int x, int y, int, void* ptr)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	start.at <float>(0, 0) = (float)y;
	start.at <float>(1, 0) = (float)x;

	isStartSet = 1;
	cout << "Start" << endl;
}

int main()
{
	LucasKanadeTracker lk;
	//打开文件读取数据
	VideoCapture capture("video.avi");

	if (!capture.isOpened())//文件存在
	{
		cout << "Cannot open the file" << endl;
		return -1;
	}

	//获取视频帧率
	double rate = capture.get(CV_CAP_PROP_FPS);
	bool stop = false;

	Mat frame, gray, grayPrev, prevFrame;
	namedWindow("LucasKanade");
	int delay = 1000 / rate;

	//获取鼠标点击位置
	setMouseCallback("LucasKanade", onMouse);

	capture.read(prevFrame);
	cvtColor(prevFrame, grayPrev, CV_RGB2GRAY);
	//若未设置初始特征点位置，停留在第一帧画面
	while (!isStartSet)
	{
		imshow("LucasKanade", prevFrame);
		waitKey(30);
	}

	while (!stop)
	{
		//存储上一帧与当前的图像金字塔
		vector <Mat> output1;
		vector <Mat> output2;
		//读到最后一帧之后，跳出循环
		if (!capture.read(frame)) break;
		//RGB->Gray
		cvtColor(frame, gray, CV_RGB2GRAY);
		//分别对上一帧以及当前帧建立图像金字塔
		lk.BuildPyramid(grayPrev, output1, 4);
		lk.BuildPyramid(gray, output2, 4);
		//进行LK跟踪
		lk.LucasKanade(output1, output2, start, finish);
		circle(frame, Point((int)finish.at <float>(1, 0), (int)finish.at <float>(0, 0)), 9, Scalar(0, 0, 255));
		imshow("LucasKanade", frame);

		gray.copyTo(grayPrev);
		finish.copyTo(start);

		if (waitKey(delay) >= 0) stop = true;
	}
	capture.release();
	waitKey();

	return 0;
}