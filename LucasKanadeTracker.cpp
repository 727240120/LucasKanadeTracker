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
	void LowpassFilter(const Mat &image, Mat &result);//��ͨ�˲�
	uchar get_value(Mat const &image, Point2f index);//��ȡ����
	uchar get_subpixel_value(Mat const &image, Point2f index);//���Բ�ֵ
	void BuildPyramid(Mat const &input, vector <Mat> &outputArray, int maxLevel);//����ͼ�������
	int LucasKanade(vector <Mat> &prevImage, vector <Mat> &nextImage,Mat &prevPoint, Mat &nextPoint);//LK
};
void LucasKanadeTracker::LowpassFilter(const Mat &image, Mat &result)
{
	Mat kernel(3, 3, CV_32F, Scalar(0));//3*3�����

	kernel.at <float>(0, 0) = kernel.at <float>(2, 0) = 1.0f / 16.0f;
	kernel.at <float>(0, 2) = kernel.at <float>(2, 2) = 1.0f / 16.0f;
	kernel.at <float>(1, 0) = kernel.at <float>(1, 2) = 1.0f / 8.0f;
	kernel.at <float>(0, 1) = kernel.at <float>(2, 1) = 1.0f / 8.0f;
	kernel.at <float>(1, 1) = 1.0f / 4.0f;

	filter2D(image, result, image.depth(), kernel);
}

uchar LucasKanadeTracker::get_value(Mat const &image, Point2f index)//��Խ���ȡ
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

void LucasKanadeTracker::BuildPyramid(Mat const &input, vector <Mat> &outputArray, int maxLevel)//�߲��һ�����ر�ʾ�Ͳ����������
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

				//���
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

		// ������δ��ڷ�Χ
		float indexXLeft = currPoint.at <float>(0, 0) - omegaX;
		float indexYLeft = currPoint.at <float>(1, 0) - omegaY;

		float indexXRight = currPoint.at <float>(0, 0) + omegaX;
		float indexYRight = currPoint.at <float>(1, 0) + omegaY;

		//�����ݶȾ���
		Mat gradient(2, 2, CV_32F, Scalar(0));
		
		vector <Point2f> derivatives;

		for (float i = indexXLeft; i <= indexXRight; i += 1.0f)
		{
			for (float j = indexYLeft; j <= indexYRight; j += 1.0f)
			{
				//IL��X��ƫ����
				float derivativeX = (get_subpixel_value(prevImage.at(level), Point2f(i + 1.0f, j))
					- get_subpixel_value(prevImage.at(level), Point2f(i - 1.0f, j))) / 2.0f;

				//IL��Y��ƫ����
				float derivativeY = (get_subpixel_value(prevImage.at(level), Point2f(i, j + 1.0f))
					- get_subpixel_value(prevImage.at(level), Point2f(i, j - 1.0f))) / 2.0f;

				derivatives.push_back(Point2f(derivativeX, derivativeY));
				//�����ݶȾ���
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
			//ͼ��ƥ������
			Mat imageMismatch(2, 1, CV_32F, Scalar(0));
			for (float i = indexXLeft; i <= indexXRight; i += 1.0f)
			{
				for (float j = indexYLeft; j <= indexYRight; j += 1.0f)
				{
					float nextIndexX = i + piramidalGuess.at <float>(0, 0) + opticalFlow.at <float>(0, 0);
					float nextIndexY = j + piramidalGuess.at <float>(1, 0) + opticalFlow.at <float>(1, 0);
					//ͼ�����ز�
					int pixelDifference = (int)(get_subpixel_value(prevImage.at(level), Point2f(i, j))
						- get_subpixel_value(nextImage.at(level), Point2f(nextIndexX, nextIndexY)));
					//ͼ��ƥ������
					imageMismatch.at <float>(0, 0) += pixelDifference * derivatives.at(cnt).x;
					imageMismatch.at <float>(1, 0) += pixelDifference * derivatives.at(cnt).y;

					cnt++;
				}
			}
			//����ʸ��
			opticalFlow += gradient * imageMismatch;
		}

		if (level == 0)     opticalFlowFinal = opticalFlow;
		else
			piramidalGuess = 2 * (piramidalGuess + opticalFlow);

	}
	//���Ĺ���ʸ��
	opticalFlowFinal += piramidalGuess;
	//����һ֡ͼ���ϱ����ٵ�������
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


//�����Ӧ�¼�
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
	//���ļ���ȡ����
	VideoCapture capture("video.avi");

	if (!capture.isOpened())//�ļ�����
	{
		cout << "Cannot open the file" << endl;
		return -1;
	}

	//��ȡ��Ƶ֡��
	double rate = capture.get(CV_CAP_PROP_FPS);
	bool stop = false;

	Mat frame, gray, grayPrev, prevFrame;
	namedWindow("LucasKanade");
	int delay = 1000 / rate;

	//��ȡ�����λ��
	setMouseCallback("LucasKanade", onMouse);

	capture.read(prevFrame);
	cvtColor(prevFrame, grayPrev, CV_RGB2GRAY);
	//��δ���ó�ʼ������λ�ã�ͣ���ڵ�һ֡����
	while (!isStartSet)
	{
		imshow("LucasKanade", prevFrame);
		waitKey(30);
	}

	while (!stop)
	{
		//�洢��һ֡�뵱ǰ��ͼ�������
		vector <Mat> output1;
		vector <Mat> output2;
		//�������һ֮֡������ѭ��
		if (!capture.read(frame)) break;
		//RGB->Gray
		cvtColor(frame, gray, CV_RGB2GRAY);
		//�ֱ����һ֡�Լ���ǰ֡����ͼ�������
		lk.BuildPyramid(grayPrev, output1, 4);
		lk.BuildPyramid(gray, output2, 4);
		//����LK����
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