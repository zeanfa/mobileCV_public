/*  For description look into the help() function. */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

void rgb_to_gray(const uint8_t* rgb, uint8_t* gray, int num_pixels)
{
	cout << "inside function rgb_to_gray" << endl;
	auto t1 = chrono::high_resolution_clock::now();
	for(int i=0; i<num_pixels; ++i, rgb+=3) {
		int v = (77*rgb[0] + 150*rgb[1] + 29*rgb[2]);
		gray[i] = v>>8;
	}
	auto t2 = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
	cout << duration << " us" << endl;
}

void rgb_to_gray_neon(const uint8_t* rgb, uint8_t* gray, int num_pixels) {
	// We'll use 64-bit NEON registers to process 8 pixels in parallel.
	num_pixels /= 8;
	// Duplicate the weight 8 times.
	uint8x8_t w_r = vdup_n_u8(77);
	uint8x8_t w_g = vdup_n_u8(150);
	uint8x8_t w_b = vdup_n_u8(29);
	// For intermediate results. 16-bit/pixel to avoid overflow.
	uint16x8_t temp;
	// For the converted grayscale values.
	uint8x8_t result;
	auto t1_neon = chrono::high_resolution_clock::now();
	for(int i=0; i<num_pixels; ++i, rgb+=8*3, gray+=8) {
	// Load 8 pixels into 3 64-bit registers, split by channel.
	uint8x8x3_t src = vld3_u8(rgb);
	// Multiply all eight red pixels by the corresponding weights.
	temp = vmull_u8(src.val[0], w_r);
	// Combined multiply and addition.
	temp = vmlal_u8(temp, src.val[1], w_g);
	temp = vmlal_u8(temp, src.val[2], w_b);
	// Shift right by 8, "narrow" to 8-bits (recall temp is 16-bit).
	result = vshrn_n_u16(temp, 8);
	// Store converted pixels in the output grayscale image.
	vst1_u8(gray, result);
	}

	auto t2_neon = chrono::high_resolution_clock::now();
	auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
	cout << "inside function rgb_to_gray_neon" << endl;
	cout << duration_neon << " us" << endl;
}

int main(int argc,char** argv)
{
	uint8_t * rgb_arr;
	uint8_t * gray_arr_neon;

	if (argc != 2) {
		cout << "Usage: opencv_neon image_name" << endl;
		return -1;
	}

	Mat rgb_image;
	rgb_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!rgb_image.data) {
		cout << "Could not open the image" << endl;
		return -1;
	}
	if (rgb_image.isContinuous()) {
		rgb_arr = rgb_image.data;
	}
	else {
		cout << "data is not continuous" << endl;
		return -2;
	}

	int width = rgb_image.cols;
	int height = rgb_image.rows;
	int num_pixels = width*height;
	Mat gray_image_neon(height, width, CV_8UC1, Scalar(0));
	gray_arr_neon = gray_image_neon.data;


	auto t1_neon = chrono::high_resolution_clock::now();
	rgb_to_gray_neon(rgb_arr, gray_arr_neon, num_pixels);
	auto t2_neon = chrono::high_resolution_clock::now();
	auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
	cout << "rgb_to_gray_neon" << endl;
	cout << duration_neon << " us" << endl;

	imwrite("gray_neon.png", gray_image_neon);

    return 0;
}
