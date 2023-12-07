#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

void rgb_with_graymask(const uint8_t* rgb, uint8_t* reduced, int num_pixels, int gr_weight)
{
	auto t1 = chrono::high_resolution_clock::now();
    gr_weight = gr_weight % 256;
    int img_weight = 255 - gr_weight;

	for(int i=0; i<num_pixels; i+=3, rgb+=3) {
		int v = (rgb[0] + rgb[1] + rgb[2])/3;
        reduced[i] = (img_weight*rgb[0] + gr_weight*v) >> 8;
        reduced[i+1] = (img_weight*rgb[1] + gr_weight*v) >> 8;
        reduced[i+2] = (img_weight*rgb[2] + gr_weight*v) >> 8; 
	}
	auto t2 = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
	cout << "rgb_with_graymask\t" << to_string(gr_weight) << "\t" << duration << endl;
}

void rgb_with_graymask_neon8(const uint8_t* rgb, uint8_t* reduced, int num_pixels, int gr_weight) {
    gr_weight = gr_weight % 256;
    int img_weight = 255 - gr_weight;
	num_pixels /= 8;
	uint8x8_t gr_w = vdup_n_u8(gr_weight);
	uint8x8_t img_w = vdup_n_u8(img_weight);
	uint8x8_t w_gr = vdup_n_u8(85);

	uint16x8_t temp;
	uint8x8_t temp_result;
	auto t1_neon = chrono::high_resolution_clock::now();
	for(int i=0; i<num_pixels; i+=3, rgb+=8*3, reduced+=8*3) {
        uint8x8x3_t src = vld3_u8(rgb);
        temp = vmull_u8(src.val[0], w_gr);
        temp = vmlal_u8(temp, src.val[1], w_gr);
        temp = vmlal_u8(temp, src.val[2], w_gr);
        temp_result = vshrn_n_u16(temp, 8);

        src.val[0] = vshrn_n_u16(
                vmull_u8(src.val[0], img_w) + vmull_u8(temp_result, gr_w), 8); 
        src.val[1] = vshrn_n_u16(
                vmull_u8(src.val[1], img_w) + vmull_u8(temp_result, gr_w), 8); 
        src.val[2] = vshrn_n_u16(
                vmull_u8(src.val[2], img_w) + vmull_u8(temp_result, gr_w), 8); 

        vst3_u8(reduced, src);
	}
	auto t2_neon = chrono::high_resolution_clock::now();
	auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
	cout << "rgb_with_graymask_neon8\t" << to_string(gr_weight) << "\t" << duration_neon << endl;
}

int main(int argc,char** argv)
{
	uint8_t * rgb_arr;
	uint8_t * rgb_graymask;
	uint8_t * rgb_graymask_neon;

	if (argc != 3) {
		cout << "Usage: opencv_neon image_name grayscale_weight" << endl;
		return -1;
	}

	Mat rgb_image;
	rgb_image = imread(argv[1], IMREAD_COLOR);
    int gr_weight = atoi(argv[2]);

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
	int num_pixels_RGB = width*height*3;

	Mat rgb_graymask_mat_neon(height, width, CV_8UC3, Scalar(0, 0, 0));
	rgb_graymask_neon = rgb_graymask_mat_neon.data;
    rgb_with_graymask_neon8(rgb_arr, rgb_graymask_neon, num_pixels_RGB, gr_weight);
	imwrite("graymask_img_neon_" + to_string(gr_weight) + ".png", rgb_graymask_mat_neon);


	Mat rgb_graymask_mat(height, width, CV_8UC3, Scalar(0, 0, 0));
    rgb_graymask = rgb_graymask_mat.data;
	rgb_with_graymask(rgb_arr, rgb_graymask, num_pixels_RGB, gr_weight);
	imwrite("graymask_img_" + to_string(gr_weight) + ".png", rgb_graymask_mat);

    return 0;
}
