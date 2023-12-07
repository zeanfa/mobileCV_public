for o in 0 1 2 3; do 
	echo '##### ->' $o;
	g++ -O$o rgb_to_gray.cc -o rgb_to_gray_$o -I "/usr/include/opencv4" -L /usr/lib/aarch64-linux-gnu -lopencv_core -lopencv_highgui -lopencv_imgcodecs
	for gr_weight in 50 100 150 200 250; do 
		for i in $(seq 1 5); do 
			./rgb_to_gray_$o ../img/Lenna.png $gr_weight
		done; 
	done;
	mkdir ./img_$o
	mv *.png ./img_$o/
	rm rgb_to_gray_$o
done

