# mobileCV
This repository contains materials for a course on mobile systems for computer vision at ITMO University, St.Petersburg, Russia

### Jetson GPIO  
https://www.jetsonhacks.com/2019/06/07/jetson-nano-gpio/  
https://pypi.org/project/Jetson.GPIO/  
https://forums.developer.nvidia.com/t/jetson-nano-gpio-example-problem/75547/20

### Jetson Camera  
https://github.com/JetsonHacksNano/CSI-Camera

### Jetson time sync issues  
https://github.com/justsoft/jetson-nano-date-sync  
Remove check if year == 2018 in /etc/network/if-up.d/date-sync

### Jetson swap setup  
https://jkjung-avt.github.io/setting-up-nano/  
Swap can also be set with **jtop**

### PyTorch and Torchvision setup  
https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-6-0-now-available/72048  

### Screen resolution fix:  
sudo gedit /etc/X11/xorg.conf  
——————————————————————-  
Section "Screen"  
Identifier "Screen0"  
Monitor    "Monitor0"  
SubSection "Display"  
Viewport   0 0  
Modes "1280x1024"  
Depth   24   
Virtual 1280 1024  
EndSubSection  
EndSection  
———————————————————————   

### Report spec  
Оформляется в README репозитория.
1. Цель работы  
1. Теоретическая база   
1. Описание разработанной системы (алгоритмы, принципы работы, архитектура)  
1. Результаты работы и тестирования системы (скриншоты, изображения, графики, закономерности)  
1. Выводы по работе  
1. Использованные источники  
