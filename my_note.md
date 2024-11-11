### check webcam
ffmpeg -t 6 -f video4linux2 -i /dev/video0 /media/frigate/test.mp4