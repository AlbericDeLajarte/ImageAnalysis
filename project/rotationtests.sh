#/usr/local/bin/bash
# Rotate video in 15-degree steps
for i in {0..345..15}
do
    echo "Rotation: "$i" degrees."
    python3 ./main.py --input=../data/robot_parcours_1_rotated_$i.mp4 --output=./output_rotated_$i.mp4
done


