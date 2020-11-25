## To render depthmaps from different viewpoints
Download blender 2.79b from https://download.blender.org/release/Blender2.79/ and add it to environment path .



Run the following command to render depthmaps with resolution res for category with ID cid
```
python create_dataset.py --shapenet_path ${ShapeNetCore.v2_path} --output_path ${save_directory} --resolution ${res} --categories ${cid}
```
Eg: To render depthmaps for chair category with resolution 64
```
python create_dataset.py --shapenet_path ../data/ShapeNetCore.v2 --output_path ./depthmaps --resolution 64 --categories 03001627
```


