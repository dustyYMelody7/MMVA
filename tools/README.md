# 工具文件：

## create_feature.py:

该文件使用来创建人脸数据库，其仅有一个参数需要设定，就是你需要识别的人脸路径，其文件格式如下：

dir:
        name1:
            picture1.jpg
            picture2.jpg
            picture3.jpg
            ...
        name2:
            picture1.jpg
            picture2.jpg
            picture3.jpg
            ...
        name3:
            picture1.jpg
            picture2.jpg
            picture3.jpg
            ...

即路径下以名称名命，每个名称文件夹中包含人脸图片多张（>=1），然后运行(**运行前确保你的数据库中已经创建了face_feature表，若未创建请参考{PROJECT}/README.md文件中环境搭建部分的指示操作！**)：

> python tools/create_feature.py --image-dir IMAGE_DIR

## make_video_table.py:

该文件是用来创建video数据表格的，考录到部分国内视频可能用中文名命，但是某些linux机器\python库可能对中文支持较差，所以转为唯一的id与视频名对应，所以才多此一举的将视频编码为一串唯一的id。

该文件可以自动将视频目录的全部视频文件（无论格式，无论路径深浅）提取并转为唯一id，你只需要提供一个最顶层的视频目录即可

**运行前确保你的数据库中已经创建了video,video_docker表，若未创建请参考{PROJECT}/README.md文件中环境搭建部分的指示操作！**

>python tools/make_video_table.py --video_dir VIDEO_DIR

## write_to_mysql.py:

这个文件是用来将人脸数据（*.npy文件）写入mysql数据库中的，原因是由于create_feature.py文件有时创建的人脸特征不是特别准，使用这个文件之前你需要下载[insightface_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)这个第三方项目，然后按照他的步骤创建好facebank.npy和names.npy文件，然后修改这个代码中相关的目录即可运行。

由于本项目并未使用insightface_Pytorch这个的mtcnn导致人脸数据库用上述第一个文件创建的可能不是特别准，若追求精确识别人脸的用户请自行了解如何使用。

> python tools/write_to_mysql.py