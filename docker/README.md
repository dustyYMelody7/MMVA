# docker documentation

本项目支持采用docker的方式运行，且docker版本采用完全分布式的解决方案（极为先进！）。缺点是占用内存较高，且调试麻烦，日志文件不集中等；优点是便于移植，在大型服务上运行效果更好等。

## 目录

+ [docker documentation](#docker documentation)
+ [目录](#目录)
+ [结构](#结构)
+ [howto](#howto)
  + [获取需要的docker image](#获取需要的docker image)
  + [获取mysql数据](#获取mysql数据)
  + [启动](#启动)
+ [其他介绍](#其他介绍)
  + [redis](#redis)
  + [mysql](#mysql)
  + [docker-compose](#docker-compose)

## 结构：

![docker框架图](../data/docker-structure.jpg)

项目结构如图所示，总框架由8个独立的docker分布式运行。

+ 顶层为video处理（即video读取和转存）进程，客户端只需post请求访问host的9933端口（赋予mysql内video_filter.video表中的id即可），用于供客户端调用；

+ 第二层有两个存储docker，一个是redis用于转存实时数据（动态数据），另一个是mysql用于存储静态数据；

+ 第三层为识别层docker，用于识别进程。

mysql暴露的端口是为客户端读取静态数据准备的。

## howto

**docker,nvidia-container安装：[nvidia-container](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**

### 获取需要的docker image

> docker pull mysql:latest
>
> docker pull redis:latest

生成运行环境需要的image：

> cd ${PWD}/recognition
>
> docker build . -t video_filter:kalenforn
>
> cd ${PWD}/videoProcess
>
> docker build . -t video_process:kalenforn

生成好以后你可以测试运行以下这两个image有没有输出

>docker run -d --name test video_filter:kalenforn
>
>docker logs -f test
>
>会有如下输出
>
>This is an example to tell you how to work of this docker.
>Please volume your code in /usr/local/src/project.
>In the /usr/local/src/project directory we need a main.py file as the main program.
>Our work dir is /usr/local/src/project.
>Example:
>cd /path/to/your/code
>docker run --name test -d -v ${PWD}:/usr/local/src/project video-filter

### 获取mysql数据

如果你有权能够获取[项目说明](../README.md)中生成的磁盘数据库文件，则直接将其拷贝到${PWD}/run/mysql/store中，然后运行```docker-compose up -d```即可开启（后续可不用查看）

*如果你不想自己建数据库也行，你可以使用我提供的/PROJECT/docker/run/mysql/data.sql文件（只有表结构没有实际内容），但是你需要自行将video，face_feature表完成，请自行更改代码*

如果你无权获取这个磁盘数据，则使用mysqldump将数据库保存为sql文件，然后挂载即可。

> mysqldump -u root -p video_filter > ${PWD}/run/mysql/data.sql
>
> vim ${PWD}/run/mysql/data.sql (首行添加)
>
> ``` USE video_filter```

然后，你需要修改${PWD}/run/docker-compose.yml文件中的一些地方：

+ >entrypoint: ["sh", "-c", "sleep 10 && python main.py"] (所有的这一行改为)
  >
  >entrypoint: ["sh", "-c", "sleep 60 && python main.py"]\

+ > \# - ./mysql/data.sql:/docker-entrypoint-initdb.d/data.sql（去掉注释）
  >
  > \- ./mysql/data.sql:/docker-entrypoint-initdb.d/data.sql

这些修改的位置仅在**第一次启动**时需要修改，再次启动时，请改回最初版本，因为mysql数据文件可以直接挂载到容器内进行初始化。

### 启动

> cd ${PWD}/run
>
> docker-compose up -d
>
> 等1分钟
>
> docker ps
>
> 查看8个docker是否全部处于启动正常状态，否则自行查看log排查问题
>
> ***请一定要修改common.cfg文件中的code_dir路径！！！***

## 其他介绍

### redis

redis中的存储方式为分赛道存储的模式，即对于5个detect-docker，分别对应不同的database进行存储相同数据。

该方式实现最为简单。

***2021-05-17暂时想不到其他解决方案能处理多线/进程中的：单生产者--多消费者--单一共享存储空间 情况下数据使用完的擦除问题***
***2021-05-19想到了解决方案，采用类似流水线的方式减少荣誉数据的保存，即识别进程按流程运行，但仍为异步处理，每个进程拥有两个RedisHold，一个用来读一个用来写，处于流水线最后的进程只有一个RedisHold用于丢弃数据***，[解决方案描述](./multiple-consumers.md)

### mysql

mysql的挂载在不同时候是不太一样的，如果你能获取[项目说明](../README.md)中数据库磁盘存储的结果，可以直接将其复制到```cp mysql/data/dir/* ${PWD}/run/mysql/store```中。

如果你只能通过mysqldump的方式获取到video_filter数据库，则在**第一次**运行docker-compose时将docker-compose中的如下行修改

> \# - ./mysql/data.sql:/docker-entrypoint-initdb.d/data.sql（去掉注释）
>
> \- ./mysql/data.sql:/docker-entrypoint-initdb.d/data.sql

确保你将video_filter数据库保存到了```${PWD}/run/mysql/```中，并且**第一次**运行docker-compose时请把所有的sleep值设置为60s，如下：

> entrypoint: ["sh", "-c", "sleep 10 && python main.py"] (所有的这一行改为)
>
> entrypoint: ["sh", "-c", "sleep 60 && python main.py"]

### docker-compose

- [x] 网络支持：redis和mysql是网络独立，其他container均可互通
- [x] gpu支持：上述结构图中，上两层结构未使用gpu，其他容器均使用gpu
- [x] 暴露端口：如上图所示，整个项目对外只暴露videoProcess的9933端口和mysql的3306端口

## 问题记录
docker内出现python3 print不出中文，此时是因为docker image默认未使用中文编码，在环境变量中添加：
> export LC_ALL=C.UTF-8
>
> export LANG=C.UTF-8

# docker所产生的日志文件

docker所产生的日志文件均放在了/PROJECT/log-docker下，输出为/PROJECT/output-docker

