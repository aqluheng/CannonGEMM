# CannonGEMM
- Cannon矩阵乘法将矩阵分块后,按块移位,按块运算
- 将矩阵分为sqrt(procNum)块,每一个核心只需保存一块的信息
### 算法流程
1. 将输入矩阵A,B转为可以scatter的形式(一块的内容是紧密连接的)
2. 初始化MPI_Cart_create通信域,让每个进程知道自己处于第几个位置,比如2,3代表第二行第三列的进程,id为2*4+3=11
3. 按照Cannon算法,将A与B进行初次移位,A按行号移位,B按列号移位
4. 循环sqrt(proc)次
   1. pC = pA * pB + pC
   2. A矩阵整体循环左移
   3. B矩阵整体循环上移
### MPI函数
- MPI_Cart_shift(comm, 1, -coord[0], &source, &dest); 
  - 按行整体左移coord[0]
  - 得到source和dest对应的序号值
- MPI_Sendrecv_replace(partialA, partialN * partialN, MPI_FLOAT, dest, 0, source, 0, comm, MPI_STATUS_IGNORE);
  - 根据上一条语句得到的source和dest进行传输
  - source->myId->dest
### 运行方法
make
