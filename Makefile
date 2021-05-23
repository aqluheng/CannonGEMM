gemm : gemm.o cblas_LINUX.a blas_LINUX.a
		mpifort -o gemm gemm.o cblas_LINUX.a blas_LINUX.a
		mpirun -np 16 ./gemm

gemm.o : gemm.c
		mpicc -c gemm.c

clean : 
		rm gemm gemm.o