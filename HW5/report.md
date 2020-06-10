# CS 4111 Introduction to Parallel Computing <br> HW5: N-body Simulation (CUDA)  

## Name, Student ID
- Name: 王元廷
- Student ID: 106062119  

## 1. What is your parallelism strategy?  
**Reduce communication** is my parallelism strategy in this assignment. Through the process of converting the sequential version into the parallel one, communication overhead are significant since there are totally **200,000** steps for us to simulate and there is no elegant method to parallelized them (The steps must be processed in order). Therefore, I do most calculations inside several kernel functions and only do memory copy at the initialization and the output stage.  

## 2. If there are 4 GPUs instead of 2, what would you do to maximize the performance?
If there are 4 GPUs, I will probably make each GPU deal with one problem. And maybe letting the last GPU simply do the ```run_step``` function. The most important issue is still the communication overhead, the tasks should be partitioned elegantly to lessen the time consumed in ```cudaMemcpy()```. In addition, I would also check the problem size and use less GPUs if the problem is too small, to optimize the performance.  

## 3. If there are 5 gravity devices, is it necessary to simulate 5 n-body simulations independently for this problem?
No, it depends on the distance and the density distribution of the devices. We may apply the Barnes-Hut Algorithm to alleviate the computation intensity.

## 4. Other feedbacks
In this assignment, I have been struggling to reach a parallelism strategy that works well. At first (actually it's until yesterday), my parallelized version of code runs even slower than the sequential version when the dataset is small. For a longer period before the "even slower parallelized version" works, I spent more than one week to debug the logic things. Looking back from now, I conclude that it is because many steps/processes are not independent in this assignment. For example, we have to be very careful when parallelizing the ```run_step``` function. The results can be ruined if we changed any value in *qx*, *qy* or *qz* before every thread has finished calculate the acceleration. In addition, I also ran into stupid mistakes that I use the ```==``` operator to compare two ```char``` arrays. *(And this takes we three days to discover, 'cause I was too confident and focusing on checking the double-precision issue)* The debugging process is more difficult when the program are as complicated that contains two GPUs and one CPU. Hence, I have learned a lesson through this assignment that we should only change one part of code and test whether it works first, or that it could be tiring to debug when something bad happens after you have changed many parts of a program.  