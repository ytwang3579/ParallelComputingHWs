# CS 4111 Introduction to Parallel Computing <br> HW1: Sokoban Report  
    Name: 王元廷/ ID: 106062119  

## 1. Implementaion  
In this assignment, I implemented a Sokoban solver using C++ with OpenMP, boost and tbb libraries.  
The following is the compile flag that I use:  
```g++ -std=c++2a -O3 -fopenmp -ltbb $in -o $out```

To breifly describe the implementation, I would like to introduce it in serveral points:  
- General concept:  
A modified BFS algorithm to guide the solver to try paths with higher possibilities first.  
Instead of using *queue*, this algorithm uses *priority_queue* as a min-heap to make states with shorter **distance** be poped out first.  
> **distance** is the total sum of the distance between every **goal target** and every **box tile**.   

- Deadlock detection and avoidance: ```checkmapvalid()```  
This function mainly reflects the concept of avoiding so-called deadlock situation to happen, as well as aborting the map if a deadlock should happen.  
- Using ``boost::hash`` and ``std::unordered_set`` to prune duplicate maps.
- Using ``tbb::concurrent_priority_queue`` and ``tbb::concurrent_queue`` to make parallel push and pop possible.

## 2. Difficulties encountered  
There are many problems I have bumped into when doing the assignment. Some of them are solved; yet, others are remained unsolved and thus results in failure in several testcases.  

1. The first problem I faced is to switch my program from the non-parallel one to the parallel approach. I am very unfamiliar to parallel programming and all its related details which require us to be careful during the approach. Hence, *segmantaion faults* often happen in the first few versions of my implementation. Fortunately, after several days of carefully debugging, I have overcome the issue and made the parallel approach really doing its optimization tasks concurrently. The most valuable lesson I gain through debugging is that **all push-and-pop actions that are not thread-safe MUST be in critical sections**; otherwise segmentation faults are likely to happen and the complier will not tell you why and where of your program is wrong.  

1. Apart from the segmentation fault issues, the most challenging (and somehow annoying) problem of all time also take its place in this assignment-- I am also facing algorithmic problems in my implementation, part of which are remained unsolved before the deadline of this homework. The reason why I am sure the problem I encountered is algorithmic is that TA's program could finish all tasks within five seconds (with parallelism assumed), which means without parallelism the program can still finish within 60 seconds (if parallelism boosts the performance perfectly, which is impossible). Take ``samples/31.txt`` as example, my program has to run up to 7 minutes before it can generate the result. Hence, there must be some elegant algorithms that could boost the performance and remain the quality of result at the same time, but it had not yet come to my mind QAQ.  
> Note: The last 4 testcases (31,32,34,35) are so hard that I could not solve it easily by my brain, let alone thinking a useful algorithm to deal with it.  

## 3. Pthread vs OpenMP  
<br> | Strength | Weakness
-----|----------|----------
Pthread | Programmers have full control about how threads would fork/join: **more flexibility**  | Programmers have to do a lot of work just to set up environment for threads
<br>    | Provides **low-level** APIs just like C/C++'s syntax | Since most works are done by programmers, program may have to be modified in order to catch up with the development of technology
OpenMP  | Compiler could handle most of work for programmers, making operations simpler to use: **user-friendly** | can only run in computers with shared memory and requires the compiler to support
<br>    | **Higher level** abstraction: more portable, scalable, easily to maintain and is cross-platform | Would introduce hard-to-find-and-debug synchronization bugs and race conditions <br> Often have lower parallel efficiency compared to Pthread

In this assignment, I choose **OpenMP** to implement because it is much more programmer-friendly. Besides, the number of threads is not specified in the homework description, which also leads me to choose to use OpenMP in my implementation.