# Skoffer comparison between Clustering, SingleThread mode of PKMeans and MultiThreadPKMeans

```julia
versioninfo()

Julia Version 1.3.1
Commit 2d5741174c (2019-12-30 21:36 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, skylake)
Environment:
  JULIA_EDITOR = atom  -a
  JULIA_NUM_THREADS = 4
```

For `X = rand(60, 1_000_000); tol = 1e-6` output of `TimerOutputs`

```
Time                   Allocations      
──────────────────────   ───────────────────────
Tot / % measured:            1541s / 85.5%           19.5GiB / 99.4%    

Section                ncalls     time   %tot     avg     alloc   %tot      avg
───────────────────────────────────────────────────────────────────────────────
Clustering                  1     662s  50.2%    662s   18.6GiB  96.1%  18.6GiB
10 clusters               1    92.6s  7.03%   92.6s   2.35GiB  12.1%  2.35GiB
9 clusters                1    89.7s  6.81%   89.7s   2.34GiB  12.1%  2.34GiB
8 clusters                1    87.1s  6.62%   87.1s   2.33GiB  12.0%  2.33GiB
7 clusters                1    85.3s  6.48%   85.3s   2.32GiB  12.0%  2.32GiB
6 clusters                1    80.6s  6.12%   80.6s   2.32GiB  12.0%  2.32GiB
5 clusters                1    78.3s  5.95%   78.3s   2.31GiB  11.9%  2.31GiB
4 clusters                1    76.6s  5.82%   76.6s   2.30GiB  11.9%  2.30GiB
3 clusters                1    50.3s  3.82%   50.3s   1.58GiB  8.16%  1.58GiB
2 clusters                1    20.9s  1.59%   20.9s    732MiB  3.69%   732MiB
PKMeans Singlethread        2     491s  37.3%    245s    208MiB  1.05%   104MiB
9 clusters                1     131s  10.0%    131s   22.9MiB  0.12%  22.9MiB
10 clusters               1    89.5s  6.80%   89.5s   22.9MiB  0.12%  22.9MiB
7 clusters                1    77.3s  5.87%   77.3s   22.9MiB  0.12%  22.9MiB
8 clusters                1    59.4s  4.51%   59.4s   22.9MiB  0.12%  22.9MiB
6 clusters                1    44.1s  3.35%   44.1s   22.9MiB  0.12%  22.9MiB
5 clusters                1    35.1s  2.67%   35.1s   22.9MiB  0.12%  22.9MiB
4 clusters                1    32.9s  2.50%   32.9s   22.9MiB  0.12%  22.9MiB
3 clusters                1    14.6s  1.11%   14.6s   22.9MiB  0.12%  22.9MiB
2 clusters                2    6.52s  0.50%   3.26s   23.3MiB  0.12%  11.7MiB
PKMeans Multithread         1     165s  12.5%    165s    575MiB  2.90%   575MiB
9 clusters                1    37.2s  2.82%   37.2s   40.1MiB  0.20%  40.1MiB
8 clusters                1    33.1s  2.51%   33.1s   23.9MiB  0.12%  23.9MiB
10 clusters               1    25.8s  1.96%   25.8s   24.0MiB  0.12%  24.0MiB
6 clusters                1    20.9s  1.59%   20.9s   23.6MiB  0.12%  23.6MiB
7 clusters                1    16.4s  1.25%   16.4s   23.4MiB  0.12%  23.4MiB
5 clusters                1    13.1s  1.00%   13.1s   23.4MiB  0.12%  23.4MiB
4 clusters                1    9.90s  0.75%   9.90s   23.4MiB  0.12%  23.4MiB
3 clusters                1    4.97s  0.38%   4.97s    370MiB  1.87%   370MiB
2 clusters                1    3.26s  0.25%   3.26s   23.2MiB  0.12%  23.2MiB
───────────────────────────────────────────────────────────────────────────────
```
