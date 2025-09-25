# Mesh Interpolation in High Performance Computing (HPC)

## 📌 Project Overview
This project implements **scattered point-to-mesh interpolation** on a structured grid using **C** and **OpenMP** for parallelization.  
The goal is to efficiently map millions of scattered 2D points onto a mesh grid, distributing their values using bilinear interpolation.  
The implementation is designed to scale on **multi-core HPC systems**, achieving significant speedups over a serial baseline.

---

## ⚡ Key Features
- **Parallel Implementation:** Uses OpenMP with thread-private buffers and manual reduction to avoid race conditions.
- **Optimized Memory Access:** Dynamic scheduling improves load balancing for uneven point distributions.
- **Scalable Performance:** Successfully tested on datasets with **20M+ points** across multiple cores.
- **Benchmarking & Profiling:** Execution time logged automatically; analyzed impact of scheduling policies and hyperthreading.

---

## 🛠️ Tech Stack
- **Language:** C  
- **Parallelization:** OpenMP  
- **Environment:** HPC cluster & lab machines  
- **Tools:** `valgrind/callgrind` for profiling, CSV-based performance logging  

---

## 📊 Performance Results
| Cores | Speedup (up to 20M points) |
|------|----------------------------|
| 1    | 1× (baseline)             |
| 2    | ~1.8×                     |
| 4    | ~3.3×                     |
| 8    | **4.6×**                  |

- **Dynamic scheduling** consistently outperformed static scheduling.
- **Thread-private buffers** significantly reduced false sharing and improved cache efficiency.

