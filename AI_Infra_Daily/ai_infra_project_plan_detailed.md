# AI Infra Bottom-Level Project Plan (Detailed Daily Version)

## Overview
Two deep projects:
1. FlashAttention Triton Implementation
2. Paged KV Cache + Scheduler Simulator

---

# Project 1: FlashAttention (Triton)

## Week 1: Fundamentals

### Day 1
- Review attention math, tensor shapes
- Write notes on Q/K/V flow

### Day 2
- Implement naive attention (PyTorch)

### Day 3
- Benchmark latency & memory (seq_len 128–1024)

### Day 4
- Study FlashAttention motivation (IO bottleneck)

### Day 5
- Learn online softmax (running max/sum)

### Day 6
- Implement block-wise attention (PyTorch)

### Day 7
- Validate correctness vs naive version

---

## Week 2: Triton Basics

### Day 8
- Setup Triton, run vector add

### Day 9
- Implement vector add kernel

### Day 10
- Implement matmul kernel

### Day 11
- Implement row softmax kernel

### Day 12
- Compare outputs with PyTorch

### Day 13
- Benchmark kernels

### Day 14
- Document Triton concepts

---

## Week 3: Attention Kernel

### Day 15
- Design tiling strategy

### Day 16
- Implement QK tile computation

### Day 17
- Add online softmax

### Day 18
- Add V aggregation

### Day 19
- Validate correctness

### Day 20
- Add multiple test configs

### Day 21
- Refactor & clean code

---

## Week 4: Optimization

### Day 22
- Tune BLOCK sizes

### Day 23
- Benchmark multiple seq lengths

### Day 24
- Compare memory usage

### Day 25
- Run profiler

### Day 26
- Analyze bottlenecks

### Day 27
- Write report

### Day 28
- Finalize README + resume bullet

---

# Project 2: Paged KV Cache + Scheduler

## Week 1: Basics

### Day 1
- Study decode vs prefill

### Day 2
- Design request lifecycle

### Day 3
- Implement request structure

### Day 4
- Implement naive KV cache

### Day 5
- Add allocate/append/free

### Day 6
- Create workload simulator

### Day 7
- Run baseline simulation

---

## Week 2: Paged Cache

### Day 8
- Design page/block structure

### Day 9
- Implement block pool

### Day 10
- Logical mapping

### Day 11
- Physical mapping

### Day 12
- Append token logic

### Day 13
- Free/recycle blocks

### Day 14
- Compare utilization vs naive

---

## Week 3: Scheduler

### Day 15
- Design prefix sharing

### Day 16
- Implement ref counting

### Day 17
- Copy-on-write logic

### Day 18
- Implement FCFS scheduler

### Day 19
- Add continuous batching

### Day 20
- Add prefill/decode phases

### Day 21
- Implement metrics

---

## Week 4: Experiments

### Day 22
- Design mixed workload configs

### Day 23
- Run naive baseline

### Day 24
- Run paged cache

### Day 25
- Run prefix sharing

### Day 26
- Compare page sizes

### Day 27
- Analyze results

### Day 28
- Final report + README

---

# 8-Week Combined Plan

| Week | Focus |
|------|------|
| 1 | KV cache basics |
| 2 | Paged allocation |
| 3 | Scheduler |
| 4 | Experiments |
| 5 | Attention basics |
| 6 | Triton kernels |
| 7 | Attention kernel |
| 8 | Optimization |

---

# Resume Summary

## FlashAttention
Implemented a Triton-based attention kernel with tiled computation and online softmax.

## KV Cache
Built a paged KV cache simulator with prefix sharing and scheduling.

