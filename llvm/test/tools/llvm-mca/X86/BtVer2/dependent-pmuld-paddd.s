# NOTE: Assertions have been autogenerated by utils/update_mca_test_checks.py
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=500 -timeline < %s | FileCheck %s

vpmuld %xmm0, %xmm0, %xmm1
vpaddd %xmm1, %xmm1, %xmm0
vpaddd %xmm0, %xmm0, %xmm3

# CHECK:      Iterations:        500
# CHECK-NEXT: Instructions:      1500
# CHECK-NEXT: Total Cycles:      1504
# CHECK-NEXT: Dispatch Width:    2
# CHECK-NEXT: IPC:               1.00
# CHECK-NEXT: Block RThroughput: 1.5

# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK-NEXT:  1      2     1.00                        vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT:  1      1     0.50                        vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT:  1      1     0.50                        vpaddd	%xmm0, %xmm0, %xmm3

# CHECK:      Resources:
# CHECK-NEXT: [0]   - JALU0
# CHECK-NEXT: [1]   - JALU1
# CHECK-NEXT: [2]   - JDiv
# CHECK-NEXT: [3]   - JFPA
# CHECK-NEXT: [4]   - JFPM
# CHECK-NEXT: [5]   - JFPU0
# CHECK-NEXT: [6]   - JFPU1
# CHECK-NEXT: [7]   - JLAGU
# CHECK-NEXT: [8]   - JMul
# CHECK-NEXT: [9]   - JSAGU
# CHECK-NEXT: [10]  - JSTC
# CHECK-NEXT: [11]  - JVALU0
# CHECK-NEXT: [12]  - JVALU1
# CHECK-NEXT: [13]  - JVIMUL

# CHECK:      Resource pressure per iteration:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]
# CHECK-NEXT:  -      -      -      -      -     1.50   1.50    -      -      -      -     1.00   1.00   1.00

# CHECK:      Resource pressure by instruction:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]   Instructions:
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -      -      -     1.00   vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT:  -      -      -      -      -     0.50   0.50    -      -      -      -      -     1.00    -     vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -      -      -     1.00    -      -     vpaddd	%xmm0, %xmm0, %xmm3

# CHECK:      Timeline view:
# CHECK-NEXT:                     0123456789          0123
# CHECK-NEXT: Index     0123456789          0123456789

# CHECK:      [0,0]     DeeER.    .    .    .    .    .  .   vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: [0,1]     D==eER    .    .    .    .    .  .   vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT: [0,2]     .D==eER   .    .    .    .    .  .   vpaddd	%xmm0, %xmm0, %xmm3
# CHECK-NEXT: [1,0]     .D==eeER  .    .    .    .    .  .   vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: [1,1]     . D===eER .    .    .    .    .  .   vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT: [1,2]     . D====eER.    .    .    .    .  .   vpaddd	%xmm0, %xmm0, %xmm3
# CHECK-NEXT: [2,0]     .  D===eeER    .    .    .    .  .   vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: [2,1]     .  D=====eER   .    .    .    .  .   vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT: [2,2]     .   D=====eER  .    .    .    .  .   vpaddd	%xmm0, %xmm0, %xmm3
# CHECK-NEXT: [3,0]     .   D=====eeER .    .    .    .  .   vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: [3,1]     .    D======eER.    .    .    .  .   vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT: [3,2]     .    D=======eER    .    .    .  .   vpaddd	%xmm0, %xmm0, %xmm3
# CHECK-NEXT: [4,0]     .    .D======eeER   .    .    .  .   vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: [4,1]     .    .D========eER  .    .    .  .   vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT: [4,2]     .    . D========eER .    .    .  .   vpaddd	%xmm0, %xmm0, %xmm3
# CHECK-NEXT: [5,0]     .    . D========eeER.    .    .  .   vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: [5,1]     .    .  D=========eER    .    .  .   vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT: [5,2]     .    .  D==========eER   .    .  .   vpaddd	%xmm0, %xmm0, %xmm3
# CHECK-NEXT: [6,0]     .    .   D=========eeER  .    .  .   vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: [6,1]     .    .   D===========eER .    .  .   vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT: [6,2]     .    .    D===========eER.    .  .   vpaddd	%xmm0, %xmm0, %xmm3
# CHECK-NEXT: [7,0]     .    .    D===========eeER    .  .   vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: [7,1]     .    .    .D============eER   .  .   vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT: [7,2]     .    .    .D=============eER  .  .   vpaddd	%xmm0, %xmm0, %xmm3
# CHECK-NEXT: [8,0]     .    .    . D============eeER .  .   vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: [8,1]     .    .    . D==============eER.  .   vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT: [8,2]     .    .    .  D==============eER  .   vpaddd	%xmm0, %xmm0, %xmm3
# CHECK-NEXT: [9,0]     .    .    .  D==============eeER .   vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: [9,1]     .    .    .   D===============eER.   vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT: [9,2]     .    .    .   D================eER   vpaddd	%xmm0, %xmm0, %xmm3

# CHECK:      Average Wait times (based on the timeline view):
# CHECK-NEXT: [0]: Executions
# CHECK-NEXT: [1]: Average time spent waiting in a scheduler's queue
# CHECK-NEXT: [2]: Average time spent waiting in a scheduler's queue while ready
# CHECK-NEXT: [3]: Average time elapsed from WB until retire stage

# CHECK:            [0]    [1]    [2]    [3]
# CHECK-NEXT: 0.     10    8.0    0.1    0.0       vpmuldq	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: 1.     10    9.5    0.0    0.0       vpaddd	%xmm1, %xmm1, %xmm0
# CHECK-NEXT: 2.     10    10.0   0.0    0.0       vpaddd	%xmm0, %xmm0, %xmm3
