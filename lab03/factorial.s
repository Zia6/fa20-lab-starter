.globl factorial

.data
n: .word 8

.text
main:
    la t0, n
    lw a0, 0(t0)
    jal ra, factorial

    addi a1, a0, 0
    addi a0, x0, 1
    ecall # Print Result

    addi a1, x0, '\n'
    addi a0, x0, 11
    ecall # Print newline

    addi a0, x0, 10
    ecall # Exit

factorial:
    addi sp, sp, -8
    sw s0, 0(sp)
    sw t0, 4(sp)
    addi s0, x0, 1
    addi t0, x0, 1
loop:
    bgt t0, a0, exit
    mul s0, s0, t0
    addi t0, t0, 1
    j loop

exit: 
    mv a0, s0
    lw t0,4(sp)
    lw s0,0(sp)
    addi sp, sp, 8
    jalr x0, x1, 0

    
