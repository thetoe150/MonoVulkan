section .data
    filename db "input.txt", 0
    alloc_size equ 4096     ; Let's allocate 4KB on the heap

section .bss
    fd_storage resq 1
    heap_start resq 1       ; To store the address of our heap memory

section .text
    global _start

_start:
    ; 1. FIND CURRENT BREAK (Start of Heap)
    ; sys_brk (12) with 0 in RDI returns the current break address
    mov rax, 12
    xor rdi, rdi
    syscall
    mov [heap_start], rax   ; This is the start of our new buffer

    ; 2. ALLOCATE SPACE
    ; Move the break forward by alloc_size
    mov rdi, rax            ; Current break address
    add rdi, alloc_size     ; New desired break address
    mov rax, 12             ; sys_brk
    syscall
    ; rax now contains the new break address if successful

    ; 3. OPEN THE FILE
    mov rax, 2              ; sys_open
    mov rdi, filename
    mov rsi, 0              ; O_RDONLY
    syscall
    test rax, rax           ; Check if FD is negative
    js exit_error
    mov [fd_storage], rax

    ; 4. READ INTO HEAP
    mov rax, 0              ; sys_read
    mov rdi, [fd_storage]
    mov rsi, [heap_start]   ; Use the address we got from brk
    mov rdx, alloc_size
    syscall
    mov r15, rax            ; Save bytes read

    ; 5. WRITE HEAP CONTENT TO STDOUT
    mov rax, 1              ; sys_write
    mov rdi, 1
    mov rsi, [heap_start]
    mov rdx, r15
    syscall

    ; 6. CLOSE AND EXIT
    mov rax, 3              ; sys_close
    mov rdi, [fd_storage]
    syscall

exit_clean:
    mov rax, 60
    xor rdi, rdi
    syscall

exit_error:
    mov rax, 60
    mov rdi, 1
    syscall

