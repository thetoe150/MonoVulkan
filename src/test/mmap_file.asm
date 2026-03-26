section .data
    filename db "input.txt", 0

section .bss
    ; stat struct is 144 bytes on x86_64. 
    ; We specifically need the st_size field at offset 48.
    stat_buf resb 144 

section .text
    global _start

_start:
    ; 1. OPEN THE FILE
    mov rax, 2              ; sys_open
    mov rdi, filename
    mov rsi, 0              ; O_RDONLY
    syscall
    test rax, rax
    js exit_error
    mov r8, rax             ; Save FD in r8 for later

    ; 2. GET FILE STATUS (fstat)
    ; syscall: sys_fstat (5)
    mov rax, 5              ; sys_fstat
    mov rdi, r8             ; file descriptor
    mov rsi, stat_buf       ; pointer to struct stat
    syscall

    ; Extract file size from stat_buf + 48 (st_size is a 64-bit integer)
    mov rsi, [stat_buf + 48] ; rsi = file size (this is the 'length' for mmap)
    mov r12, rsi            ; Save size in r12 for the write syscall later

    ; 3. MMAP THE FILE
    ; syscall: sys_mmap (9)
    mov rax, 9              ; sys_mmap
    xor rdi, rdi            ; addr = NULL
    ; rsi already contains the file size from fstat
    mov rdx, 1              ; prot = PROT_READ
    mov r10, 2              ; flags = MAP_PRIVATE
    ; r8 already contains the FD
    mov r9, 0               ; offset = 0
    syscall
    
    ; rax now contains the pointer to the mapped memory
    mov r13, rax            ; Save memory pointer in r13

    ; 4. WRITE TO STDOUT
    mov rax, 1              ; sys_write
    mov rdi, 1              ; stdout
    mov rsi, r13            ; pointer to mapped memory
    mov rdx, r12            ; file size we saved in r12
    syscall

    ; 5. CLEANUP AND EXIT
    ; Optional: munmap and close, but exit handles this automatically
    mov rax, 60
    xor rdi, rdi
    syscall

exit_error:
    mov rax, 60
    mov rdi, 1
    syscall
