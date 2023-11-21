program fNetwork
    use FortranNetwork
    implicit none
    
    type(fNet) :: fn
    character :: a = "s"
    integer :: ls = 3
    integer :: ln(3)
    integer :: i, j
    ln = (/2, 4, 3/)
    call initiateNetwork(fn, ls, a, ln)
    do i = 1, 2
        do j = 1, ln(i)
            print *, fn%layer(i)%weights(j,:)
        end do
        print *, ""
        print *, fn%layer(i)%biases
        print *, ""
    end do
end program fNetwork