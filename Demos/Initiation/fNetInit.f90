program fNetwork
    use FortranNetwork
    implicit none
    
    integer :: ln(3) = (/2, 4, 3/)
    character :: a = "s"
    integer :: i, j
    type(fNet) :: fn
    fn = initiateNetwork(ln, a)
    do i = 1, 2
        do j = 1, ln(i)
            print *, fn%layer(i)%weights(j,:)
        end do
        print *, ""
        print *, fn%layer(i)%biases
        print *, ""
    end do
end program fNetwork