program fNetwork
    use FortranNetwork
    implicit none
    
    integer :: ln(3) = (/2, 4, 3/)
    character (len = 2) :: a
    integer :: i, j
    type(fNetLayer), dimension(2) :: fn
    a = "ss"
    fn = initiateNetwork(ln, a)
    do i = 1, 2
        do j = 1, ln(i)
            print *, fn(i)%weights(j,:)
        end do
        print *, ""
        print *, fn(i)%biases
        print *, fn(i)%activation
        print *, fn(i)%layerSize
        print *, ""
    end do
end program fNetwork