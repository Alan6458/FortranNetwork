program fNetwork
    use FortranNetwork
    implicit none
    
    integer :: ln(2) = (/2, 3/)
    character :: a = "s"
    integer :: i
    type(fNet) :: fn
    type(fNetOutLayer), dimension(size(ln)) :: res
    fn = initiateNetwork(ln, a)
    fn%layer(1)%weights(1,:) = (/2, 3, 5/)
    fn%layer(1)%weights(2,:) = (/6, 7, 4/)
    fn%layer(1)%biases = (/2, 3, 4/)
    do i = 1, 2
        print *, fn%layer(1)%weights(i,:)
    end do
    print *, ""
    print *, fn%layer(1)%biases
    print *, ""
    res = forwardProp(fn, (/3, 4/))
    do i = 1, 2
        print *, res(i)%outLayer
    end do
    print *, forwardPropAns(res)
end program fNetwork